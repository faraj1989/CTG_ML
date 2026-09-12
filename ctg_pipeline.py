#!/usr/bin/env python3
"""
CTG ML Pipeline – Command Line Version
=======================================
- Loads CTG_dataset.csv (semicolon separated)
- Removes duplicates, converts NSP to 0,1,2
- Stratified 80/20 split
- Preprocessing: Original, SMOTE, or SMOTE+Tomek
- Optional feature selection (SelectKBest, top 15)
- Trains: Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, Stacking Ensemble
- Outputs metrics, classification report, confusion matrix, feature importance
- Saves results (CSV, plots) and optionally a Word report
"""

import os
import sys
import argparse
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score, cohen_kappa_score,
                             matthews_corrcoef, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# Optional Word report
try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

warnings.filterwarnings("ignore")

# Create output directories
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ----------------------------------------------------------------------
def load_and_prepare_data(filepath):
    """Load CSV, remove duplicates, split, and return X_train, X_test, y_train, y_test."""
    df = pd.read_csv(filepath, sep=";")
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Remove duplicates
    initial = len(df)
    df = df.drop_duplicates()
    removed = initial - len(df)
    if removed:
        print(f"Removed {removed} duplicate rows")
    print(f"After removing duplicates: {len(df)}")

    # Target: NSP -> 0,1,2 (original values 1,2,3)
    X = df.drop(columns=["NSP"])
    y = df["NSP"] - 1

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Class distribution
    nsp = df["NSP"].value_counts()
    print(f"Class distribution — Normal: {nsp.get(1,0)}, Suspect: {nsp.get(2,0)}, Pathologic: {nsp.get(3,0)}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def apply_preprocessing(X_train, y_train, X_test, prep_type):
    """Apply SMOTE or SMOTE+Tomek to training data, scale features."""
    scaler = StandardScaler()
    if prep_type == "SMOTE":
        X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    elif prep_type == "SMOTE+Tomek":
        X_res, y_res = SMOTETomek(random_state=42).fit_resample(X_train, y_train)
    else:  # Original
        X_res, y_res = X_train, y_train

    print(f"Training samples after resampling: {len(X_res)}")

    # Scale
    X_res_scaled = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test)
    return X_res_scaled, y_res, X_test_scaled, scaler


def select_features(X_train, y_train, X_test, feature_names, k=15):
    """Select top k features using f_classif."""
    selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    print(f"Feature selection: keeping {len(selected_features)} features")
    return X_train_selected, X_test_selected, selector, selected_features


def get_models():
    """Return dictionary of model instances."""
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                      random_state=42, n_jobs=-1)),
        ('et', ExtraTreesClassifier(n_estimators=300, class_weight='balanced',
                                    random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                              subsample=0.8, colsample_bytree=0.8,
                              objective="multi:softprob", num_class=3,
                              random_state=42, eval_metric="mlogloss"))
    ]
    models = {
        "Random Forest": base_estimators[0][1],
        "Extra Trees": base_estimators[1][1],
        "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                        random_state=42),
        "XGBoost": base_estimators[2][1],
        "Stacking Ensemble": StackingClassifier(estimators=base_estimators,
                                                final_estimator=LogisticRegression(max_iter=1000),
                                                cv=5, n_jobs=-1)
    }
    return models


def train_and_evaluate(models, X_train, y_train, X_test, y_test, model_names=None):
    """Train each model and return results list."""
    if model_names is None:
        model_names = list(models.keys())

    results = []
    for name in model_names:
        model = models[name]
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # AUC (one-vs-rest)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
            y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
            auc = roc_auc_score(y_test_bin, y_score, multi_class="ovr")
        else:
            auc = np.nan

        print(f"  Test Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Balanced_Accuracy": bal_acc,
            "Kappa": kappa,
            "MCC": mcc,
            "AUC": auc,
            "Train_Time": train_time,
            "y_pred": y_pred,
            "model": model
        })
    return results


def print_summary(results):
    """Print best model and detailed classification report."""
    if not results:
        return
    best = max(results, key=lambda x: x["Accuracy"])
    print("\n" + "=" * 60)
    print("BEST MODEL:")
    print(f"  {best['Model']}")
    print(f"  Test Accuracy: {best['Accuracy']:.4f} ({best['Accuracy']*100:.2f}%)")
    print(f"  F1-Score: {best['F1']:.4f}")
    print(f"  MCC: {best['MCC']:.4f}")
    print("=" * 60)

    print("\nClassification Report (best model):")
    report = classification_report(y_test, best["y_pred"],
                                   target_names=['Normal', 'Suspect', 'Pathologic'],
                                   zero_division=0)
    print(report)
    return best


def save_plots(best, X_test, y_test, feature_names, selected_features=None):
    """Save confusion matrix and feature importance plots."""
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, best["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=['Normal', 'Suspect', 'Pathologic'],
                yticklabels=['Normal', 'Suspect', 'Pathologic'])
    ax.set_title(f"Confusion Matrix – {best['Model']}\nAccuracy: {best['Accuracy']:.2%}")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance (if available)
    model = best["model"]
    if hasattr(model, "feature_importances_"):
        # Use selected features if provided, else all features
        if selected_features is not None:
            feat_names = selected_features
        else:
            feat_names = feature_names
        imp = model.feature_importances_
        # If length mismatch, use the first len(feat_names) or pad
        if len(imp) != len(feat_names):
            # In stacking, feature_importances_ not defined; skip
            pass
        else:
            feat_df = pd.DataFrame({"feature": feat_names, "importance": imp})
            feat_df = feat_df.sort_values("importance", ascending=True).tail(15)
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(feat_df)))
            ax.barh(feat_df["feature"], feat_df["importance"], color=colors)
            ax.set_xlabel("Importance")
            ax.set_title(f"Top 15 Features – {best['Model']}")
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.savefig("plots/feature_importance.png", dpi=150)
            plt.close()

    # ROC curves (for best model if probability available)
    if hasattr(model, "predict_proba"):
        # Need X_test scaled – we don't have it here, but we can use the passed X_test
        # However, best model already trained on scaled data; we need to use the same scaling.
        # Since we don't pass scaler, we will assume X_test is already scaled.
        # We'll just compute ROC using the test data that was used for prediction.
        # We'll rely on the fact that X_test is already scaled.
        # But we need to ensure we have the correct X_test. We'll pass it.
        # We'll just skip ROC plot for simplicity; you can add later.
        pass


def generate_word_report(results, best, feature_names, selected_features, preprocessing, fs_enabled):
    """Generate a Word document report if python-docx is available."""
    if not DOCX_AVAILABLE:
        print("python-docx not installed. Word report skipped.")
        return

    doc = Document()
    title = doc.add_heading("CTG ML Pipeline Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"Preprocessing: {preprocessing}")
    doc.add_paragraph(f"Feature Selection: {'Enabled (top 15)' if fs_enabled else 'Disabled'}")

    # Results table
    doc.add_heading("Model Performance", level=1)
    table = doc.add_table(rows=len(results)+1, cols=10)
    table.style = "Table Grid"
    headers = ["Model", "Acc", "Prec", "Recall", "F1", "BalAcc", "Kappa", "MCC", "AUC", "Time"]
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
    for i, r in enumerate(sorted(results, key=lambda x: x["Accuracy"], reverse=True)):
        row = table.rows[i+1]
        row.cells[0].text = r["Model"]
        row.cells[1].text = f"{r['Accuracy']:.4f}"
        row.cells[2].text = f"{r['Precision']:.4f}"
        row.cells[3].text = f"{r['Recall']:.4f}"
        row.cells[4].text = f"{r['F1']:.4f}"
        row.cells[5].text = f"{r['Balanced_Accuracy']:.4f}"
        row.cells[6].text = f"{r['Kappa']:.4f}"
        row.cells[7].text = f"{r['MCC']:.4f}"
        row.cells[8].text = f"{r['AUC']:.4f}"
        row.cells[9].text = f"{r['Train_Time']:.2f}s"

    # Best model details
    doc.add_heading("Best Model", level=1)
    doc.add_paragraph(f"Model: {best['Model']}")
    doc.add_paragraph(f"Accuracy: {best['Accuracy']:.2%}")
    doc.add_paragraph(f"F1-Score: {best['F1']:.4f}")
    doc.add_paragraph(f"MCC: {best['MCC']:.4f}")

    # Confusion matrix image
    doc.add_heading("Confusion Matrix", level=1)
    if os.path.exists("plots/confusion_matrix.png"):
        doc.add_picture("plots/confusion_matrix.png", width=Inches(5))

    # Classification report
    doc.add_heading("Classification Report", level=1)
    report = classification_report(y_test, best["y_pred"],
                                   target_names=['Normal', 'Suspect', 'Pathologic'],
                                   zero_division=0)
    p = doc.add_paragraph()
    run = p.add_run(report)
    run.font.name = "Courier New"
    run.font.size = Pt(9)

    # Save
    filename = f"results/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    doc.save(filename)
    print(f"Word report saved to {filename}")


def save_model(best, scaler, selector, feature_names, selected_features):
    """Persist the best model plus the preprocessing needed to use it on new data."""
    bundle = {
        "model": best["model"],
        "model_name": best["Model"],
        "scaler": scaler,
        "selector": selector,
        "feature_names": feature_names,
        "selected_features": selected_features,
        "class_labels": ["Normal", "Suspect", "Pathologic"],
    }
    safe_name = best["Model"].replace(" ", "_")
    filename = f"models/{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(bundle, filename)
    print(f"Best model saved to {filename}")


def save_results_csv(results):
    """Save all metrics to a CSV file."""
    df = pd.DataFrame(results)
    df = df.drop(columns=["y_pred", "model"], errors="ignore")
    filename = f"results/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CTG ML Pipeline")
    parser.add_argument("--data", type=str, default="CTG_dataset.csv",
                        help="Path to CTG dataset CSV (semicolon separated)")
    parser.add_argument("--preprocessing", type=str, choices=["Original", "SMOTE", "SMOTE+Tomek"],
                        default="Original", help="Preprocessing method")
    parser.add_argument("--feature_selection", action="store_true",
                        help="Enable feature selection (top 15)")
    parser.add_argument("--models", nargs="+",
                        default=["Random Forest", "Extra Trees", "AdaBoost",
                                 "Gradient Boosting", "XGBoost", "Stacking Ensemble"],
                        help="List of models to train")
    parser.add_argument("--report", action="store_true",
                        help="Generate Word report (requires python-docx)")
    args = parser.parse_args()

    # 1. Load data
    global X_train, X_test, y_train, y_test, feature_names
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(args.data)

    # 2. Preprocessing
    X_train_proc, y_train_proc, X_test_proc, scaler = apply_preprocessing(
        X_train, y_train, X_test, args.preprocessing
    )

    # 3. Feature selection
    selector = None
    selected_features = None
    if args.feature_selection:
        X_train_proc, X_test_proc, selector, selected_features = select_features(
            X_train_proc, y_train_proc, X_test_proc, feature_names, k=15
        )

    # 4. Get models
    all_models = get_models()
    # Filter by requested models
    models = {name: all_models[name] for name in args.models if name in all_models}
    if not models:
        print("No valid models selected. Exiting.")
        sys.exit(1)

    # 5. Train and evaluate
    results = train_and_evaluate(models, X_train_proc, y_train_proc, X_test_proc, y_test,
                                 model_names=list(models.keys()))

    # 6. Print summary
    best = print_summary(results)

    # 7. Save plots
    # We need to pass X_test_proc and y_test
    # Also need selected features for importance plot
    save_plots(best, X_test_proc, y_test, feature_names, selected_features)

    # 8. Save CSV results
    save_results_csv(results)

    # 8b. Persist best model + preprocessing for later inference
    save_model(best, scaler, selector, feature_names, selected_features)

    # 9. Generate Word report if requested
    if args.report:
        generate_word_report(results, best, feature_names, selected_features,
                             args.preprocessing, args.feature_selection)

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()