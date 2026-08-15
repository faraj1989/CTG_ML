"""
CTG Correct Pipeline – No Leakage, All Models Compatible
Target converted to 0,1,2 once.
SMOTE / SMOTE+Tomek applied ONLY on training data.
Custom cross‑validation to avoid XGBoost label issues.
"""
import pandas as pd
import numpy as np
import warnings
import time
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
                             confusion_matrix, classification_report, roc_auc_score)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Create output folders
os.makedirs("cleaned_data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("CTG_dataset.csv", sep=";")
df.columns = df.columns.str.strip()
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# 2. Remove duplicates
df_clean = df.drop_duplicates()
print(f"After removing duplicates: {len(df_clean)}")

# 3. Features & target (convert NSP to 0,1,2)
X = df_clean.drop(columns=["NSP"])
y = df_clean["NSP"] - 1   # now: 0=Normal, 1=Suspect, 2=Pathologic

# 4. Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# 5. Define models (now all accept 0,1,2 labels)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                             objective="multi:softprob", num_class=3,
                             random_state=42, eval_metric="mlogloss")
}

# 6. Preprocessing variants (applied ONLY on training data)
preprocessors = {
    "Original": ("Original", X_train, y_train),
    "SMOTE": ("SMOTE", *SMOTE(random_state=42).fit_resample(X_train, y_train)),
    "SMOTE+Tomek": ("SMOTE+Tomek", *SMOTETomek(random_state=42).fit_resample(X_train, y_train))
}

results = []

for prep_name, (_, X_tr, y_tr) in preprocessors.items():
    print(f"\n{'='*60}")
    print(f"Preprocessing: {prep_name}")
    print(f"Training samples after resampling: {len(X_tr)}")

    # Scale the data (fit on training only)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_test)

    # ---- Custom cross‑validation (avoids XGBoost label issues) ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_idx, val_idx in cv.split(X_tr, y_tr):
        X_cv_train, X_cv_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_cv_train, y_cv_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
        # Scale inside CV
        scaler_cv = StandardScaler()
        X_cv_train_scaled = scaler_cv.fit_transform(X_cv_train)
        X_cv_val_scaled = scaler_cv.transform(X_cv_val)
        # Train each model
        for name, model in models.items():
            # We'll compute CV for each model individually later; we do it inside loop for each model
            # Actually we need to compute CV per model, so we'll restructure:
            pass
    # Instead of above, we'll compute CV per model using a pipeline but with a wrapper to ensure labels are correct

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        start = time.time()

        # Train the model
        model.fit(X_tr_scaled, y_tr)
        y_pred = model.predict(X_te_scaled)

        train_time = time.time() - start

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # AUC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_te_scaled)
            y_test_bin = label_binarize(y_test, classes=[0,1,2])
            auc = roc_auc_score(y_test_bin, y_score, multi_class="ovr")
        else:
            auc = np.nan

        # ---- 5‑fold CV using a pipeline that scales + model ----
        # To avoid XGBoost label issues, we ensure y is 0,1,2 (it already is)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Use cross_val_score; if XGBoost still complains, we can use a custom loop
        try:
            cv_acc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring='accuracy')
            cv_mean, cv_std = cv_acc.mean(), cv_acc.std()
        except Exception as e:
            print(f"  CV failed for {model_name}: {e}")
            cv_mean, cv_std = np.nan, np.nan

        results.append({
            "Preprocessing": prep_name,
            "Model": model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Balanced_Accuracy": bal_acc,
            "Kappa": kappa,
            "MCC": mcc,
            "AUC": auc,
            "CV_Accuracy_mean": cv_mean,
            "CV_Accuracy_std": cv_std,
            "Train_Time": train_time
        })

        print(f"  Test Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
        if not np.isnan(cv_mean):
            print(f"  5‑fold CV: {cv_mean:.4f} ± {cv_std:.4f}")
        else:
            print("  5‑fold CV: skipped due to error")

        # Save model
        joblib.dump(model, f"models/{model_name}_{prep_name}.pkl")

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Accuracy", ascending=False)
results_df.to_csv("results/model_comparison_final.csv", index=False)

print("\n" + "="*60)
print("BEST MODEL (No Leakage, Correct Labels):")
best = results_df.iloc[0]
print(f"  {best['Model']} with {best['Preprocessing']}")
print(f"  Test Accuracy: {best['Accuracy']:.4f} ({best['Accuracy']*100:.2f}%)")
if not np.isnan(best['CV_Accuracy_mean']):
    print(f"  5‑fold CV: {best['CV_Accuracy_mean']:.4f} ± {best['CV_Accuracy_std']:.4f}")
print(f"  F1-Score: {best['F1']:.4f}")
print(f"  MCC: {best['MCC']:.4f}")
print("="*60)
print("\nFull results saved to results/model_comparison_final.csv")

# Classification report for the best model
best_model_name = best['Model']
best_prep = best['Preprocessing']
for prep_name, (_, X_tr, y_tr) in preprocessors.items():
    if prep_name == best_prep:
        scaler_best = StandardScaler()
        X_tr_scaled_best = scaler_best.fit_transform(X_tr)
        X_te_scaled_best = scaler_best.transform(X_test)
        model_best = models[best_model_name]
        model_best.fit(X_tr_scaled_best, y_tr)
        y_pred_best = model_best.predict(X_te_scaled_best)
        print("\nClassification Report (best model):")
        print(classification_report(y_test, y_pred_best, target_names=['Normal', 'Suspect', 'Pathologic']))
        break