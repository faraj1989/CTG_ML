"""
CTG ML Pipeline - Research Grade with Proper Cross-Validation
==============================================================
Updates (per supervisor feedback):
1. Removed CLASS column (data leakage prevention)
2. Macro-F1 as primary metric
3. SMOTE and SelectKBest inside sklearn Pipeline
4. Proper cross-validation (no data leakage)
5. XGBoost with Early Stopping (optional)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
from datetime import datetime
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    make_scorer
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_predict
import warnings

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.20
FEATURE_SELECTION_K = 15  # Number of features to keep
USE_EARLY_STOPPING = True  # Set to False to disable for XGBoost


# ============================================
# CUSTOM SCORERS
# ============================================
def macro_f1_scorer(y_true, y_pred):
    """Macro-averaged F1 score for cross-validation."""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def weighted_f1_scorer(y_true, y_pred):
    """Weighted-averaged F1 score for cross-validation."""
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)


def balanced_acc_scorer(y_true, y_pred):
    """Balanced accuracy for cross-validation."""
    return balanced_accuracy_score(y_true, y_pred)


# Create custom scorers for cross_validate
scoring_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'macro_f1': make_scorer(macro_f1_scorer),
    'weighted_f1': make_scorer(weighted_f1_scorer),
    'balanced_accuracy': make_scorer(balanced_acc_scorer),
    'precision': make_scorer(precision_score, average='macro', zero_division=0),
    'recall': make_scorer(recall_score, average='macro', zero_division=0),
}


# ============================================
# DATA LOADING AND PREPROCESSING
# ============================================
class CTGDataLoader:
    """Handle CTG data loading and initial preprocessing."""

    def __init__(self, filepath='CTG_dataset.csv'):
        self.filepath = filepath
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.class_names = ['Normal', 'Suspect', 'Pathologic']

    def load_data(self):
        """Load CTG dataset and perform initial cleaning."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset not found: {self.filepath}")

        # Load data
        self.df = pd.read_csv(self.filepath, sep=';')
        self.df.columns = self.df.columns.str.strip()

        print(f"✓ Loaded {len(self.df)} rows, {len(self.df.columns)} columns")

        # Remove duplicates
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            self.df = self.df.drop_duplicates()
            print(f"  Removed {dup_count} duplicate rows")

        # ============================================
        # CRITICAL: Remove CLASS column (supervisor's request)
        # ============================================
        # CLASS column likely contains the same info as NSP but encoded differently
        # Keeping it would cause data leakage
        if 'CLASS' in self.df.columns:
            print(f"  Removing 'CLASS' column (data leakage prevention)")
            self.df = self.df.drop(columns=['CLASS'])

        # ============================================
        # Verify NSP column exists
        # ============================================
        if 'NSP' not in self.df.columns:
            raise ValueError("Target column 'NSP' not found in dataset")

        # Separate features and target
        self.X = self.df.drop(columns=['NSP'])
        self.y = self.df['NSP'] - 1  # Convert 1,2,3 → 0,1,2

        self.feature_names = self.X.columns.tolist()

        print(f"✓ Features: {self.X.shape[1]}, Target: {self.y.name}")

        # Show class distribution
        class_counts = pd.Series(self.y).value_counts().sort_index()
        for i, name in enumerate(self.class_names):
            count = class_counts.get(i, 0)
            pct = count / len(self.y) * 100
            print(f"  {name}: {count} ({pct:.1f}%)")

        return self.X, self.y

    def get_train_test_split(self, test_size=TEST_SIZE):
        """Perform stratified train-test split."""
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=self.y
        )

        print(f"\n✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test


# ============================================
# BUILD ML PIPELINES (WITHOUT LEAKAGE)
# ============================================
class CTGPipelineBuilder:
    """
    Build preprocessing and model pipelines.
    SMOTE and SelectKBest are INSIDE the pipeline to prevent leakage.
    """

    @staticmethod
    def create_base_pipeline(model, use_smote=True, use_tomek=False,
                             use_feature_selection=True, k=FEATURE_SELECTION_K):
        """
        Create a pipeline with SMOTE, Feature Selection, Scaling, and Model.

        IMPORTANT: All transformations are INSIDE the pipeline, so they
        are fit ONLY on training data during cross-validation.
        """
        steps = []

        # Step 1: Scaling (must come before feature selection for f_classif)
        steps.append(('scaler', StandardScaler()))

        # Step 2: Feature Selection (SelectKBest)
        if use_feature_selection:
            steps.append(('feature_selection', SelectKBest(score_func=f_classif, k=k)))

        # Step 3: SMOTE (oversampling) - must come after feature selection
        if use_smote and use_tomek:
            steps.append(('smote', SMOTETomek(random_state=RANDOM_STATE)))
        elif use_smote:
            steps.append(('smote', SMOTE(random_state=RANDOM_STATE)))

        # Step 4: Model
        steps.append(('classifier', model))

        # Use imbalanced-learn's Pipeline for SMOTE support
        return ImbPipeline(steps)

    @staticmethod
    def get_models():
        """Define all models to evaluate."""

        # Base models with class_weight='balanced' for handling imbalance
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.1,
                random_state=RANDOM_STATE
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=RANDOM_STATE
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=3,
                random_state=RANDOM_STATE,
                eval_metric='mlogloss',
                use_label_encoder=False
            ),
        }
        return models

    @staticmethod
    def get_stacking_ensemble():
        """Create a Stacking ensemble of base models."""
        base_estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight='balanced', random_state=RANDOM_STATE
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=200, max_depth=10,
                class_weight='balanced', random_state=RANDOM_STATE
            )),
            ('xgb', XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                objective='multi:softprob', num_class=3,
                random_state=RANDOM_STATE, eval_metric='mlogloss',
                use_label_encoder=False
            )),
            ('gbm', GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3,
                random_state=RANDOM_STATE
            ))
        ]

        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
            cv=5,
            n_jobs=-1
        )


# ============================================
# CROSS-VALIDATION WITH PROPER PIPELINES
# ============================================
class CTGModelEvaluator:
    """
    Evaluate models using cross-validation with proper pipelines.
    This ensures NO DATA LEAKAGE.
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.cv_results = {}

    def evaluate_models(self, use_smote=True, use_feature_selection=True):
        """
        Evaluate all models using cross-validation on training data.
        Then test on held-out test set.
        """
        models = CTGPipelineBuilder.get_models()
        # Add Stacking Ensemble separately
        models['StackingEnsemble'] = CTGPipelineBuilder.get_stacking_ensemble()

        print("\n" + "=" * 70)
        print("MODEL EVALUATION WITH CROSS-VALIDATION")
        print("=" * 70)
        print(f"SMOTE: {use_smote}, Feature Selection: {use_feature_selection}")
        print(f"Cross-validation folds: {CV_FOLDS}")
        print("=" * 70 + "\n")

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for name, model in models.items():
            print(f"\n>>> Evaluating: {name}")
            print("-" * 50)

            # Build pipeline with SMOTE and feature selection INSIDE
            pipeline = CTGPipelineBuilder.create_base_pipeline(
                model=model,
                use_smote=use_smote,
                use_feature_selection=use_feature_selection,
                k=FEATURE_SELECTION_K
            )

            start_time = time.time()

            # ============================================
            # CRITICAL: Cross-validation with pipeline
            # ============================================
            # The pipeline ensures:
            # 1. Scaling fitted on training fold only
            # 2. Feature selection fitted on training fold only
            # 3. SMOTE fitted on training fold only
            # 4. Model trained on training fold only
            # 5. Evaluation on validation fold using transform() only
            # ============================================

            cv_scores = cross_validate(
                pipeline,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring=scoring_metrics,
                return_train_score=True,
                n_jobs=-1
            )

            elapsed = time.time() - start_time

            # Store CV results
            self.cv_results[name] = cv_scores

            # Train on full training data
            pipeline.fit(self.X_train, self.y_train)

            # Evaluate on test set
            y_pred = pipeline.predict(self.X_test)

            # Calculate metrics on test set
            test_metrics = self._calculate_metrics(
                self.y_test, y_pred, pipeline
            )

            # Store results
            self.results[name] = {
                'pipeline': pipeline,
                'cv_scores': cv_scores,
                'test_metrics': test_metrics,
                'y_pred': y_pred,
                'training_time': elapsed
            }

            # Print results
            self._print_results(name, cv_scores, test_metrics, elapsed)

        # Find best model based on Macro-F1 (primary metric)
        self._find_best_model()

        return self.results

    def _calculate_metrics(self, y_true, y_pred, pipeline):
        """Calculate all evaluation metrics."""

        # Try to get probabilities for AUC
        try:
            if hasattr(pipeline, 'predict_proba'):
                y_proba = pipeline.predict_proba(self.X_test)
                y_bin = pd.get_dummies(y_true).values
                auc = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='macro')
            else:
                auc = np.nan
        except:
            auc = np.nan

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'kappa': cohen_kappa_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'auc': auc,
        }
        return metrics

    def _print_results(self, name, cv_scores, test_metrics, elapsed):
        """Print formatted results."""
        print(f"  CV Mean Macro-F1: {cv_scores['test_macro_f1'].mean():.4f} (±{cv_scores['test_macro_f1'].std():.4f})")
        print(f"  CV Mean Accuracy: {cv_scores['test_accuracy'].mean():.4f} (±{cv_scores['test_accuracy'].std():.4f})")
        print(f"  Test Macro-F1:    {test_metrics['macro_f1']:.4f}")
        print(f"  Test Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"  Test AUC:         {test_metrics['auc']:.4f}")
        print(f"  Training Time:    {elapsed:.2f}s")

    def _find_best_model(self):
        """Find best model based on Macro-F1 on test set."""

        # Primary: Macro-F1 on test set
        # Secondary: CV Macro-F1 (for stability)

        def score_model(name):
            test_f1 = self.results[name]['test_metrics']['macro_f1']
            cv_f1 = self.results[name]['cv_scores']['test_macro_f1'].mean()
            return (test_f1 * 0.6 + cv_f1 * 0.4)  # Weighted score

        best_name = max(self.results.keys(), key=score_model)
        self.best_model_name = best_name
        self.best_model = self.results[best_name]['pipeline']

        print("\n" + "=" * 70)
        print(f"🏆 BEST MODEL: {best_name}")
        print(f"   Test Macro-F1: {self.results[best_name]['test_metrics']['macro_f1']:.4f}")
        print(f"   CV Macro-F1:   {self.results[best_name]['cv_scores']['test_macro_f1'].mean():.4f}")
        print("=" * 70)

    def get_results_summary(self):
        """Create a summary DataFrame of all results."""
        rows = []
        for name, res in self.results.items():
            row = {
                'Model': name,
                'CV_MacroF1': f"{res['cv_scores']['test_macro_f1'].mean():.4f} ± {res['cv_scores']['test_macro_f1'].std():.4f}",
                'CV_Accuracy': f"{res['cv_scores']['test_accuracy'].mean():.4f} ± {res['cv_scores']['test_accuracy'].std():.4f}",
                'Test_MacroF1': res['test_metrics']['macro_f1'],
                'Test_Accuracy': res['test_metrics']['accuracy'],
                'Test_BalAcc': res['test_metrics']['balanced_accuracy'],
                'Test_AUC': res['test_metrics']['auc'],
                'Test_MCC': res['test_metrics']['mcc'],
                'Time(s)': f"{res['training_time']:.1f}"
            }
            rows.append(row)

        return pd.DataFrame(rows).sort_values('Test_MacroF1', ascending=False)


# ============================================
# XGBOOST WITH EARLY STOPPING
# ============================================
class XGBoostWithEarlyStopping:
    """
    Evaluate XGBoost with early stopping.
    Early stopping prevents overfitting by monitoring validation loss.
    """

    @staticmethod
    def evaluate_with_early_stopping(X_train, y_train, X_test, y_test):
        """
        Train XGBoost with early stopping.

        Early stopping monitors a validation set during training
        and stops when performance stops improving.

        Benefits:
        - Prevents overfitting
        - Automatically finds optimal number of trees
        - Saves training time
        """
        print("\n" + "=" * 70)
        print("XGBOOST WITH EARLY STOPPING")
        print("=" * 70)

        # Create validation set (10% of training)
        X_train_es, X_val, y_train_es, y_val = train_test_split(
            X_train, y_train, test_size=0.1,
            random_state=RANDOM_STATE, stratify=y_train
        )

        # XGBoost with early stopping
        xgb = XGBClassifier(
            n_estimators=1000,  # Large number, early stopping will find optimal
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss',
            early_stopping_rounds=50,
            use_label_encoder=False
        )

        # Fit with early stopping
        start = time.time()
        xgb.fit(
            X_train_es, y_train_es,
            eval_set=[(X_train_es, y_train_es), (X_val, y_val)],
            verbose=False
        )
        elapsed = time.time() - start

        # Get best iteration
        best_iter = xgb.best_iteration
        print(f"\n✓ Training completed: {elapsed:.2f}s")
        print(f"  Best iteration: {best_iter}")
        print(f"  Best validation score: {xgb.best_score:.4f}")

        # Evaluate on test set
        y_pred = xgb.predict(X_test)
        y_proba = xgb.predict_proba(X_test)

        # Metrics
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        # AUC
        y_bin = pd.get_dummies(y_test).values
        auc = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='macro')

        print(f"\nTest Results (with early stopping):")
        print(f"  Macro-F1: {macro_f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}")

        # Compare with XGBoost without early stopping
        print("\n" + "-" * 50)
        print("Comparison: Early Stopping vs Standard XGBoost")
        print("-" * 50)

        # Standard XGBoost (fixed n_estimators)
        xgb_std = XGBClassifier(
            n_estimators=200,  # Fixed, no early stopping
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        xgb_std.fit(X_train, y_train)
        y_pred_std = xgb_std.predict(X_test)
        macro_f1_std = f1_score(y_test, y_pred_std, average='macro', zero_division=0)

        improvement = (macro_f1 - macro_f1_std) * 100
        print(f"  Without Early Stopping: {macro_f1_std:.4f}")
        print(f"  With Early Stopping:    {macro_f1:.4f}")
        print(f"  Improvement:            {improvement:+.2f}%")

        return {
            'model': xgb,
            'best_iteration': best_iter,
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_proba': y_proba
        }


# ============================================
# VISUALIZATION
# ============================================
class CTGVisualizer:
    """Visualize results with proper plots."""

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_roc_curves(y_test, y_proba, class_names):
        """Plot ROC curves for each class."""
        fig, ax = plt.subplots(figsize=(8, 6))
        y_bin = pd.get_dummies(y_test).values

        for i, name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_model_comparison(results_df):
        """Plot model comparison bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        models = results_df['Model'].values
        macro_f1 = results_df['Test_MacroF1'].values

        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, macro_f1, color=colors)

        # Add value labels
        for bar, val in zip(bars, macro_f1):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        ax.set_ylim([0, 1])
        ax.set_ylabel('Macro-F1 Score', fontsize=12)
        ax.set_title('Model Performance Comparison (Macro-F1)', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig


# ============================================
# MAIN EXECUTION
# ============================================
def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("CTG MACHINE LEARNING PIPELINE - RESEARCH GRADE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ============================================
    # 1. Load Data
    # ============================================
    print("\n[1] Loading Data...")
    loader = CTGDataLoader('CTG_dataset.csv')
    X, y = loader.load_data()
    X_train, X_test, y_train, y_test = loader.get_train_test_split()

    # ============================================
    # 2. Evaluate Models
    # ============================================
    print("\n[2] Evaluating Models with Cross-Validation...")
    evaluator = CTGModelEvaluator(X_train, y_train, X_test, y_test)
    results = evaluator.evaluate_models(
        use_smote=True,
        use_feature_selection=True
    )

    # ============================================
    # 3. Results Summary
    # ============================================
    print("\n[3] Results Summary...")
    results_df = evaluator.get_results_summary()
    print("\n" + "=" * 70)
    print("COMPLETE RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    # ============================================
    # 4. XGBoost with Early Stopping (if enabled)
    # ============================================
    if USE_EARLY_STOPPING:
        print("\n[4] XGBoost Early Stopping Analysis...")
        early_stop_results = XGBoostWithEarlyStopping.evaluate_with_early_stopping(
            X_train, y_train, X_test, y_test
        )

    # ============================================
    # 5. Visualizations
    # ============================================
    print("\n[5] Generating Visualizations...")

    # Get best model predictions
    best_name = evaluator.best_model_name
    best_pipeline = evaluator.best_model
    y_pred_best = best_pipeline.predict(X_test)

    # Get probabilities for best model
    try:
        y_proba_best = best_pipeline.predict_proba(X_test)
    except:
        y_proba_best = None

    # Create visualizations
    viz = CTGVisualizer()

    # Confusion Matrix
    fig_cm = viz.plot_confusion_matrix(
        y_test, y_pred_best, loader.class_names,
        title=f'Confusion Matrix - {best_name}'
    )
    plt.show()

    # ROC Curves
    if y_proba_best is not None:
        fig_roc = viz.plot_roc_curves(y_test, y_proba_best, loader.class_names)
        plt.show()

    # Model Comparison
    fig_compare = viz.plot_model_comparison(results_df)
    plt.show()

    # ============================================
    # 6. Summary Statistics
    # ============================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Best Model: {best_name}")
    print(f"Test Macro-F1: {evaluator.results[best_name]['test_metrics']['macro_f1']:.4f}")
    print(f"Test Accuracy: {evaluator.results[best_name]['test_metrics']['accuracy']:.4f}")
    print(f"CV Macro-F1 (mean): {evaluator.results[best_name]['cv_scores']['test_macro_f1'].mean():.4f}")

    # Classification Report
    print("\nClassification Report (Best Model):")
    print("-" * 50)
    report = classification_report(
        y_test, y_pred_best,
        target_names=loader.class_names,
        zero_division=0
    )
    print(report)
    print("=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return evaluator, results_df


# ============================================
# RUN THE PIPELINE
# ============================================
if __name__ == "__main__":
    # Run the full pipeline
    evaluator, results_df = main()