# CTG Machine Learning Pipeline

Machine learning pipeline for classifying fetal state (Normal / Suspect / Pathologic) from
Cardiotocography (CTG) recordings, using the UCI CTG dataset (`CTG_dataset.csv`).

## Files

| File | Purpose |
|---|---|
| `ctg_pipeline.py` | Main pipeline: load data, preprocess, train models, evaluate, save results. |
| `ctg_pipeline.sh` | Example commands for running the pipeline with different options. |
| `ctg_dashboard.py` | Streamlit dashboard: browse run results/plots and run live predictions with a saved model. |
| `CTG_dataset.csv` | Source dataset (semicolon-separated, 2126 records, 21 features + `NSP` target). |
| `requirements.txt` | Pinned Python dependencies. |
| `old scripts/` | Archived earlier iterations of the pipeline, kept for reference only — not maintained. |

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

## Running the pipeline

```bash
# Default: Original preprocessing, no feature selection, all 6 models
python ctg_pipeline.py

# With SMOTE oversampling and top-15 feature selection
python ctg_pipeline.py --preprocessing SMOTE --feature_selection

# Train only specific models
python ctg_pipeline.py --models "Random Forest" XGBoost

# Also generate a Word (.docx) report
python ctg_pipeline.py --report
```

See `ctg_pipeline.sh` for more examples.

### CLI options

| Flag | Values | Default | Description |
|---|---|---|---|
| `--data` | path | `CTG_dataset.csv` | Path to the dataset CSV. |
| `--preprocessing` | `Original`, `SMOTE`, `SMOTE+Tomek` | `Original` | Class-imbalance handling on the training split only. |
| `--feature_selection` | flag | off | Keep only the top 15 features by ANOVA F-value (`SelectKBest`). |
| `--models` | one or more model names | all 6 | Subset of models to train (see below). |
| `--report` | flag | off | Also generate a Word document report in `results/`. |

## Pipeline steps

1. **Load & clean** — read `CTG_dataset.csv`, drop duplicate rows, map `NSP` (1/2/3) to
   class labels (Normal/Suspect/Pathologic → 0/1/2).
2. **Split** — stratified 80/20 train/test split (`random_state=42`, fixed for reproducibility).
3. **Preprocess** — `StandardScaler` always; optionally `SMOTE` or `SMOTE+Tomek` resampling,
   fit on the training split only (no leakage into the test set).
4. **Feature selection** *(optional)* — `SelectKBest(f_classif)`, top 15 features, fit on
   training data only.
5. **Train & evaluate** — trains the selected models and computes accuracy, precision,
   recall, F1, balanced accuracy, Cohen's kappa, MCC, and one-vs-rest AUC on the held-out
   test set.
6. **Report** — prints a classification report for the best model (currently selected by
   highest accuracy), saves a confusion matrix and feature-importance plot, a metrics CSV,
   the trained model bundle, and optionally a Word report.

### Models

- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- XGBoost
- Stacking Ensemble (RF + Extra Trees + XGBoost → Logistic Regression meta-learner)

## Outputs

Every run is timestamped and additive — nothing is overwritten between runs.

- `results/metrics_<timestamp>.csv` — accuracy/precision/recall/F1/MCC/AUC/etc. per model.
- `results/report_<timestamp>.docx` — Word report (only with `--report`).
- `plots/confusion_matrix.png` — confusion matrix for the best model (overwritten each run).
- `plots/feature_importance.png` — top-15 feature importances for the best model, if
  available (overwritten each run).
- `models/<ModelName>_<timestamp>.pkl` — `joblib` bundle containing the best trained model,
  its fitted scaler, feature selector (if used), feature names, and class labels, so it can
  be reloaded for inference without retraining:

  ```python
  import joblib
  bundle = joblib.load("models/XGBoost_20260821_200425.pkl")
  model, scaler = bundle["model"], bundle["scaler"]
  ```

## Reference results (default run, `Original` preprocessing, all models)

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Random Forest | 0.9456 | 0.9455 | 0.9889 |
| Extra Trees | 0.9338 | 0.9299 | 0.9837 |
| AdaBoost | 0.8889 | 0.8771 | 0.9558 |
| Gradient Boosting | 0.9598 | 0.9576 | 0.9859 |
| **XGBoost (best)** | **0.9622** | **0.9607** | **0.9909** |
| Stacking Ensemble | 0.9574 | 0.9560 | 0.9924 |

XGBoost per-class (test set, n=423): Normal recall 0.99, Suspect recall 0.78,
Pathologic recall 0.97. Suspect is the weakest class — the main target for future
improvement (see below).

## Dashboard

A Streamlit app for browsing results and running live predictions — useful for a defense
demo without touching the command line.

```bash
streamlit run ctg_dashboard.py
```

- **Results Overview** — pick any `results/metrics_*.csv` run and view the model comparison
  table (best model highlighted) plus the confusion matrix / feature importance plots from
  the most recent run.
- **Live Prediction** — pick a saved model from `models/`, adjust feature sliders (defaulted
  to the dataset mean), and get a predicted class with a probability bar chart.

## Known limitations / next steps

- Best model is currently selected by raw accuracy, which favors the majority (Normal)
  class; consider ranking by F1-macro or minority-class recall instead.
- Single fixed train/test split — results are not cross-validated.
- No hyperparameter tuning; all models use fixed, hand-picked parameters.
- The ROC-curve section of `save_plots()` is not yet implemented.
