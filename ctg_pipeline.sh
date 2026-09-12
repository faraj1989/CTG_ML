# Default: Original preprocessing, no feature selection, all models
python ctg_pipeline.py

# With SMOTE and feature selection
python ctg_pipeline.py --preprocessing SMOTE --feature_selection

# Only train Random Forest and XGBoost
python ctg_pipeline.py --models "Random Forest" XGBoost

# Generate Word report (requires python-docx)
python ctg_pipeline.py --report