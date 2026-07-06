#!/usr/bin/env python3
"""
run_phase3.py  –  Entry-point for the Phase 3 ML pipeline.

Usage
-----
    python run_phase3.py

Run from the project root (the files/ folder).

Outputs
-------
  models/phase3/logistic_regression.joblib
  models/phase3/random_forest.joblib
  models/phase3/xgboost.joblib
  models/phase3/best_model.joblib
  models/phase3/best_model_name.joblib
  models/phase3/label_encoder.joblib
  models/phase3/feature_names.joblib
  models/phase3/scaler.joblib
  models/phase3/X_test.joblib
  models/phase3/y_test.joblib
  models/phase3/evaluation_results.joblib
  outputs/phase3/plots/  — confusion matrices, ROC curves, feature importance,
                           model comparison
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.pipeline import run_phase3_pipeline

if __name__ == "__main__":
    run_phase3_pipeline()
