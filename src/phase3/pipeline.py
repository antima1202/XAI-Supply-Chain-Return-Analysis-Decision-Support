"""
pipeline.py  –  Phase 3 ML pipeline orchestrator.

Execution order
---------------
  1.  Feature engineering (load, encode, SMOTE, split, scale)
  2.  Train Logistic Regression, Random Forest, XGBoost
  3.  Evaluate all three models
  4.  Select best model by Macro F1
  5.  Save best model + test set for Phase 4 (SHAP)
"""

import pandas as pd

from .feature_engineering import prepare_data
from .models import train_all_models
from .evaluation import evaluate_all_models
from .utils import (
    BEST_MODEL_PATH,
    BEST_MODEL_NAME_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    get_logger,
    save_artefact,
    timer,
)

logger = get_logger("phase3.pipeline")


def run_phase3_pipeline() -> dict:
    """
    Execute the complete Phase 3 ML pipeline end-to-end.

    Returns
    -------
    dict with keys:
      'models'       — all three trained models
      'results'      — evaluation metrics per model
      'best_name'    — name of best-performing model
      'best_model'   — the best model object
      'data'         — feature engineering outputs
    """
    logger.info("=" * 60)
    logger.info("Phase 3 ML Pipeline — START")
    logger.info("=" * 60)

    # 1. Feature engineering
    with timer("Feature engineering"):
        data = prepare_data()

    # 2. Train models
    with timer("Model training"):
        models = train_all_models(data)

    # 3. Evaluate all models
    with timer("Model evaluation"):
        results, best_name = evaluate_all_models(models, data)

    # 4. Save best model and test set for Phase 4
    best_model = models[best_name]
    save_artefact(best_model,      BEST_MODEL_PATH)
    save_artefact(best_name,       BEST_MODEL_NAME_PATH)
    save_artefact(data["X_test"],  X_TEST_PATH)
    save_artefact(data["y_test"],  Y_TEST_PATH)

    logger.info("=" * 60)
    logger.info("Phase 3 ML Pipeline — COMPLETE")
    logger.info("Best model: %s", best_name)
    logger.info(
        "Accuracy=%.4f  Macro-F1=%.4f  Macro-AUC=%.4f",
        results[best_name]["accuracy"],
        results[best_name]["macro_f1"],
        results[best_name]["macro_auc"],
    )
    logger.info("=" * 60)

    return {
        "models":     models,
        "results":    results,
        "best_name":  best_name,
        "best_model": best_model,
        "data":       data,
    }


if __name__ == "__main__":
    run_phase3_pipeline()
