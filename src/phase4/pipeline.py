"""
pipeline.py  –  Phase 4 SHAP explainability pipeline orchestrator.

Execution order
---------------
  1.  Load best Phase 3 model, test set, feature names, label encoder
  2.  Build SHAP TreeExplainer (or LinearExplainer)
  3.  Compute SHAP values on test set sample (1000 rows)
  4.  Generate all 5 SHAP visualisation types
  5.  Save explainer and SHAP values for Phase 5+ use
"""

import numpy as np
import pandas as pd

from .explainer import build_explainer, compute_shap_values
from .visualisations import generate_all_shap_plots
from .utils import (
    BEST_MODEL_PATH,
    BEST_MODEL_NAME_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    FEATURE_NAMES_PATH,
    LABEL_ENCODER_PATH,
    get_logger,
    load_artefact,
    timer,
)

logger = get_logger("phase4.pipeline")


def run_phase4_pipeline() -> dict:
    """
    Execute the complete Phase 4 SHAP pipeline end-to-end.

    Returns
    -------
    dict with keys:
      'explainer'      — fitted SHAP explainer
      'shap_values'    — np.ndarray (n_samples, n_features, n_classes)
      'X_sample'       — sampled test features used for SHAP
      'y_test_sample'  — true labels for the sample
      'y_pred_sample'  — predicted labels for the sample
    """
    logger.info("=" * 60)
    logger.info("Phase 4 SHAP Pipeline — START")
    logger.info("=" * 60)

    # 1. Load Phase 3 artefacts
    with timer("Loading Phase 3 artefacts"):
        model        = load_artefact(BEST_MODEL_PATH)
        model_name   = load_artefact(BEST_MODEL_NAME_PATH)
        X_test       = load_artefact(X_TEST_PATH)
        y_test       = load_artefact(Y_TEST_PATH)
        feature_names = load_artefact(FEATURE_NAMES_PATH)
        label_encoder = load_artefact(LABEL_ENCODER_PATH)

    logger.info("Best model loaded: %s", model_name)
    logger.info("Test set: %d rows × %d features", *X_test.shape)

    # Ensure X_test has correct column names
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    # 2. Build SHAP explainer
    # Use a background sample from the test set (200 rows sufficient)
    X_background = X_test.sample(n=min(200, len(X_test)), random_state=42)
    with timer("Building SHAP explainer"):
        explainer = build_explainer(model, X_background)

    # 3. Compute SHAP values
    with timer("Computing SHAP values"):
        shap_values, X_sample = compute_shap_values(
            explainer, X_test, sample_size=1000
        )

    # 4. Get predictions for the sample
    y_test_arr = np.array(y_test)
    sample_indices = X_test.index.get_indexer(X_sample.index) \
        if hasattr(X_test, 'index') else list(range(len(X_sample)))

    # Align y_test to the sample
    y_test_reset = pd.Series(y_test).reset_index(drop=True)
    X_test_reset = X_test.reset_index(drop=True)

    sample_mask = X_test_reset.index.isin(
        X_test.reset_index(drop=True).sample(
            n=min(1000, len(X_test)), random_state=42
        ).index
    )
    y_test_sample = y_test_reset[sample_mask].values[:len(X_sample)]

    # Predict on the sample
    if model_name == "Logistic Regression":
        # LR needs scaled features — use unscaled with predict for consistency
        y_pred_sample = model.predict(X_sample)
    else:
        y_pred_sample = model.predict(X_sample)

    logger.info(
        "Sample: %d rows  |  SHAP values shape: %s",
        len(X_sample), shap_values.shape,
    )

    # 5. Generate all plots
    with timer("Generating SHAP visualisations"):
        generate_all_shap_plots(
            explainer=explainer,
            shap_values=shap_values,
            X_sample=X_sample,
            y_test_sample=y_test_sample,
            y_pred_sample=y_pred_sample,
            feature_names=feature_names,
        )

    logger.info("=" * 60)
    logger.info("Phase 4 SHAP Pipeline — COMPLETE")
    logger.info("=" * 60)

    return {
        "explainer":     explainer,
        "shap_values":   shap_values,
        "X_sample":      X_sample,
        "y_test_sample": y_test_sample,
        "y_pred_sample": y_pred_sample,
    }


if __name__ == "__main__":
    run_phase4_pipeline()
