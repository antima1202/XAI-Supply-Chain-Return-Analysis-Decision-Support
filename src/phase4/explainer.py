"""
explainer.py  –  SHAP explainability for the Phase 3 best model.

Design
------
Uses shap.TreeExplainer which is optimised for tree-based models
(XGBoost, Random Forest). If the best model is Logistic Regression,
falls back to shap.LinearExplainer.

SHAP values for a 6-class problem are a 3D array:
  shape = (n_samples, n_features, n_classes)

Each value represents how much a feature pushed the prediction
toward or away from each class for that sample.

For visualisation we use:
  - Per-class SHAP values for global plots
  - Predicted class SHAP values for individual explanations
"""

import numpy as np
import pandas as pd
import shap

from .utils import (
    SHAP_EXPLAINER_PATH,
    SHAP_VALUES_PATH,
    get_logger,
    save_artefact,
    timer,
)

logger = get_logger("phase4.explainer")


def build_explainer(model, X_background: pd.DataFrame):
    """
    Build a SHAP explainer appropriate for the model type.

    Parameters
    ----------
    model        : trained sklearn/xgboost model
    X_background : background dataset for explainer initialisation
                   (use a sample of training data — 200 rows is sufficient)

    Returns
    -------
    shap.Explainer
    """
    model_type = type(model).__name__
    logger.info("Building SHAP explainer for model type: %s", model_type)

    if model_type in ("XGBClassifier", "RandomForestClassifier",
                      "GradientBoostingClassifier", "DecisionTreeClassifier"):
        explainer = shap.TreeExplainer(model)
        logger.info("Using TreeExplainer")
    elif model_type == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_background)
        logger.info("Using LinearExplainer")
    else:
        # Universal fallback — slower but works for any model
        explainer = shap.KernelExplainer(
            model.predict_proba,
            shap.sample(X_background, 100),
        )
        logger.info("Using KernelExplainer (fallback)")

    save_artefact(explainer, SHAP_EXPLAINER_PATH)
    return explainer


def compute_shap_values(
    explainer,
    X_test: pd.DataFrame,
    sample_size: int = 1000,
) -> np.ndarray:
    """
    Compute SHAP values for the test set.

    For speed and memory efficiency, uses a sample of the test set
    (1000 rows by default — sufficient for stable global plots).

    Parameters
    ----------
    explainer   : fitted SHAP explainer
    X_test      : test feature matrix
    sample_size : number of rows to compute SHAP values for

    Returns
    -------
    shap_values : np.ndarray
        For tree models: shape (n_samples, n_features, n_classes)
        For linear models: shape (n_classes, n_samples, n_features)
    X_sample    : pd.DataFrame — the sampled rows used
    """
    # Sample for speed
    n = min(sample_size, len(X_test))
    X_sample = X_test.sample(n=n, random_state=42).reset_index(drop=True)
    logger.info("Computing SHAP values for %d samples …", n)

    with timer("SHAP value computation"):
        shap_values = explainer.shap_values(X_sample)

    # Normalise to consistent shape: (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        # LinearExplainer returns list of arrays, one per class
        shap_values = np.stack(shap_values, axis=-1)   # (n, f, c)
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 2:
            # Single class output — expand
            shap_values = shap_values[:, :, np.newaxis]
        # TreeExplainer for multiclass: already (n, f, c)

    logger.info("SHAP values shape: %s", shap_values.shape)
    save_artefact({"shap_values": shap_values, "X_sample": X_sample}, SHAP_VALUES_PATH)

    return shap_values, X_sample
