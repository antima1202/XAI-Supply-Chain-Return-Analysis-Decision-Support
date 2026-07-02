"""
utils.py  –  Shared utilities for the Phase 4 SHAP explainability pipeline.
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR_P3  = PROJECT_ROOT / "models" / "phase3"
MODELS_DIR     = PROJECT_ROOT / "models" / "phase4"
PLOTS_DIR      = PROJECT_ROOT / "outputs" / "phase4" / "plots"

# Phase 3 inputs
BEST_MODEL_PATH      = MODELS_DIR_P3 / "best_model.joblib"
BEST_MODEL_NAME_PATH = MODELS_DIR_P3 / "best_model_name.joblib"
X_TEST_PATH          = MODELS_DIR_P3 / "X_test.joblib"
Y_TEST_PATH          = MODELS_DIR_P3 / "y_test.joblib"
FEATURE_NAMES_PATH   = MODELS_DIR_P3 / "feature_names.joblib"
LABEL_ENCODER_PATH   = MODELS_DIR_P3 / "label_encoder.joblib"

# Phase 4 outputs
SHAP_EXPLAINER_PATH  = MODELS_DIR / "shap_explainer.joblib"
SHAP_VALUES_PATH     = MODELS_DIR / "shap_values.joblib"

ROOT_CAUSE_CLASSES = [
    "Customer Preference",
    "Logistics / Delivery",
    "Manufacturing / Quality Control",
    "Product Listing / Information",
    "Supplier Issues",
    "Warehouse / Packaging",
]

ROOT_CAUSE_SHORT = {
    "Customer Preference":             "Cust. Pref.",
    "Logistics / Delivery":            "Logistics",
    "Manufacturing / Quality Control": "Mfg / QC",
    "Product Listing / Information":   "Prod. Listing",
    "Supplier Issues":                 "Supplier",
    "Warehouse / Packaging":           "Warehouse",
}


def get_logger(name: str = "phase4") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s  [%(levelname)s]  %(name)s – %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def save_artefact(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    get_logger().info("Saved artefact → %s", path.name)


def load_artefact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Artefact not found: {path}")
    return joblib.load(path)


@contextmanager
def timer(label: str):
    log = get_logger()
    log.info("START  %s", label)
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    log.info("DONE   %s  (%.1f s)", label, elapsed)
