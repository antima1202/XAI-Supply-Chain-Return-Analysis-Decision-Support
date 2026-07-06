"""
utils.py  –  Shared utilities for the Phase 3 ML pipeline.

Mirrors the Phase 2 utils pattern so every phase is consistent.
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import joblib

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR         = PROJECT_ROOT / "models" / "phase3"
PLOTS_DIR          = PROJECT_ROOT / "outputs" / "phase3" / "plots"

INPUT_CSV = DATA_PROCESSED_DIR / "fashion_returns_dataset_nlp.csv"

# Artefact paths
LABEL_ENCODER_PATH  = MODELS_DIR / "label_encoder.joblib"
FEATURE_NAMES_PATH  = MODELS_DIR / "feature_names.joblib"
SCALER_PATH         = MODELS_DIR / "scaler.joblib"
LR_MODEL_PATH       = MODELS_DIR / "logistic_regression.joblib"
RF_MODEL_PATH       = MODELS_DIR / "random_forest.joblib"
XGB_MODEL_PATH      = MODELS_DIR / "xgboost.joblib"
BEST_MODEL_PATH     = MODELS_DIR / "best_model.joblib"
BEST_MODEL_NAME_PATH = MODELS_DIR / "best_model_name.joblib"
X_TEST_PATH         = MODELS_DIR / "X_test.joblib"
Y_TEST_PATH         = MODELS_DIR / "y_test.joblib"
EVAL_RESULTS_PATH   = MODELS_DIR / "evaluation_results.joblib"

# Root cause class labels (consistent ordering used throughout)
ROOT_CAUSE_CLASSES = [
    "Customer Preference",
    "Logistics / Delivery",
    "Manufacturing / Quality Control",
    "Product Listing / Information",
    "Supplier Issues",
    "Warehouse / Packaging",
]

# Short labels for plots
ROOT_CAUSE_SHORT = {
    "Customer Preference":            "Cust. Pref.",
    "Logistics / Delivery":           "Logistics",
    "Manufacturing / Quality Control": "Mfg / QC",
    "Product Listing / Information":  "Prod. Listing",
    "Supplier Issues":                "Supplier",
    "Warehouse / Packaging":          "Warehouse",
}


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str = "phase3") -> logging.Logger:
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


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_artefact(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    get_logger().info("Saved artefact → %s", path.name)


def load_artefact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Artefact not found: {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str):
    log = get_logger()
    log.info("START  %s", label)
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    log.info("DONE   %s  (%.1f s)", label, elapsed)
