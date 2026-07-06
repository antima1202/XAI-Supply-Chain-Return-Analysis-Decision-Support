"""
utils.py  –  Shared utilities for the Phase 5 Root Cause Analysis layer.

Mirrors the Phase 2/3/4 utils pattern for consistency across the project.
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
MODELS_DIR         = PROJECT_ROOT / "models" / "phase5"
PLOTS_DIR          = PROJECT_ROOT / "outputs" / "phase5" / "plots"

# Input — Phase 2 enriched dataset
INPUT_CSV = DATA_PROCESSED_DIR / "fashion_returns_dataset_nlp.csv"

# Phase 5 artefact outputs
ROOT_CAUSE_SUMMARY_PATH = MODELS_DIR / "root_cause_summary.joblib"
TREND_DATA_PATH         = MODELS_DIR / "monthly_trend.joblib"
PRODUCT_BREAKDOWN_PATH  = MODELS_DIR / "product_breakdown.joblib"
BRAND_BREAKDOWN_PATH    = MODELS_DIR / "brand_breakdown.joblib"
REGION_BREAKDOWN_PATH   = MODELS_DIR / "region_breakdown.joblib"
ANALYSIS_SUMMARY_CSV    = MODELS_DIR / "root_cause_analysis_summary.csv"

# Root cause labels — consistent with Phase 3
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

# Colour palette — one colour per root cause (consistent across all plots)
ROOT_CAUSE_COLOURS = {
    "Customer Preference":             "#4A90D9",
    "Logistics / Delivery":            "#E8563A",
    "Manufacturing / Quality Control": "#F5A623",
    "Product Listing / Information":   "#7ED321",
    "Supplier Issues":                 "#9B59B6",
    "Warehouse / Packaging":           "#1ABC9C",
}


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str = "phase5") -> logging.Logger:
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
