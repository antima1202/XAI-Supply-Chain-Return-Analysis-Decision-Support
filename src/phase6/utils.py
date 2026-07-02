"""
utils.py  –  Shared utilities for the Phase 6 Risk Assessment pipeline.
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

MODELS_DIR_P5 = PROJECT_ROOT / "models" / "phase5"
MODELS_DIR    = PROJECT_ROOT / "models" / "phase6"
PLOTS_DIR     = PROJECT_ROOT / "outputs" / "phase6" / "plots"

# Phase 5 inputs
ROOT_CAUSE_SUMMARY_PATH = MODELS_DIR_P5 / "root_cause_summary.joblib"
TREND_DATA_PATH         = MODELS_DIR_P5 / "monthly_trend.joblib"
REGION_BREAKDOWN_PATH   = MODELS_DIR_P5 / "region_breakdown.joblib"
BRAND_BREAKDOWN_PATH    = MODELS_DIR_P5 / "brand_breakdown.joblib"

# Phase 6 outputs
RISK_SCORES_PATH    = MODELS_DIR / "risk_scores.joblib"
RISK_SCORES_CSV     = MODELS_DIR / "risk_scores.csv"

# ---------------------------------------------------------------------------
# Root cause labels — consistent ordering
# ---------------------------------------------------------------------------

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

ROOT_CAUSE_COLOURS = {
    "Customer Preference":             "#4A90D9",
    "Logistics / Delivery":            "#E8563A",
    "Manufacturing / Quality Control": "#F5A623",
    "Product Listing / Information":   "#7ED321",
    "Supplier Issues":                 "#9B59B6",
    "Warehouse / Packaging":           "#1ABC9C",
}

# ---------------------------------------------------------------------------
# Risk level thresholds (score 0–100)
# ---------------------------------------------------------------------------

RISK_LEVELS = {
    "Critical": (75, 100),
    "High":     (50, 74),
    "Medium":   (25, 49),
    "Low":      (0,  24),
}

RISK_LEVEL_COLOURS = {
    "Critical": "#D94F3D",
    "High":     "#E8A838",
    "Medium":   "#4A90D9",
    "Low":      "#7ED321",
}

# Domain-defined operational impact weights (0–100)
# Reflects cost, complexity, and strategic risk of each root cause.
# Justification in the dissertation:
#   Manufacturing/QC  — affects entire batches; may require product recall
#   Supplier Issues   — strategic vendor risk; slow and costly to resolve
#   Warehouse/Packaging — operational process fix; medium complexity
#   Logistics/Delivery — carrier relationship; quick wins available
#   Product Listing   — content update; low cost, fast fix
#   Customer Preference — change-of-mind; no operational fix possible
OPERATIONAL_IMPACT = {
    "Manufacturing / Quality Control": 90,
    "Supplier Issues":                 85,
    "Warehouse / Packaging":           70,
    "Logistics / Delivery":            65,
    "Product Listing / Information":   50,
    "Customer Preference":             25,
}


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str = "phase6") -> logging.Logger:
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
