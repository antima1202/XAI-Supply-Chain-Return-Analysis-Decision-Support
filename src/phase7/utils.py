"""
utils.py  –  Shared utilities for the Phase 7 LLM Recommendation Engine.
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR_P5  = PROJECT_ROOT / "models" / "phase5"
MODELS_DIR_P6  = PROJECT_ROOT / "models" / "phase6"
MODELS_DIR_P7A = PROJECT_ROOT / "models" / "phase7a"
MODELS_DIR     = PROJECT_ROOT / "models" / "phase7"
OUTPUTS_DIR    = PROJECT_ROOT / "outputs" / "phase7"

RISK_SCORES_PATH        = MODELS_DIR_P6  / "risk_scores.joblib"
ROOT_CAUSE_SUMMARY_PATH = MODELS_DIR_P5  / "root_cause_summary.joblib"
BRAND_BREAKDOWN_PATH    = MODELS_DIR_P5  / "brand_breakdown.joblib"
REGION_BREAKDOWN_PATH   = MODELS_DIR_P5  / "region_breakdown.joblib"
OPT_RESULTS_PATH        = MODELS_DIR_P7A / "optimisation_results.joblib"
TREND_DATA_PATH         = MODELS_DIR_P5 / "monthly_trend.joblib"

RECOMMENDATIONS_PATH   = MODELS_DIR  / "recommendations.joblib"
RECOMMENDATIONS_JSON   = OUTPUTS_DIR / "recommendations.json"
EXECUTIVE_SUMMARY_PATH = OUTPUTS_DIR / "executive_summary.txt"

ROOT_CAUSE_CLASSES = [
    "Customer Preference",
    "Logistics / Delivery",
    "Manufacturing / Quality Control",
    "Product Listing / Information",
    "Supplier Issues",
    "Warehouse / Packaging",
]

STAKEHOLDER_MAP = {
    "Product Listing / Information":   "Marketing & Content Team",
    "Warehouse / Packaging":           "Warehouse Operations Manager",
    "Manufacturing / Quality Control": "Quality Assurance Team",
    "Supplier Issues":                 "Procurement & Supplier Manager",
    "Logistics / Delivery":            "Logistics & Carrier Manager",
    "Customer Preference":             "Marketing & UX Team",
}

COST_PER_RETURN = 22.0

REDUCTION_POTENTIAL = {
    "Product Listing / Information":   0.35,
    "Warehouse / Packaging":           0.30,
    "Manufacturing / Quality Control": 0.40,
    "Supplier Issues":                 0.45,
    "Logistics / Delivery":            0.25,
    "Customer Preference":             0.15,
}


def get_logger(name: str = "phase7") -> logging.Logger:
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

# Colour palette for visualisations
ROOT_CAUSE_COLOURS = {
    "Customer Preference":             "#4A90D9",
    "Logistics / Delivery":            "#E8563A",
    "Manufacturing / Quality Control": "#F5A623",
    "Product Listing / Information":   "#7ED321",
    "Supplier Issues":                 "#9B59B6",
    "Warehouse / Packaging":           "#1ABC9C",
}
