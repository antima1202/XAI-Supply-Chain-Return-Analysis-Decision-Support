"""
utils.py  –  Shared utilities for Phase 7a Operational Optimisation (RL + LP).

All business-specific parameters are defined here as defaults.
The dashboard will override these with user inputs at runtime.

Configurable parameters (dashboard inputs)
------------------------------------------
  BUDGET                — total intervention budget (£)
  COST_PER_RETURN       — reverse logistics cost per return (£)
  INTERVENTION_COSTS    — minimum cost to implement each intervention (£)
  REDUCTION_POTENTIAL   — expected return reduction % per root cause
  RISK_WEIGHTS          — weights for frequency / impact / trend scoring
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import joblib

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

PROJECT_ROOT  = Path(__file__).resolve().parents[2]

MODELS_DIR_P6 = PROJECT_ROOT / "models" / "phase6"
MODELS_DIR    = PROJECT_ROOT / "models" / "phase7a"
PLOTS_DIR     = PROJECT_ROOT / "outputs" / "phase7a" / "plots"

# Inputs from Phase 6
RISK_SCORES_PATH = MODELS_DIR_P6 / "risk_scores.joblib"

# Phase 7a outputs
RL_RESULTS_PATH   = MODELS_DIR / "rl_results.joblib"
LP_RESULTS_PATH   = MODELS_DIR / "lp_results.joblib"
OPT_RESULTS_PATH  = MODELS_DIR / "optimisation_results.joblib"
OPT_RESULTS_CSV   = MODELS_DIR / "optimisation_results.csv"

# ---------------------------------------------------------------------------
# Root cause labels
# ---------------------------------------------------------------------------

ROOT_CAUSE_CLASSES = [
    "Product Listing / Information",
    "Warehouse / Packaging",
    "Manufacturing / Quality Control",
    "Supplier Issues",
    "Customer Preference",
    "Logistics / Delivery",
]

ROOT_CAUSE_SHORT = {
    "Product Listing / Information":   "Prod. Listing",
    "Warehouse / Packaging":           "Warehouse",
    "Manufacturing / Quality Control": "Mfg / QC",
    "Supplier Issues":                 "Supplier",
    "Customer Preference":             "Cust. Pref.",
    "Logistics / Delivery":            "Logistics",
}

ROOT_CAUSE_COLOURS = {
    "Product Listing / Information":   "#7ED321",
    "Warehouse / Packaging":           "#1ABC9C",
    "Manufacturing / Quality Control": "#F5A623",
    "Supplier Issues":                 "#9B59B6",
    "Customer Preference":             "#4A90D9",
    "Logistics / Delivery":            "#E8563A",
}

# ---------------------------------------------------------------------------
# Configurable business parameters (dashboard will override these)
# ---------------------------------------------------------------------------

# Total intervention budget (£)
DEFAULT_BUDGET = 150_000.0

# Reverse logistics + admin cost per return (£)
DEFAULT_COST_PER_RETURN = 22.0

# Minimum investment required to make each intervention viable (£)
# Below this threshold the intervention is not worth implementing
DEFAULT_INTERVENTION_COSTS = {
    "Product Listing / Information":   30_000.0,
    "Warehouse / Packaging":           40_000.0,
    "Manufacturing / Quality Control": 35_000.0,
    "Supplier Issues":                 45_000.0,
    "Customer Preference":             20_000.0,
    "Logistics / Delivery":            25_000.0,
}

# Expected return reduction % if intervention is fully funded
# These are realistic industry estimates for fashion e-commerce
DEFAULT_REDUCTION_POTENTIAL = {
    "Product Listing / Information":   0.35,  # 35%
    "Warehouse / Packaging":           0.30,  # 30%
    "Manufacturing / Quality Control": 0.40,  # 40%
    "Supplier Issues":                 0.45,  # 45%
    "Customer Preference":             0.15,  # 15%
    "Logistics / Delivery":            0.25,  # 25%
}

# RL training parameters
DEFAULT_N_EPISODES    = 2000   # training episodes
DEFAULT_LEARNING_RATE = 0.1    # Q-learning alpha
DEFAULT_DISCOUNT      = 0.95   # future reward discount (gamma)
DEFAULT_EPSILON       = 1.0    # initial exploration rate
DEFAULT_EPSILON_DECAY = 0.995  # exploration decay per episode
DEFAULT_EPSILON_MIN   = 0.01   # minimum exploration rate


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str = "phase7a") -> logging.Logger:
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
