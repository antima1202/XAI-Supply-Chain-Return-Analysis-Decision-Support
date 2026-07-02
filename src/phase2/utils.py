"""
utils.py  –  Shared utilities for the Phase 2 NLP pipeline.

Provides:
  - Project-wide path constants
  - A pre-configured logger
  - Serialisation helpers (joblib save / load)
  - A lightweight timer context manager
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import joblib

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # dissertation/

DATA_RAW_DIR       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR         = PROJECT_ROOT / "models" / "phase2"
PLOTS_DIR          = PROJECT_ROOT / "outputs" / "phase2" / "plots"

INPUT_CSV    = DATA_RAW_DIR       / "fashion_returns_dataset.csv"
OUTPUT_CSV   = DATA_PROCESSED_DIR / "fashion_returns_dataset_nlp.csv"

# Artefact filenames (saved inside MODELS_DIR)
TFIDF_PATH      = MODELS_DIR / "tfidf_vectorizer.joblib"
LDA_MODEL_PATH  = MODELS_DIR / "lda_model.joblib"
LDA_DICT_PATH   = MODELS_DIR / "lda_dictionary.joblib"
LDA_CORPUS_PATH = MODELS_DIR / "lda_corpus.joblib"
TOPIC_MAP_PATH  = MODELS_DIR / "topic_label_map.joblib"

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str = "phase2") -> logging.Logger:
    """Return a consistently formatted logger for Phase 2 modules."""
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
    """Persist any Python object to *path* using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    get_logger().info("Saved artefact → %s", path.name)


def load_artefact(path: Path):
    """Load a joblib-serialised artefact from *path*."""
    if not path.exists():
        raise FileNotFoundError(f"Artefact not found: {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str):
    """Context manager that logs elapsed time for a named block."""
    log = get_logger()
    log.info("START  %s", label)
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    log.info("DONE   %s  (%.1f s)", label, elapsed)
