#!/usr/bin/env python3
"""
run_phase2.py  –  Entry-point script for the Phase 2 NLP pipeline.

Usage
-----
    python run_phase2.py

Run this from the project root (the dissertation/ folder).
The script adds the project root to sys.path automatically so the
src.phase2 package is importable without installing the project.

Outputs
-------
  data/processed/fashion_returns_dataset_nlp.csv   — enriched dataset
  models/phase2/tfidf_vectorizer.joblib            — fitted TF-IDF vectorizer
  models/phase2/lda_model.joblib                   — trained LDA model
  models/phase2/lda_dictionary.joblib              — Gensim dictionary
  models/phase2/lda_corpus.joblib                  — BoW corpus
  models/phase2/topic_label_map.joblib             — topic ID → label map
  outputs/phase2/plots/                            — all dissertation figures
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'src' is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2.pipeline import run_phase2_pipeline  # noqa: E402

if __name__ == "__main__":
    run_phase2_pipeline()
