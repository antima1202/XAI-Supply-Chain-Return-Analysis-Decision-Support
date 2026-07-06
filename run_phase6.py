#!/usr/bin/env python3
"""
run_phase6.py  –  Entry-point for the Phase 6 Risk Assessment pipeline.

Usage
-----
    python run_phase6.py

Must be run AFTER run_phase5.py (requires Phase 5 artefacts).

Outputs
-------
  models/phase6/risk_scores.joblib       — full risk DataFrame
  models/phase6/risk_scores.csv          — dissertation appendix table
  outputs/phase6/plots/
      risk_priority_ranking.png
      risk_matrix.png
      risk_score_breakdown.png
      risk_level_donut.png
      risk_vs_sentiment.png
      risk_score_table.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase6.pipeline import run_phase6_pipeline

if __name__ == "__main__":
    run_phase6_pipeline()
