#!/usr/bin/env python3
"""
run_phase5.py  –  Entry-point for the Phase 5 Root Cause Analysis pipeline.

Usage
-----
    python run_phase5.py

Must be run AFTER run_phase2.py (requires the NLP-enriched dataset).

Outputs
-------
  models/phase5/root_cause_summary.joblib
  models/phase5/monthly_trend.joblib
  models/phase5/product_breakdown.joblib
  models/phase5/brand_breakdown.joblib
  models/phase5/region_breakdown.joblib
  models/phase5/root_cause_analysis_summary.csv
  outputs/phase5/plots/  — 9 dissertation plots
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase5.pipeline import run_phase5_pipeline

if __name__ == "__main__":
    run_phase5_pipeline()
