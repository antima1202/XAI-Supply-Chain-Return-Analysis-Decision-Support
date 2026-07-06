#!/usr/bin/env python3
"""
run_phase4.py  –  Entry-point for the Phase 4 SHAP explainability pipeline.

Usage
-----
    python run_phase4.py

Must be run AFTER run_phase3.py — loads the best Phase 3 model.

Outputs
-------
  models/phase4/shap_explainer.joblib
  models/phase4/shap_values.joblib
  outputs/phase4/plots/shap_global_importance.png
  outputs/phase4/plots/shap_beeswarm_<class>.png   (6 plots)
  outputs/phase4/plots/shap_beeswarm_summary.png
  outputs/phase4/plots/shap_waterfall_<class>.png  (6 plots)
  outputs/phase4/plots/shap_class_importance_heatmap.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase4.pipeline import run_phase4_pipeline

if __name__ == "__main__":
    run_phase4_pipeline()
