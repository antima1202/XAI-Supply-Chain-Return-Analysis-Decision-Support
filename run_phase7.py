#!/usr/bin/env python3
"""
run_phase7.py  –  Entry-point for the Phase 7 LLM Recommendation Engine.

Usage
-----
    python run_phase7.py

Prerequisites
-------------
  - Ollama must be running locally
  - Model must be pulled: ollama pull llama3.2
  - run_phase5.py must have been run (provides analysis artefacts)
  - run_phase6.py must have been run (provides risk scores)
  - pip install ollama

Outputs
-------
  models/phase7/recommendations.joblib
  outputs/phase7/recommendations.json
  outputs/phase7/executive_summary.txt
  outputs/phase7/financial_impact.png
  outputs/phase7/recommendation_summary.png
  outputs/phase7/savings_waterfall.png
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase7.pipeline import run_phase7_pipeline

if __name__ == "__main__":
    run_phase7_pipeline()