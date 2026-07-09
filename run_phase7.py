#!/usr/bin/env python3
"""
run_phase7.py  –  Entry-point for the Phase 7 LLM Recommendation Engine.

Usage
-----
    # Step 1: Set your Gemini API key (get free key from aistudio.google.com)

    # Windows PowerShell:
    $env:GEMINI_API_KEY = "your-key-here"

    # macOS / Linux:
    export GEMINI_API_KEY="your-key-here"

    # Step 2: Run the pipeline
    python run_phase7.py

    # Alternative: pass key directly (not recommended for shared environments)
    python run_phase7.py --api-key your-key-here

Prerequisites
-------------
  - run_phase5.py must have been run (provides analysis artefacts)
  - run_phase6.py must have been run (provides risk scores)
  - pip install google-generativeai

Outputs
-------
  models/phase7/recommendations.joblib   — full recommendations dict
  outputs/phase7/recommendations.json    — human-readable JSON
  outputs/phase7/executive_summary.txt   — board-level summary text
  outputs/phase7/financial_impact.png    — cost vs saving bar chart
  outputs/phase7/recommendation_summary.png — summary table
  outputs/phase7/savings_waterfall.png   — cumulative savings waterfall
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase7.pipeline import run_phase7_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 7 — LLM Recommendation Engine"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (alternatively set GEMINI_API_KEY env variable)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_phase7_pipeline(api_key=args.api_key)
