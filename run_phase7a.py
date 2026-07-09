#!/usr/bin/env python3
"""
run_phase7a.py  –  Phase 7a Operational Optimisation (RL + LP).

Usage
-----
    python run_phase7a.py

    # Custom budget:
    python run_phase7a.py --budget 200000

    # Custom cost per return:
    python run_phase7a.py --budget 150000 --cost-per-return 25

Prerequisites
-------------
  run_phase6.py must have been run (provides risk scores).
  pip install pulp  (for LP solver)

Outputs
-------
  models/phase7a/rl_results.joblib           — RL optimal policy
  models/phase7a/lp_results.joblib           — LP baseline solution
  models/phase7a/optimisation_results.joblib — full results + comparison
  models/phase7a/optimisation_results.csv    — summary table
  outputs/phase7a/plots/
      rl_training_curve.png
      rl_convergence.png
      rl_vs_lp_comparison.png
      budget_allocation_rl.png
      budget_allocation_lp.png
      roi_ranking.png
      epsilon_decay.png
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase7a.pipeline import run_phase7a_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 7a — Operational Optimisation (RL + LP)"
    )
    parser.add_argument("--budget",          type=float, default=150_000,
                        help="Total intervention budget in £ (default: 150000)")
    parser.add_argument("--cost-per-return", type=float, default=22.0,
                        help="Reverse logistics cost per return in £ (default: 22)")
    parser.add_argument("--episodes",        type=int,   default=2000,
                        help="RL training episodes (default: 2000)")
    parser.add_argument("--seed",            type=int,   default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_phase7a_pipeline(
        budget=args.budget,
        cost_per_return=args.cost_per_return,
        n_episodes=args.episodes,
        random_seed=args.seed,
    )
