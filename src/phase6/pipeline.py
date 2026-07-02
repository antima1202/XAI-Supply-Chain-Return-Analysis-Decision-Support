"""
pipeline.py  –  Phase 6 Risk Assessment pipeline orchestrator.

Execution order
---------------
  1.  Load Phase 5 artefacts (summary + trend)
  2.  Compute composite risk scores (frequency + impact + trend)
  3.  Assign risk levels and priority ranking
  4.  Save risk scores as joblib and CSV
  5.  Generate all 6 risk visualisation plots
  6.  Log a dissertation-ready risk summary
"""

import pandas as pd

from .risk_scorer import compute_risk_scores
from .visualisations import generate_all_plots
from .utils import (
    ROOT_CAUSE_SUMMARY_PATH,
    TREND_DATA_PATH,
    RISK_SCORES_PATH,
    RISK_SCORES_CSV,
    MODELS_DIR,
    get_logger,
    load_artefact,
    save_artefact,
    timer,
)

logger = get_logger("phase6.pipeline")


def run_phase6_pipeline() -> pd.DataFrame:
    """
    Execute the complete Phase 6 Risk Assessment pipeline.

    Returns
    -------
    risk_df : pd.DataFrame
        Risk scores with all components — used by Phase 7 Recommendation Engine.
    """
    logger.info("=" * 60)
    logger.info("Phase 6 Risk Assessment — START")
    logger.info("=" * 60)

    # 1. Load Phase 5 inputs
    with timer("Loading Phase 5 artefacts"):
        summary = load_artefact(ROOT_CAUSE_SUMMARY_PATH)
        trend   = load_artefact(TREND_DATA_PATH)
    logger.info("Loaded summary (%d rows) and trend (%d months)", len(summary), len(trend))

    # 2. Compute risk scores
    with timer("Computing risk scores"):
        risk_df = compute_risk_scores(summary, trend)

    # 3. Save artefacts
    with timer("Saving artefacts"):
        save_artefact(risk_df, RISK_SCORES_PATH)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        risk_df.to_csv(RISK_SCORES_CSV, index=False)
        logger.info("Risk scores CSV saved → %s", RISK_SCORES_CSV.name)

    # 4. Log dissertation-ready summary
    logger.info("\n" + "=" * 60)
    logger.info("RISK ASSESSMENT RESULTS")
    logger.info("=" * 60)
    logger.info("%-5s  %-40s  %6s  %6s  %8s  %-10s",
                "Rank", "Root Cause", "Score", "Level", "Returns", "% Share")
    logger.info("-" * 85)
    for _, row in risk_df.iterrows():
        logger.info(
            "P%-4d  %-40s  %6.1f  %-10s  %6d  %5.1f%%",
            row["priority"],
            row["root_cause_category"],
            row["risk_score"],
            row["risk_level"],
            row["count"],
            row["pct_of_returns"],
        )

    # 5. Generate plots
    with timer("Generating risk visualisations"):
        generate_all_plots(risk_df)

    logger.info("=" * 60)
    logger.info("Phase 6 Risk Assessment — COMPLETE")
    logger.info("=" * 60)

    return risk_df


if __name__ == "__main__":
    run_phase6_pipeline()
