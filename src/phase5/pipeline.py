"""
pipeline.py  –  Phase 5 Root Cause Analysis pipeline orchestrator.

Execution order
---------------
  1.  Load enriched dataset (Phase 2 output)
  2.  Compute all analysis outputs (summary, trend, product, brand, region, sentiment)
  3.  Save all artefacts to models/phase5/
  4.  Export summary CSV for dissertation appendix
  5.  Generate all 9 visualisation plots
"""

import pandas as pd

from .analyser import run_analysis
from .visualisations import generate_all_plots
from .utils import (
    INPUT_CSV,
    ROOT_CAUSE_SUMMARY_PATH,
    TREND_DATA_PATH,
    PRODUCT_BREAKDOWN_PATH,
    BRAND_BREAKDOWN_PATH,
    REGION_BREAKDOWN_PATH,
    ANALYSIS_SUMMARY_CSV,
    MODELS_DIR,
    get_logger,
    save_artefact,
    timer,
)

logger = get_logger("phase5.pipeline")


def run_phase5_pipeline() -> dict:
    """
    Execute the complete Phase 5 Root Cause Analysis pipeline.

    Returns the full results dict for downstream use (Phase 6 risk scoring).
    """
    logger.info("=" * 60)
    logger.info("Phase 5 Root Cause Analysis — START")
    logger.info("=" * 60)

    # 1. Run all analysis computations
    with timer("Root cause analysis"):
        results = run_analysis(INPUT_CSV)

    # 2. Save artefacts for Phase 6
    with timer("Saving artefacts"):
        save_artefact(results["summary"],       ROOT_CAUSE_SUMMARY_PATH)
        save_artefact(results["monthly_trend"], TREND_DATA_PATH)
        save_artefact(results["product"],       PRODUCT_BREAKDOWN_PATH)
        save_artefact(results["brand_summary"], BRAND_BREAKDOWN_PATH)
        save_artefact(results["region"],        REGION_BREAKDOWN_PATH)

    # 3. Export summary CSV for dissertation
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results["summary"].to_csv(ANALYSIS_SUMMARY_CSV, index=False)
    logger.info("Summary CSV saved → %s", ANALYSIS_SUMMARY_CSV.name)

    # 4. Log key findings
    summary = results["summary"]
    logger.info("\n--- Root Cause Summary ---")
    for _, row in summary.iterrows():
        logger.info(
            "  %-40s  %5d  (%5.1f%%)  avg_sentiment=%.3f",
            row["root_cause_category"],
            row["count"],
            row["pct_of_returns"],
            row["avg_sentiment"],
        )

    trend = results["monthly_trend"]
    logger.info("\nMonthly trend: %d months × %d root causes", *trend.shape)

    # 5. Generate visualisations
    with timer("Generating visualisations"):
        generate_all_plots(results)

    logger.info("=" * 60)
    logger.info("Phase 5 Root Cause Analysis — COMPLETE")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_phase5_pipeline()
