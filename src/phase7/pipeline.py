"""
pipeline.py  –  Phase 7 LLM Recommendation Engine orchestrator.

Execution order
---------------
  1.  Load Phase 5 (analysis) and Phase 6 (risk scores) artefacts
  2.  Initialise Gemini LLM client
  3.  Generate 6 stakeholder-specific recommendations (one per root cause)
  4.  Generate board-level executive summary
  5.  Save all outputs (joblib + JSON + plain text)
  6.  Generate 3 financial visualisation plots
"""

from .recommendation_engine import run_recommendation_engine
from .visualisations import generate_all_plots
from .utils import get_logger, timer

logger = get_logger("phase7.pipeline")


def run_phase7_pipeline(api_key: str | None = None) -> dict:
    """
    Execute the complete Phase 7 LLM Recommendation Engine.

    Parameters
    ----------
    api_key : str | None
        Gemini API key. If None, reads from GEMINI_API_KEY environment variable.
        If neither is set, runs in fallback mode with structured templates.

    Returns
    -------
    dict with keys:
        'recommendations'  — dict of 6 stakeholder recommendations
        'executive_summary'— board-level summary string
    """
    logger.info("=" * 60)
    logger.info("Phase 7 LLM Recommendation Engine — START")
    logger.info("=" * 60)

    # Run recommendation engine (loads data, calls LLM, saves outputs)
    with timer("Full recommendation generation"):
        results = run_recommendation_engine(api_key=api_key)

    # Generate visualisations
    with timer("Generating visualisations"):
        generate_all_plots(results["recommendations"])

    logger.info("=" * 60)
    logger.info("Phase 7 LLM Recommendation Engine — COMPLETE")
    logger.info("%d recommendations generated", len(results["recommendations"]))
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_phase7_pipeline()
