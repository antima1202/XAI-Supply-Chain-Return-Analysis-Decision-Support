"""
recommendation_engine.py  –  Core recommendation generation for Phase 7.

Updated to consume Phase 7a (RL + LP) optimisation outputs.
The Gemini LLM now receives:
  - Risk scores (Phase 6)
  - Brand and region analysis (Phase 5)
  - RL optimal budget allocation (Phase 7a)
  - LP baseline comparison (Phase 7a)
  - Convergence gap (Phase 7a)

This makes every recommendation grounded in the optimised budget decision,
not just the risk assessment alone.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .llm_client     import GeminiClient
from .prompt_builder import per_root_cause_prompt, executive_summary_prompt
from .utils import (
    RISK_SCORES_PATH,
    ROOT_CAUSE_SUMMARY_PATH,
    BRAND_BREAKDOWN_PATH,
    REGION_BREAKDOWN_PATH,
    OPT_RESULTS_PATH,
    STAKEHOLDER_MAP,
    COST_PER_RETURN,
    REDUCTION_POTENTIAL,
    RECOMMENDATIONS_PATH,
    RECOMMENDATIONS_JSON,
    EXECUTIVE_SUMMARY_PATH,
    get_logger,
    load_artefact,
    save_artefact,
    timer,
)

logger = get_logger("phase7.engine")


# ---------------------------------------------------------------------------
# Load all inputs — Phase 5, 6, and 7a
# ---------------------------------------------------------------------------

def load_inputs() -> dict:
    """
    Load all required artefacts from Phases 5, 6, and 7a.

    Phase 7a (optimisation) is optional — if the artefact does not exist
    the pipeline runs without it and logs a warning.
    """
    logger.info("Loading Phase 5 and 6 artefacts …")
    risk_df = load_artefact(RISK_SCORES_PATH)
    summary = load_artefact(ROOT_CAUSE_SUMMARY_PATH)
    brand   = load_artefact(BRAND_BREAKDOWN_PATH)
    region  = load_artefact(REGION_BREAKDOWN_PATH)

    logger.info("Loaded risk scores (%d rows)", len(risk_df))
    logger.info("Loaded brand breakdown (%d rows)", len(brand))
    logger.info("Loaded region breakdown (%d rows)", len(region))

    # Phase 7a — RL + LP optimisation results (optional)
    opt_results = None
    if OPT_RESULTS_PATH.exists():
        try:
            opt_results = load_artefact(OPT_RESULTS_PATH)
            rl_saving   = opt_results.get("rl_policy", {}).get("total_saving", 0)
            lp_saving   = opt_results.get("lp_result", {}).get("total_saving", 0)
            gap         = opt_results.get("comparison", {}).get("convergence_gap_pct", 0)
            logger.info(
                "Loaded Phase 7a optimisation — RL saving: £%.0f | LP saving: £%.0f | gap: %.1f%%",
                rl_saving, lp_saving, gap,
            )
        except Exception as exc:
            logger.warning("Could not load Phase 7a optimisation results: %s", exc)
    else:
        logger.warning(
            "Phase 7a optimisation results not found at %s. "
            "Run run_phase7a.py first for richer recommendations.",
            OPT_RESULTS_PATH,
        )

    return {
        "risk_df":     risk_df,
        "summary":     summary,
        "brand":       brand,
        "region":      region,
        "opt_results": opt_results,
    }


# ---------------------------------------------------------------------------
# Generate per-root-cause recommendations
# ---------------------------------------------------------------------------

def generate_recommendations(
    inputs: dict,
    client: GeminiClient,
) -> dict:
    """
    Generate one LLM recommendation per root cause.

    The prompt now includes Phase 7a RL + LP optimisation context so
    Gemini generates recommendations grounded in the budget allocation
    decision — not risk scores alone.

    Returns a dict keyed by root_cause_category.
    """
    risk_df     = inputs["risk_df"]
    brand       = inputs["brand"]
    region      = inputs["region"]
    opt_results = inputs["opt_results"]

    if opt_results:
        logger.info("Phase 7a optimisation context INCLUDED in prompts")
    else:
        logger.info("Phase 7a optimisation context NOT available — prompts use risk scores only")

    recommendations = {}

    for _, row in risk_df.iterrows():
        rc = row["root_cause_category"]
        logger.info("Generating recommendation for: %s (P%d)", rc, int(row["priority"]))

        # Build prompt — now passes opt_results
        prompt = per_root_cause_prompt(row, brand, region, opt_results)

        # Call LLM
        with timer(f"LLM call — {rc}"):
            rec_text = client.generate(prompt, label=rc)

        # Core financial figures
        cost_impact       = row["count"] * COST_PER_RETURN
        reduction_pct     = REDUCTION_POTENTIAL.get(rc, 0.3)
        returns_prevented = int(row["count"] * reduction_pct)
        potential_saving  = returns_prevented * COST_PER_RETURN

        # RL-specific allocation figures (from Phase 7a)
        rl_allocation = None
        if opt_results:
            rl_policy = opt_results.get("rl_policy", {})
            rl_allocation = next(
                (a for a in rl_policy.get("allocation", []) if a["root_cause"] == rc),
                None,
            )

        recommendations[rc] = {
            "root_cause":          rc,
            "priority":            int(row["priority"]),
            "risk_score":          float(row["risk_score"]),
            "risk_level":          row["risk_level"],
            "stakeholder":         STAKEHOLDER_MAP.get(rc, "Operations Team"),
            "returns":             int(row["count"]),
            "pct_share":           float(row["pct_of_returns"]),
            "avg_sentiment":       float(row["avg_sentiment"]),
            "cost_impact":         round(cost_impact, 2),
            "reduction_pct":       round(reduction_pct * 100, 1),
            "returns_prevented":   returns_prevented,
            "potential_saving":    round(potential_saving, 2),
            # RL optimisation fields
            "rl_funded":           rl_allocation["invested"] if rl_allocation else None,
            "rl_budget_allocated": rl_allocation["cost"] if rl_allocation else None,
            "rl_saving":           rl_allocation["financial_saving"] if rl_allocation else None,
            "rl_roi":              rl_allocation["roi"] if rl_allocation else None,
            # LLM output
            "recommendation_text": rec_text,
        }

        logger.info(
            "  P%d  %-40s  saving=£%.0f  rl_funded=%s",
            int(row["priority"]), rc, potential_saving,
            str(rl_allocation["invested"]) if rl_allocation else "N/A",
        )

    return recommendations


# ---------------------------------------------------------------------------
# Generate executive summary
# ---------------------------------------------------------------------------

def generate_executive_summary(
    risk_df: pd.DataFrame,
    opt_results: dict | None,
    client: GeminiClient,
) -> str:
    """
    Generate a board-level executive summary.
    Now includes the full RL + LP optimisation output.
    """
    logger.info("Generating executive summary …")
    prompt = executive_summary_prompt(risk_df, opt_results)
    with timer("LLM call — executive summary"):
        summary = client.generate(prompt, label="executive_summary")
    return summary


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    recommendations: dict,
    executive_summary: str,
    opt_results: dict | None,
    client: GeminiClient,
) -> None:
    """Save all Phase 7 outputs to disk."""
    total_returns = sum(v["returns"] for v in recommendations.values())
    total_cost    = total_returns * COST_PER_RETURN
    total_saving  = sum(v["potential_saving"] for v in recommendations.values())

    # Include optimisation summary in metadata if available
    opt_summary = {}
    if opt_results:
        opt_summary = {
            "rl_total_saving":       opt_results.get("rl_policy", {}).get("total_saving", 0),
            "lp_total_saving":       opt_results.get("lp_result", {}).get("total_saving", 0),
            "convergence_gap_pct":   opt_results.get("comparison", {}).get("convergence_gap_pct", 0),
            "decision_agreement_pct":opt_results.get("comparison", {}).get("agreement_pct", 0),
            "budget":                opt_results.get("metadata", {}).get("budget", 150_000),
        }

    output = {
        "recommendations":   recommendations,
        "executive_summary": executive_summary,
        "metadata": {
            "model":            "gemini-2.0-flash" if client.available else "fallback",
            "generated_at":     datetime.now().isoformat(),
            "total_returns":    total_returns,
            "total_cost":       round(total_cost, 2),
            "total_saving":     round(total_saving, 2),
            "phase7a_included": opt_results is not None,
            "optimisation":     opt_summary,
        },
    }

    # Save joblib for dashboard
    save_artefact(output, RECOMMENDATIONS_PATH)

    # Save JSON for readability
    RECOMMENDATIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RECOMMENDATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Recommendations JSON saved → %s", RECOMMENDATIONS_JSON.name)

    # Save executive summary as plain text
    EXECUTIVE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXECUTIVE_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(executive_summary)
    logger.info("Executive summary saved → %s", EXECUTIVE_SUMMARY_PATH.name)

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 7 — RECOMMENDATION ENGINE SUMMARY")
    logger.info("=" * 60)
    logger.info("Total returns analysed   : %d", total_returns)
    logger.info("Total annual cost        : £%.0f", total_cost)
    logger.info("Total potential saving   : £%.0f", total_saving)
    if opt_summary:
        logger.info("RL optimisation saving   : £%.0f", opt_summary.get("rl_total_saving", 0))
        logger.info("LP baseline saving       : £%.0f", opt_summary.get("lp_total_saving", 0))
        logger.info("RL convergence gap       : %.1f%%", opt_summary.get("convergence_gap_pct", 0))
    logger.info("Phase 7a context included: %s", "YES" if opt_results else "NO")
    logger.info("=" * 60)
    for rc, rec in sorted(recommendations.items(), key=lambda x: x[1]["priority"]):
        rl_status = ""
        if rec.get("rl_funded") is not None:
            rl_status = "  [RL: FUNDED]" if rec["rl_funded"] else "  [RL: SKIPPED]"
        logger.info("  P%d  %-40s  £%.0f saving%s",
                    rec["priority"], rc, rec["potential_saving"], rl_status)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def run_recommendation_engine(api_key: str | None = None) -> dict:
    """
    Run the complete Phase 7 recommendation engine.

    Automatically loads Phase 7a optimisation results if available.
    If Phase 7a has not been run, the engine still works but recommendations
    will not include the budget allocation context.

    Parameters
    ----------
    api_key : str | None
        Gemini API key. If None reads from GEMINI_API_KEY env variable.

    Returns
    -------
    dict with keys: recommendations, executive_summary
    """
    client = GeminiClient(api_key=api_key)
    inputs = load_inputs()

    with timer("Generating all recommendations"):
        recommendations = generate_recommendations(inputs, client)

    with timer("Generating executive summary"):
        executive_summary = generate_executive_summary(
            inputs["risk_df"], inputs["opt_results"], client
        )

    save_outputs(recommendations, executive_summary, inputs["opt_results"], client)

    return {
        "recommendations":   recommendations,
        "executive_summary": executive_summary,
    }
