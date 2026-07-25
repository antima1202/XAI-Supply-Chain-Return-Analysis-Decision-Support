"""
recommendation_engine.py  –  Core recommendation generation for Phase 7.
Loads Phase 5 and Phase 6 artefacts, calls Ollama LLM, saves outputs.
"""

import json
from datetime import datetime

import pandas as pd

from .llm_client     import GeminiClient
from .prompt_builder import per_root_cause_prompt, executive_summary_prompt
from .utils import (
    RISK_SCORES_PATH,
    ROOT_CAUSE_SUMMARY_PATH,
    BRAND_BREAKDOWN_PATH,
    REGION_BREAKDOWN_PATH,
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


def load_inputs() -> dict:
    """Load all required artefacts from Phases 5 and 6."""
    logger.info("Loading Phase 5 and 6 artefacts …")
    risk_df = load_artefact(RISK_SCORES_PATH)
    summary = load_artefact(ROOT_CAUSE_SUMMARY_PATH)
    brand   = load_artefact(BRAND_BREAKDOWN_PATH)
    region  = load_artefact(REGION_BREAKDOWN_PATH)
    logger.info("Loaded risk scores (%d rows)", len(risk_df))
    logger.info("Loaded brand breakdown (%d rows)", len(brand))
    logger.info("Loaded region breakdown (%d rows)", len(region))
    return {
        "risk_df": risk_df,
        "summary": summary,
        "brand":   brand,
        "region":  region,
    }


def generate_recommendations(inputs: dict, client: GeminiClient) -> dict:
    """Generate one LLM recommendation per root cause."""
    risk_df = inputs["risk_df"]
    brand   = inputs["brand"]
    region  = inputs["region"]
    recommendations = {}

    for _, row in risk_df.iterrows():
        rc = row["root_cause_category"]
        logger.info("Generating recommendation for: %s (P%d)", rc, int(row["priority"]))
        prompt   = per_root_cause_prompt(row, brand, region)
        with timer(f"LLM call — {rc}"):
            rec_text = client.generate(prompt, label=rc)

        cost_impact       = row["count"] * COST_PER_RETURN
        reduction_pct     = REDUCTION_POTENTIAL.get(rc, 0.3)
        returns_prevented = int(row["count"] * reduction_pct)
        potential_saving  = returns_prevented * COST_PER_RETURN

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
            "recommendation_text": rec_text,
        }
        logger.info(
            "  P%d  %-40s  saving=£%.0f",
            int(row["priority"]), rc, potential_saving,
        )
    return recommendations


def generate_executive_summary(
    risk_df: pd.DataFrame,
    client: GeminiClient,
) -> str:
    """Generate a board-level executive summary using the LLM."""
    logger.info("Generating executive summary …")
    prompt = executive_summary_prompt(risk_df)
    with timer("LLM call — executive summary"):
        summary = client.generate(prompt, label="executive_summary")
    return summary


def save_outputs(
    recommendations: dict,
    executive_summary: str,
    client: GeminiClient,
) -> None:
    """Save all Phase 7 outputs to disk."""
    total_returns = sum(v["returns"] for v in recommendations.values())
    total_cost    = total_returns * COST_PER_RETURN
    total_saving  = sum(v["potential_saving"] for v in recommendations.values())

    output = {
        "recommendations":   recommendations,
        "executive_summary": executive_summary,
        "metadata": {
            "model":         "Ollama — llama3.2" if client.available else "fallback",
            "generated_at":  datetime.now().isoformat(),
            "total_returns": total_returns,
            "total_cost":    round(total_cost, 2),
            "total_saving":  round(total_saving, 2),
        },
    }

    save_artefact(output, RECOMMENDATIONS_PATH)

    RECOMMENDATIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RECOMMENDATIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Recommendations JSON saved → %s", RECOMMENDATIONS_JSON.name)

    EXECUTIVE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXECUTIVE_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(executive_summary)
    logger.info("Executive summary saved → %s", EXECUTIVE_SUMMARY_PATH.name)

    logger.info("=" * 60)
    logger.info("PHASE 7 SUMMARY")
    logger.info("Total returns : %d", total_returns)
    logger.info("Total cost    : £%.0f", total_cost)
    logger.info("Total saving  : £%.0f", total_saving)
    logger.info("=" * 60)
    for rc, rec in sorted(recommendations.items(), key=lambda x: x[1]["priority"]):
        logger.info("  P%d  %-40s  £%.0f saving",
                    rec["priority"], rc, rec["potential_saving"])


def run_recommendation_engine(api_key: str | None = None) -> dict:
    """Run the complete Phase 7 recommendation engine."""
    client = GeminiClient(api_key=api_key)
    inputs = load_inputs()

    with timer("Generating all recommendations"):
        recommendations = generate_recommendations(inputs, client)

    with timer("Generating executive summary"):
        executive_summary = generate_executive_summary(inputs["risk_df"], client)

    save_outputs(recommendations, executive_summary, client)

    return {
        "recommendations":   recommendations,
        "executive_summary": executive_summary,
    }