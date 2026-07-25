"""
prompt_builder.py  –  Constructs data-rich prompts for the Gemini LLM.

Two prompt types:
  1. per_root_cause_prompt()    — detailed stakeholder recommendation
  2. executive_summary_prompt() — board-level executive summary
"""

import pandas as pd
from .utils import STAKEHOLDER_MAP, COST_PER_RETURN, REDUCTION_POTENTIAL, get_logger

logger = get_logger("phase7.prompt_builder")


def _format_risk_row(row: pd.Series) -> str:
    rc        = row["root_cause_category"]
    cost      = row["count"] * COST_PER_RETURN
    reducible = int(row["count"] * REDUCTION_POTENTIAL.get(rc, 0.3))
    saving    = reducible * COST_PER_RETURN
    return f"""Root cause category    : {rc}
Priority ranking       : P{int(row['priority'])} of 6
Risk score             : {row['risk_score']:.1f} / 100
Risk level             : {row['risk_level']}
Total confirmed returns: {row['count']:,}
Share of all returns   : {row['pct_of_returns']:.1f}%
Frequency score        : {row['frequency_score']:.1f} / 100
Operational impact     : {row['impact_score']:.1f} / 100
Trend direction score  : {row['trend_score']:.1f} / 100
Average sentiment      : {row['avg_sentiment']:.3f}
Average return lag     : {row['avg_return_lag']:.1f} days after delivery
Responsible stakeholder: {STAKEHOLDER_MAP.get(rc, 'Operations Team')}
Estimated annual cost  : £{cost:,.0f}
Reducible returns      : {reducible:,} ({REDUCTION_POTENTIAL.get(rc, 0.3)*100:.0f}% with intervention)
Potential annual saving: £{saving:,.0f}"""


def _format_brand_context(brand_detail: pd.DataFrame, root_cause: str) -> str:
    try:
        filtered = brand_detail[brand_detail["root_cause_category"] == root_cause]
        if filtered.empty:
            return "  Brand data not available."
        top2  = filtered.nlargest(2, "count")[["brand", "count", "pct_within_brand"]]
        lines = []
        for _, r in top2.iterrows():
            lines.append(
                f"  - {r['brand']}: {int(r['count']):,} returns "
                f"({r['pct_within_brand']:.1f}% of that brand's returns)"
            )
        return "\n".join(lines)
    except Exception:
        return "  Brand data not available."


def _format_region_context(region: pd.DataFrame, root_cause: str) -> str:
    try:
        filtered = region[region["root_cause_category"] == root_cause]
        if filtered.empty:
            return "  Region data not available."
        top = filtered.nlargest(1, "pct_within_region").iloc[0]
        return f"  - {top['region']} region: {top['pct_within_region']:.1f}% of regional returns"
    except Exception:
        return "  Region data not available."


def per_root_cause_prompt(
    row: pd.Series,
    brand_detail: pd.DataFrame,
    region: pd.DataFrame,
) -> str:
    rc          = row["root_cause_category"]
    stakeholder = STAKEHOLDER_MAP.get(rc, "Operations Team")
    risk_data   = _format_risk_row(row)
    brand_ctx   = _format_brand_context(brand_detail, rc)
    region_ctx  = _format_region_context(region, rc)
    reducible   = int(row["count"] * REDUCTION_POTENTIAL.get(rc, 0.3))
    saving      = reducible * COST_PER_RETURN

    return f"""You are an expert supply chain analyst for a UK fashion e-commerce retailer.
You have been given results from a multi-phase AI framework:
  Phase 2: NLP analysis of 50,000 customer reviews
  Phase 3: XGBoost root cause classification (97.1% accuracy)
  Phase 4: SHAP explainability
  Phase 5: Root cause analysis layer
  Phase 6: Quantified risk assessment

Use ONLY the data provided below. Do not invent figures.

RISK ASSESSMENT DATA:
{risk_data}

MOST AFFECTED BRANDS:
{brand_ctx}

MOST AFFECTED REGION:
{region_ctx}

Generate a structured supply chain recommendation addressed to the {stakeholder}.

RECOMMENDATION FOR: {rc.upper()}
ADDRESSED TO: {stakeholder}
PRIORITY: P{int(row['priority'])} | RISK LEVEL: {row['risk_level']} | RISK SCORE: {row['risk_score']:.1f}/100

SITUATION SUMMARY:
[2-3 sentences on the scale and nature of this return problem using specific numbers]

ROOT CAUSE ANALYSIS:
[2-3 sentences explaining operational cause based on risk scores and sentiment data]

IMMEDIATE ACTIONS (next 30 days):
1. [Specific action with measurable outcome]
2. [Specific action with measurable outcome]
3. [Specific action with measurable outcome]

MEDIUM-TERM ACTIONS (30-90 days):
1. [Specific action]
2. [Specific action]

EXPECTED IMPACT:
- Returns prevented: approximately {reducible:,} per year
- Cost saving: approximately £{saving:,.0f} per year
- Timeline to see results: [realistic timeline]

KEY PERFORMANCE INDICATORS TO TRACK:
1. [Specific measurable KPI]
2. [Specific measurable KPI]

STAKEHOLDER MESSAGE:
[One paragraph a manager can read in 30 seconds capturing urgency and action]"""


def executive_summary_prompt(
    risk_df: pd.DataFrame,
    total_returns: int = 14000,
) -> str:
    total_cost   = total_returns * COST_PER_RETURN
    total_saving = sum(
        int(r["count"] * REDUCTION_POTENTIAL.get(r["root_cause_category"], 0.3)) * COST_PER_RETURN
        for _, r in risk_df.iterrows()
    )

    table_lines = []
    for _, r in risk_df.iterrows():
        table_lines.append(
            f"  P{int(r['priority'])}  {r['root_cause_category']:<40} "
            f"Score:{r['risk_score']:.1f}  {r['risk_level']:<8}  "
            f"{r['count']:,} returns ({r['pct_of_returns']:.1f}%)"
        )

    return f"""You are a senior AI consultant presenting to the board of directors of a
UK fashion e-commerce retailer. You have results from an AI framework covering
NLP, XGBoost classification, SHAP explainability, and risk assessment.

RISK ASSESSMENT — ALL ROOT CAUSES:
Total confirmed returns : {total_returns:,}
Annual return cost      : £{total_cost:,.0f}
Max achievable saving   : £{total_saving:,.0f}

PRIORITY RANKING:
{chr(10).join(table_lines)}

Write a professional executive summary for the board of directors.

EXECUTIVE SUMMARY — AI-POWERED SUPPLY CHAIN RETURN ANALYSIS

OVERVIEW:
[2-3 sentences on the scale of the problem and what the AI framework discovered]

KEY FINDINGS:
1. [Most important finding with specific data]
2. [Second most important finding]
3. [Third most important finding]

FINANCIAL IMPACT:
[2 sentences on current cost and achievable saving]

TOP 3 PRIORITY ACTIONS FOR THE BOARD:
1. [Board-level action — highest priority]
2. [Board-level action — second priority]
3. [Board-level action — third priority]

CONCLUSION:
[2-3 sentences on next steps]"""