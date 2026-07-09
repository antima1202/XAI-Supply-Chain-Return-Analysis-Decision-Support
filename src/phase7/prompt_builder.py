"""
prompt_builder.py  –  Constructs data-rich prompts for the Gemini LLM.

Updated in Phase 7 (post Phase 7a) to include:
  - RL optimal budget allocation per root cause
  - LP baseline comparison
  - Convergence gap between RL and LP
  - Whether this root cause was funded by the optimisation

Two prompt types:
  1. per_root_cause_prompt()   — detailed stakeholder recommendation
  2. executive_summary_prompt() — board-level executive summary
"""

import pandas as pd
from .utils import STAKEHOLDER_MAP, COST_PER_RETURN, REDUCTION_POTENTIAL, get_logger

logger = get_logger("phase7.prompt_builder")


# ---------------------------------------------------------------------------
# Helper — format risk row
# ---------------------------------------------------------------------------

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
Trend direction score  : {row['trend_score']:.1f} / 100  (50=stable, >50=worsening, <50=improving)
Average customer sentiment: {row['avg_sentiment']:.3f}  (-1=very negative, +1=very positive)
Average return lag     : {row['avg_return_lag']:.1f} days after delivery
Responsible stakeholder: {STAKEHOLDER_MAP.get(rc, 'Operations Team')}
Estimated annual cost  : £{cost:,.0f}
Estimated reducible    : {reducible:,} returns ({REDUCTION_POTENTIAL.get(rc, 0.3)*100:.0f}% with intervention)
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


def _format_optimisation_context(root_cause: str, opt_results: dict | None) -> str:
    """
    Format the Phase 7a RL + LP optimisation decision for this root cause.
    Returns a context block showing whether the optimisation funded this
    intervention and at what budget level.
    """
    if not opt_results:
        return "  Optimisation data not available (run Phase 7a first)."

    rl_policy  = opt_results.get("rl_policy",  {})
    lp_result  = opt_results.get("lp_result",  {})
    comparison = opt_results.get("comparison", {})
    metadata   = opt_results.get("metadata",   {})

    total_budget = metadata.get("budget", 150_000)

    # Find this root cause in RL allocation
    rl_alloc = next(
        (a for a in rl_policy.get("allocation", []) if a["root_cause"] == root_cause),
        None
    )
    lp_alloc = next(
        (a for a in lp_result.get("allocation", []) if a["root_cause"] == root_cause),
        None
    )

    if not rl_alloc:
        return "  Optimisation allocation data not available for this root cause."

    rl_decision = "FUND" if rl_alloc["invested"] else "SKIP"
    lp_decision = "FUND" if (lp_alloc and lp_alloc["invested"]) else "SKIP"
    agreement   = "✓ AGREE" if rl_decision == lp_decision else "✗ DISAGREE"

    lines = [
        f"  Total optimisation budget  : £{total_budget:,.0f}",
        f"  RL agent decision          : {rl_decision}",
        f"  LP baseline decision       : {lp_decision}",
        f"  RL vs LP agreement         : {agreement}",
        f"  RL convergence gap vs LP   : {comparison.get('convergence_gap_pct', 0):.1f}%",
    ]

    if rl_alloc["invested"]:
        lines += [
            f"  Budget allocated (RL)      : £{rl_alloc['cost']:,.0f}",
            f"  Returns to prevent (RL)    : {rl_alloc['returns_prevented']:,}",
            f"  Financial saving (RL)      : £{rl_alloc['financial_saving']:,.0f}",
            f"  ROI                        : {rl_alloc['roi']:.2f}x (£ saving per £ invested)",
        ]
    else:
        lines.append(
            "  Reason skipped             : Insufficient budget after higher-ROI interventions funded"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt 1: Per root cause recommendation (UPDATED with RL/LP context)
# ---------------------------------------------------------------------------

def per_root_cause_prompt(
    row: pd.Series,
    brand_detail: pd.DataFrame,
    region: pd.DataFrame,
    opt_results: dict | None = None,
) -> str:
    """
    Build a detailed prompt for one root cause recommendation.
    Now includes Phase 7a RL + LP optimisation context so Gemini
    generates recommendations grounded in the budget allocation decision.
    """
    rc          = row["root_cause_category"]
    stakeholder = STAKEHOLDER_MAP.get(rc, "Operations Team")
    risk_data   = _format_risk_row(row)
    brand_ctx   = _format_brand_context(brand_detail, rc)
    region_ctx  = _format_region_context(region, rc)
    opt_ctx     = _format_optimisation_context(rc, opt_results)
    reducible   = int(row["count"] * REDUCTION_POTENTIAL.get(rc, 0.3))
    saving      = reducible * COST_PER_RETURN

    # Check if this root cause was funded by the optimisation
    rl_alloc    = None
    if opt_results:
        rl_alloc = next(
            (a for a in opt_results.get("rl_policy", {}).get("allocation", [])
             if a["root_cause"] == rc), None
        )
    funded      = rl_alloc["invested"] if rl_alloc else True
    budget_note = (
        f"The RL optimisation has ALLOCATED £{rl_alloc['cost']:,.0f} to this intervention."
        if funded and rl_alloc
        else "Note: The RL optimisation did NOT fund this intervention in the current budget scenario."
    )

        # Pre-compute values that cannot be inline in f-string conditionals
    _budget_req = rl_alloc["cost"] if (rl_alloc and funded) else 0
    _roi_str    = f"{rl_alloc['roi']:.2f}x" if (rl_alloc and funded) else "N/A — not funded in current budget"

    return f"""You are an expert supply chain analyst for a UK fashion e-commerce retailer.
You have been given results from a multi-phase AI framework:
  Phase 3: XGBoost root cause classification (97.3% accuracy)
  Phase 4: SHAP explainability (identifies key return drivers)
  Phase 6: Risk assessment (composite risk scoring)
  Phase 7a: Operational optimisation (Q-learning RL agent + LP baseline)

Use ONLY the data provided below. Do not invent figures.

═══════════════════════════════════════════════════════
RISK ASSESSMENT DATA (Phase 6)
═══════════════════════════════════════════════════════
{risk_data}

MOST AFFECTED BRANDS:
{brand_ctx}

MOST AFFECTED REGION:
{region_ctx}

═══════════════════════════════════════════════════════
OPTIMISATION DECISION (Phase 7a — RL + LP)
═══════════════════════════════════════════════════════
{opt_ctx}

{budget_note}
═══════════════════════════════════════════════════════

Generate a structured supply chain recommendation addressed to the {stakeholder}.
The recommendation MUST reference the optimisation budget allocation and explain
whether this intervention should proceed given the current budget scenario.

RECOMMENDATION FOR: {rc.upper()}
ADDRESSED TO: {stakeholder}
PRIORITY: P{int(row['priority'])} | RISK LEVEL: {row['risk_level']} | RISK SCORE: {row['risk_score']:.1f}/100
OPTIMISATION STATUS: {"✓ FUNDED" if funded else "✗ NOT FUNDED IN CURRENT BUDGET"}

SITUATION SUMMARY:
[2-3 sentences on the scale and nature of this return problem using specific numbers]

ROOT CAUSE ANALYSIS:
[2-3 sentences explaining operational cause based on risk scores and sentiment data]

OPTIMISATION RATIONALE:
[1-2 sentences explaining why the RL agent {"funded" if funded else "did not fund"} this intervention
and what the ROI justification is]

IMMEDIATE ACTIONS (next 30 days):
1. [Specific action with measurable outcome]
2. [Specific action with measurable outcome]
3. [Specific action with measurable outcome]

MEDIUM-TERM ACTIONS (30-90 days):
1. [Specific action]
2. [Specific action]

EXPECTED IMPACT:
- Budget required: £{_budget_req:,.0f}
- Returns prevented: approximately {reducible:,} per year
- Cost saving: approximately £{saving:,.0f} per year
- ROI: {_roi_str}
- Timeline to see results: [realistic timeline based on complexity]

KEY PERFORMANCE INDICATORS TO TRACK:
1. [Specific measurable KPI]
2. [Specific measurable KPI]

STAKEHOLDER MESSAGE:
[One paragraph a manager can read in 30 seconds capturing urgency, budget allocation, and action]"""


# ---------------------------------------------------------------------------
# Prompt 2: Executive summary (UPDATED with RL optimisation results)
# ---------------------------------------------------------------------------

def executive_summary_prompt(
    risk_df: pd.DataFrame,
    opt_results: dict | None = None,
    total_returns: int = 14000,
) -> str:
    """Build a board-level executive summary including RL + LP optimisation output."""
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

    # Format RL optimisation section
    if opt_results:
        rl_policy  = opt_results.get("rl_policy",  {})
        lp_result  = opt_results.get("lp_result",  {})
        comparison = opt_results.get("comparison", {})
        metadata   = opt_results.get("metadata",   {})
        budget     = metadata.get("budget", 150_000)

        rl_lines = []
        for a in sorted(rl_policy.get("allocation", []),
                        key=lambda x: x["roi"], reverse=True):
            status = "✓ FUND" if a["invested"] else "✗ SKIP"
            if a["invested"]:
                rl_lines.append(
                    f"  {status}  {a['root_cause']:<40} "
                    f"£{a['cost']:,.0f} → saves £{a['financial_saving']:,.0f} "
                    f"(ROI {a['roi']:.2f}x)"
                )
            else:
                rl_lines.append(f"  {status}  {a['root_cause']:<40} (budget exhausted)")

        opt_section = f"""RL OPTIMISATION RESULTS (Budget: £{budget:,.0f}):
{chr(10).join(rl_lines)}

  RL total saving      : £{rl_policy.get('total_saving', 0):,.0f}
  LP baseline saving   : £{lp_result.get('total_saving', 0):,.0f}
  Convergence gap      : {comparison.get('convergence_gap_pct', 0):.1f}%
  Decision agreement   : {comparison.get('agreement_pct', 0):.1f}%
  Returns prevented    : {rl_policy.get('total_returns_prevented', 0):,}"""
    else:
        opt_section = "Optimisation data not available — run Phase 7a first."

    return f"""You are a senior AI consultant presenting to the board of directors of a
UK fashion e-commerce retailer. You have results from a 10-phase AI framework:
  NLP (Phase 2) → ML Classification (Phase 3) → SHAP XAI (Phase 4) →
  Risk Assessment (Phase 6) → RL + LP Optimisation (Phase 7a) → LLM (Phase 7)

═══════════════════════════════════════════════════════
RISK ASSESSMENT — ALL ROOT CAUSES
═══════════════════════════════════════════════════════
Total confirmed returns : {total_returns:,}
Annual return cost      : £{total_cost:,.0f}
Max achievable saving   : £{total_saving:,.0f}

PRIORITY RANKING:
{chr(10).join(table_lines)}

═══════════════════════════════════════════════════════
{opt_section}
═══════════════════════════════════════════════════════

Write a professional executive summary for the board of directors.

EXECUTIVE SUMMARY — AI-POWERED SUPPLY CHAIN RETURN ANALYSIS

OVERVIEW:
[2-3 sentences on the scale of the problem and what the AI framework discovered]

KEY FINDINGS:
1. [Most important finding with specific data]
2. [Second most important finding]
3. [Third most important finding — mention RL convergence to LP optimum]

FINANCIAL IMPACT:
[2 sentences on current cost and achievable saving]

TOP 3 PRIORITY ACTIONS FOR THE BOARD:
1. [Board-level action — highest priority root cause]
2. [Board-level action — second priority]
3. [Board-level action — third priority]

RECOMMENDED BUDGET ALLOCATION:
[2-3 sentences summarising the RL optimisation output — mention that the AI agent
converged to the mathematical optimum and what the ROI is]

AI FRAMEWORK VALUE:
[2 sentences on why this AI approach — NLP + ML + XAI + RL + LLM — delivers
more value than traditional analytics alone]

CONCLUSION:
[2-3 sentences on next steps and continuous improvement]"""
