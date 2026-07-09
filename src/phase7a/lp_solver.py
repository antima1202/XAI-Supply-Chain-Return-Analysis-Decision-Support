"""
lp_solver.py  –  Linear Programming baseline for Phase 7a.

Purpose
-------
LP provides the mathematically optimal solution under perfect information.
It serves as a benchmark to evaluate whether the RL agent's learned
policy converges toward the theoretical optimum.

Formulation
-----------
Decision variables:
  x_i ∈ {0, 1}  — binary: fund intervention i or not

Objective (maximise):
  Σ x_i · returns_prevented_i · cost_per_return_i

Subject to:
  Σ x_i · intervention_cost_i ≤ budget          (budget constraint)
  x_i ∈ {0, 1}  for all i                       (binary)

This is a 0-1 Knapsack problem — NP-hard in general but trivially
solvable for 6 items using PuLP's CBC solver.

Academic role
-------------
The LP solution represents the best possible outcome with:
  - Perfect knowledge of reduction percentages
  - Exact intervention costs
  - No uncertainty

The RL agent operates under simulated uncertainty and learns through
trial and error. Comparing RL vs LP convergence is the key academic
contribution of Phase 7a.
"""

import pandas as pd

from .utils import (
    ROOT_CAUSE_CLASSES,
    DEFAULT_BUDGET,
    DEFAULT_COST_PER_RETURN,
    DEFAULT_INTERVENTION_COSTS,
    DEFAULT_REDUCTION_POTENTIAL,
    get_logger,
)

logger = get_logger("phase7a.lp_solver")


def solve_lp(
    risk_df: pd.DataFrame,
    budget: float = DEFAULT_BUDGET,
    cost_per_return: float = DEFAULT_COST_PER_RETURN,
    intervention_costs: dict = None,
    reduction_potential: dict = None,
) -> dict:
    """
    Solve the budget allocation problem using Linear Programming (0-1 Knapsack).

    Parameters
    ----------
    risk_df             : Phase 6 risk scores DataFrame
    budget              : total budget (£)
    cost_per_return     : reverse logistics cost per return (£)
    intervention_costs  : dict {root_cause: min_cost}
    reduction_potential : dict {root_cause: reduction_fraction}

    Returns
    -------
    dict with LP solution including allocation, returns prevented, saving
    """
    try:
        import pulp
    except ImportError:
        logger.warning("PuLP not installed. Run: pip install pulp")
        return _fallback_greedy(risk_df, budget, cost_per_return,
                                intervention_costs, reduction_potential)

    costs      = intervention_costs  or DEFAULT_INTERVENTION_COSTS
    reductions = reduction_potential or DEFAULT_REDUCTION_POTENTIAL

    rc_data = {}
    for rc in ROOT_CAUSE_CLASSES:
        if rc in risk_df.set_index("root_cause_category").index:
            count    = int(risk_df.set_index("root_cause_category").loc[rc, "count"])
            prevented = count * reductions.get(rc, 0.3)
            saving    = prevented * cost_per_return
            rc_data[rc] = {
                "count":             count,
                "cost":              costs.get(rc, 30_000),
                "returns_prevented": prevented,
                "saving":            saving,
            }

    # Define LP problem
    prob = pulp.LpProblem("SupplyChainBudgetAllocation", pulp.LpMaximize)

    # Binary decision variables
    x = {rc: pulp.LpVariable(f"x_{i}", cat="Binary")
         for i, rc in enumerate(ROOT_CAUSE_CLASSES) if rc in rc_data}

    # Objective: maximise total financial saving
    prob += pulp.lpSum(x[rc] * rc_data[rc]["saving"] for rc in x)

    # Budget constraint
    prob += pulp.lpSum(x[rc] * rc_data[rc]["cost"] for rc in x) <= budget

    # Solve (suppress solver output)
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        logger.warning("LP did not find optimal solution — status: %s", pulp.LpStatus[status])
        return _fallback_greedy(risk_df, budget, cost_per_return,
                                intervention_costs, reduction_potential)

    # Extract solution
    allocation   = []
    total_cost   = 0.0
    total_saving = 0.0
    total_prevented = 0.0

    for rc in ROOT_CAUSE_CLASSES:
        if rc not in rc_data:
            continue
        funded = pulp.value(x[rc]) > 0.5 if rc in x else False
        d      = rc_data[rc]

        allocation.append({
            "root_cause":        rc,
            "invested":          funded,
            "cost":              d["cost"] if funded else 0,
            "returns_prevented": round(d["returns_prevented"]) if funded else 0,
            "financial_saving":  round(d["saving"], 2) if funded else 0.0,
            "roi":               round(d["saving"] / d["cost"], 3) if funded and d["cost"] > 0 else 0.0,
        })

        if funded:
            total_cost      += d["cost"]
            total_saving    += d["saving"]
            total_prevented += d["returns_prevented"]

    allocation.sort(key=lambda x: x["roi"], reverse=True)

    result = {
        "allocation":              allocation,
        "budget_total":            budget,
        "budget_spent":            round(total_cost, 2),
        "budget_remaining":        round(budget - total_cost, 2),
        "total_returns_prevented": round(total_prevented),
        "total_saving":            round(total_saving, 2),
        "funded_count":            sum(1 for a in allocation if a["invested"]),
        "solver":                  "LP (PuLP CBC)",
    }

    logger.info("LP solution found:")
    logger.info("  Budget spent    : £%.0f / £%.0f", total_cost, budget)
    logger.info("  Returns prevented: %d", round(total_prevented))
    logger.info("  Total saving    : £%.0f", total_saving)
    for a in allocation:
        status_str = "✓ FUND" if a["invested"] else "✗ SKIP"
        logger.info("  %s  %-40s  £%.0f → saves £%.0f",
                    status_str, a["root_cause"], a["cost"], a["financial_saving"])

    return result


def _fallback_greedy(
    risk_df, budget, cost_per_return, intervention_costs, reduction_potential
) -> dict:
    """
    Greedy fallback when PuLP is unavailable.
    Funds interventions in descending ROI order until budget is exhausted.
    """
    logger.info("Using greedy fallback (PuLP unavailable)")
    costs      = intervention_costs  or DEFAULT_INTERVENTION_COSTS
    reductions = reduction_potential or DEFAULT_REDUCTION_POTENTIAL

    items = []
    for rc in ROOT_CAUSE_CLASSES:
        idx = risk_df.set_index("root_cause_category")
        if rc not in idx.index:
            continue
        count   = int(idx.loc[rc, "count"])
        cost    = costs.get(rc, 30_000)
        saving  = count * reductions.get(rc, 0.3) * cost_per_return
        roi     = saving / cost if cost > 0 else 0
        items.append((rc, cost, saving, roi, count))

    items.sort(key=lambda x: x[3], reverse=True)

    allocation     = []
    remaining      = budget
    total_saving   = 0.0
    total_prevented = 0.0
    total_cost     = 0.0

    funded_set = set()
    for rc, cost, saving, roi, count in items:
        if cost <= remaining:
            funded_set.add(rc)
            remaining      -= cost
            total_saving   += saving
            total_prevented += count * reductions.get(rc, 0.3)
            total_cost     += cost

    for rc, cost, saving, roi, count in items:
        funded = rc in funded_set
        allocation.append({
            "root_cause":        rc,
            "invested":          funded,
            "cost":              cost if funded else 0,
            "returns_prevented": round(count * reductions.get(rc, 0.3)) if funded else 0,
            "financial_saving":  round(saving, 2) if funded else 0.0,
            "roi":               round(roi, 3) if funded else 0.0,
        })

    return {
        "allocation":              allocation,
        "budget_total":            budget,
        "budget_spent":            round(total_cost, 2),
        "budget_remaining":        round(budget - total_cost, 2),
        "total_returns_prevented": round(total_prevented),
        "total_saving":            round(total_saving, 2),
        "funded_count":            len(funded_set),
        "solver":                  "Greedy (PuLP fallback)",
    }
