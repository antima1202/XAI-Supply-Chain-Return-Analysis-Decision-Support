"""
pipeline.py  –  Phase 7a Operational Optimisation pipeline orchestrator.

Execution order
---------------
  1.  Load Phase 6 risk scores
  2.  Initialise supply chain simulation environment
  3.  Train Q-learning RL agent (2000 episodes)
  4.  Extract optimal RL policy (greedy episode)
  5.  Solve LP baseline (mathematical optimum)
  6.  Compare RL vs LP — compute convergence gap
  7.  Save all artefacts
  8.  Generate 7 dissertation plots

All business parameters (budget, costs, reductions) are passed as
arguments so the dashboard can override them with user inputs.
"""

import json
import pandas as pd
from datetime import datetime

from .rl_environment  import SupplyChainEnv
from .rl_agent        import QLearningAgent
from .lp_solver       import solve_lp
from .visualisations  import generate_all_plots
from .utils import (
    RISK_SCORES_PATH,
    RL_RESULTS_PATH,
    LP_RESULTS_PATH,
    OPT_RESULTS_PATH,
    OPT_RESULTS_CSV,
    MODELS_DIR,
    DEFAULT_BUDGET,
    DEFAULT_COST_PER_RETURN,
    DEFAULT_INTERVENTION_COSTS,
    DEFAULT_REDUCTION_POTENTIAL,
    DEFAULT_N_EPISODES,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DISCOUNT,
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_EPSILON_MIN,
    get_logger,
    load_artefact,
    save_artefact,
    timer,
)

logger = get_logger("phase7a.pipeline")


def run_phase7a_pipeline(
    budget:               float = DEFAULT_BUDGET,
    cost_per_return:      float = DEFAULT_COST_PER_RETURN,
    intervention_costs:   dict  = None,
    reduction_potential:  dict  = None,
    n_episodes:           int   = DEFAULT_N_EPISODES,
    learning_rate:        float = DEFAULT_LEARNING_RATE,
    discount:             float = DEFAULT_DISCOUNT,
    random_seed:          int   = 42,
) -> dict:
    """
    Execute the complete Phase 7a Operational Optimisation pipeline.

    Parameters (all overridable from dashboard)
    -------------------------------------------
    budget              : total intervention budget (£)
    cost_per_return     : reverse logistics cost per return (£)
    intervention_costs  : minimum cost per root cause (£)
    reduction_potential : expected reduction % per root cause
    n_episodes          : RL training episodes
    learning_rate       : Q-learning alpha
    discount            : Q-learning gamma (future reward discount)
    random_seed         : reproducibility

    Returns
    -------
    dict with keys: rl_policy, lp_result, comparison, history
    """
    import numpy as np
    np.random.seed(random_seed)

    costs      = intervention_costs  or DEFAULT_INTERVENTION_COSTS
    reductions = reduction_potential or DEFAULT_REDUCTION_POTENTIAL

    logger.info("=" * 60)
    logger.info("Phase 7a Operational Optimisation — START")
    logger.info("=" * 60)
    logger.info("Budget: £%.0f | Cost/return: £%.0f | Episodes: %d",
                budget, cost_per_return, n_episodes)

    # 1. Load Phase 6 risk scores
    with timer("Loading Phase 6 risk scores"):
        risk_df = load_artefact(RISK_SCORES_PATH)
    logger.info("Loaded risk scores (%d root causes)", len(risk_df))

    # 2. Initialise environment
    env = SupplyChainEnv(
        risk_df=risk_df,
        budget=budget,
        cost_per_return=cost_per_return,
        intervention_costs=costs,
        reduction_potential=reductions,
    )

    # 3. Train RL agent
    agent = QLearningAgent(
        env=env,
        learning_rate=learning_rate,
        discount=discount,
        epsilon=DEFAULT_EPSILON,
        epsilon_decay=DEFAULT_EPSILON_DECAY,
        epsilon_min=DEFAULT_EPSILON_MIN,
    )
    with timer(f"RL training ({n_episodes} episodes)"):
        history = agent.train(n_episodes=n_episodes)

    # 4. Extract optimal RL policy
    with timer("Extracting optimal RL policy"):
        rl_policy = agent.get_optimal_policy()

    # 5. Solve LP baseline
    with timer("Solving LP baseline"):
        lp_result = solve_lp(
            risk_df=risk_df,
            budget=budget,
            cost_per_return=cost_per_return,
            intervention_costs=costs,
            reduction_potential=reductions,
        )

    # 6. Compare RL vs LP
    convergence_gap_pct = abs(
        lp_result["total_saving"] - rl_policy["total_saving"]
    ) / max(lp_result["total_saving"], 1) * 100

    # Check if RL and LP made the same allocation decisions
    lp_funded = {a["root_cause"] for a in lp_result["allocation"] if a["invested"]}
    rl_funded = {a["root_cause"] for a in rl_policy["allocation"] if a["invested"]}
    agreement_pct = len(lp_funded & rl_funded) / max(len(lp_funded | rl_funded), 1) * 100

    comparison = {
        "rl_saving":              rl_policy["total_saving"],
        "lp_saving":              lp_result["total_saving"],
        "convergence_gap_pct":    round(convergence_gap_pct, 2),
        "agreement_pct":          round(agreement_pct, 1),
        "rl_returns_prevented":   rl_policy["total_returns_prevented"],
        "lp_returns_prevented":   lp_result["total_returns_prevented"],
        "rl_budget_spent":        rl_policy["budget_spent"],
        "lp_budget_spent":        lp_result["budget_spent"],
        "rl_funded":              list(rl_funded),
        "lp_funded":              list(lp_funded),
    }

    logger.info("\n--- RL vs LP Comparison ---")
    logger.info("RL saving  : £%.0f  |  LP saving  : £%.0f", rl_policy["total_saving"], lp_result["total_saving"])
    logger.info("RL prevented: %d   |  LP prevented: %d", rl_policy["total_returns_prevented"], lp_result["total_returns_prevented"])
    logger.info("Convergence gap : %.1f%%", convergence_gap_pct)
    logger.info("Decision agreement: %.1f%%", agreement_pct)

    # 7. Save artefacts
    full_results = {
        "rl_policy":   rl_policy,
        "lp_result":   lp_result,
        "comparison":  comparison,
        "metadata": {
            "budget":             budget,
            "cost_per_return":    cost_per_return,
            "n_episodes":         n_episodes,
            "learning_rate":      learning_rate,
            "discount":           discount,
            "generated_at":       datetime.now().isoformat(),
        },
    }

    save_artefact(rl_policy,     RL_RESULTS_PATH)
    save_artefact(lp_result,     LP_RESULTS_PATH)
    save_artefact(full_results,  OPT_RESULTS_PATH)

    # Save CSV summary
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in rl_policy["allocation"]:
        lp_a = next((x for x in lp_result["allocation"]
                     if x["root_cause"] == a["root_cause"]), {})
        rows.append({
            "root_cause":           a["root_cause"],
            "rl_invest":            a["invested"],
            "lp_invest":            lp_a.get("invested", False),
            "cost":                 a["cost"],
            "rl_saving":            a["financial_saving"],
            "lp_saving":            lp_a.get("financial_saving", 0),
            "rl_returns_prevented": a["returns_prevented"],
            "lp_returns_prevented": lp_a.get("returns_prevented", 0),
            "roi":                  a["roi"],
        })
    pd.DataFrame(rows).to_csv(OPT_RESULTS_CSV, index=False)
    logger.info("Optimisation CSV saved → %s", OPT_RESULTS_CSV.name)

    # 8. Generate plots
    with timer("Generating visualisations"):
        generate_all_plots(history, rl_policy, lp_result)

    logger.info("=" * 60)
    logger.info("Phase 7a Operational Optimisation — COMPLETE")
    logger.info("RL  → £%.0f saving | %d returns prevented",
                rl_policy["total_saving"], rl_policy["total_returns_prevented"])
    logger.info("LP  → £%.0f saving | %d returns prevented",
                lp_result["total_saving"], lp_result["total_returns_prevented"])
    logger.info("Gap → %.1f%%", convergence_gap_pct)
    logger.info("=" * 60)

    return full_results


if __name__ == "__main__":
    run_phase7a_pipeline()
