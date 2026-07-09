"""
rl_environment.py  –  Supply chain simulation environment for the RL agent.

Design
------
The environment models the budget allocation problem as a sequential
decision process. At each step the agent chooses one root cause to invest
in. The environment returns a reward based on the simulated return reduction
and updates the remaining budget.

State space
-----------
A tuple of (budget_remaining_bin, interventions_applied_bitmask)

  budget_remaining_bin : int
      Remaining budget discretised into 10 bins (0=empty, 9=full budget)

  interventions_applied : frozenset
      Which root causes have already been invested in this episode

The state space is kept small and discrete so Q-learning converges
in a reasonable number of episodes without needing deep RL.

Action space
------------
0–5 : invest in root cause i (if budget allows and not already done)
6   : stop (end episode, bank current returns prevented)

Reward structure
----------------
  Positive reward : returns_prevented × cost_per_return  (financial saving)
  Penalty         : -500 if trying to invest with insufficient budget
  Penalty         : -100 if trying to invest in an already-done root cause
  Terminal reward : 0 (agent already collected rewards at each step)

The agent learns to allocate budget to maximise cumulative financial saving
while respecting the budget constraint.
"""

import numpy as np
from typing import Any

from .utils import (
    ROOT_CAUSE_CLASSES,
    DEFAULT_BUDGET,
    DEFAULT_COST_PER_RETURN,
    DEFAULT_INTERVENTION_COSTS,
    DEFAULT_REDUCTION_POTENTIAL,
    get_logger,
)

logger = get_logger("phase7a.environment")

N_ACTIONS     = len(ROOT_CAUSE_CLASSES) + 1   # 6 root causes + stop
BUDGET_BINS   = 10                              # discretisation resolution


class SupplyChainEnv:
    """
    Discrete supply chain budget allocation environment.

    Parameters
    ----------
    risk_df            : pd.DataFrame — Phase 6 risk scores
    budget             : float        — total intervention budget (£)
    cost_per_return    : float        — reverse logistics cost per return (£)
    intervention_costs : dict         — minimum cost per root cause (£)
    reduction_potential: dict         — expected reduction % per root cause
    """

    def __init__(
        self,
        risk_df,
        budget: float = DEFAULT_BUDGET,
        cost_per_return: float = DEFAULT_COST_PER_RETURN,
        intervention_costs: dict = None,
        reduction_potential: dict = None,
    ):
        self.risk_df             = risk_df.set_index("root_cause_category")
        self.budget              = budget
        self.cost_per_return     = cost_per_return
        self.intervention_costs  = intervention_costs or DEFAULT_INTERVENTION_COSTS
        self.reduction_potential = reduction_potential or DEFAULT_REDUCTION_POTENTIAL
        self.n_actions           = N_ACTIONS
        self.n_root_causes       = len(ROOT_CAUSE_CLASSES)

        # Pre-compute return counts per root cause for fast reward calculation
        self._return_counts = {
            rc: int(self.risk_df.loc[rc, "count"])
            if rc in self.risk_df.index else 0
            for rc in ROOT_CAUSE_CLASSES
        }

        # Pre-compute financial saving per root cause if fully funded
        self._savings = {
            rc: self._return_counts[rc] * self.reduction_potential.get(rc, 0.3) * cost_per_return
            for rc in ROOT_CAUSE_CLASSES
        }

        self.reset()
        logger.info(
            "Environment initialised | budget=£%.0f | %d root causes | %d actions",
            budget, self.n_root_causes, self.n_actions,
        )

    # ---------------------------------------------------------------------------
    # State representation
    # ---------------------------------------------------------------------------

    def _get_state(self) -> tuple:
        """
        Return the current state as a hashable tuple.

        State = (budget_bin, interventions_bitmask)
        budget_bin in [0, BUDGET_BINS-1]
        bitmask: integer where bit i=1 means root cause i was funded
        """
        budget_bin   = min(
            int((self.remaining_budget / self.budget) * BUDGET_BINS),
            BUDGET_BINS - 1,
        )
        bitmask = sum(
            (1 << i) for i, rc in enumerate(ROOT_CAUSE_CLASSES)
            if rc in self.funded_interventions
        )
        return (budget_bin, bitmask)

    # ---------------------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------------------

    def reset(self) -> tuple:
        """Reset environment to start of a new episode."""
        self.remaining_budget       = self.budget
        self.funded_interventions   = set()
        self.total_returns_prevented = 0.0
        self.total_saving            = 0.0
        self.done                    = False
        self.steps                   = 0
        return self._get_state()

    # ---------------------------------------------------------------------------
    # Step
    # ---------------------------------------------------------------------------

    def step(self, action: int) -> tuple[tuple, float, bool, dict]:
        """
        Execute one action and return (next_state, reward, done, info).

        Parameters
        ----------
        action : int
            0–5 : invest in ROOT_CAUSE_CLASSES[action]
            6   : stop episode

        Returns
        -------
        next_state : tuple
        reward     : float   — financial saving achieved by this action
        done       : bool
        info       : dict    — diagnostic information
        """
        if self.done:
            return self._get_state(), 0.0, True, {}

        self.steps += 1
        reward = 0.0
        info   = {}

        # Action 6 = stop
        if action == self.n_root_causes:
            self.done = True
            info["action"] = "STOP"
            return self._get_state(), 0.0, True, info

        rc   = ROOT_CAUSE_CLASSES[action]
        cost = self.intervention_costs.get(rc, 30_000.0)

        # --- Penalty: already funded ---
        if rc in self.funded_interventions:
            reward = -100.0
            info["action"] = f"DUPLICATE:{rc}"
            return self._get_state(), reward, False, info

        # --- Penalty: insufficient budget ---
        if cost > self.remaining_budget:
            reward = -500.0
            info["action"] = f"NO_BUDGET:{rc}"
            # If no action is affordable, force stop
            affordable = [
                r for r in ROOT_CAUSE_CLASSES
                if r not in self.funded_interventions
                and self.intervention_costs.get(r, 0) <= self.remaining_budget
            ]
            if not affordable:
                self.done = True
                return self._get_state(), reward, True, info
            return self._get_state(), reward, False, info

        # --- Valid investment ---
        self.remaining_budget -= cost
        self.funded_interventions.add(rc)

        # Reward = financial saving from this intervention
        saving = self._savings[rc]
        returns_prevented = self._return_counts[rc] * self.reduction_potential.get(rc, 0.3)
        reward = saving

        self.total_saving            += saving
        self.total_returns_prevented += returns_prevented

        info = {
            "action":            f"INVEST:{rc}",
            "cost":              cost,
            "saving":            saving,
            "returns_prevented": returns_prevented,
            "budget_remaining":  self.remaining_budget,
        }

        # Auto-stop if all root causes funded or budget exhausted
        affordable = [
            r for r in ROOT_CAUSE_CLASSES
            if r not in self.funded_interventions
            and self.intervention_costs.get(r, 0) <= self.remaining_budget
        ]
        if not affordable:
            self.done = True

        return self._get_state(), reward, self.done, info

    # ---------------------------------------------------------------------------
    # Action masking
    # ---------------------------------------------------------------------------

    def valid_actions(self) -> list[int]:
        """Return list of currently valid action indices."""
        valid = [self.n_root_causes]  # stop is always valid
        for i, rc in enumerate(ROOT_CAUSE_CLASSES):
            if (rc not in self.funded_interventions and
                    self.intervention_costs.get(rc, 0) <= self.remaining_budget):
                valid.append(i)
        return valid

    # ---------------------------------------------------------------------------
    # Episode summary
    # ---------------------------------------------------------------------------

    def episode_summary(self) -> dict:
        """Return summary of the completed episode."""
        return {
            "funded_interventions":    list(self.funded_interventions),
            "budget_spent":            self.budget - self.remaining_budget,
            "budget_remaining":        self.remaining_budget,
            "total_returns_prevented": round(self.total_returns_prevented),
            "total_saving":            round(self.total_saving, 2),
            "n_steps":                 self.steps,
        }
