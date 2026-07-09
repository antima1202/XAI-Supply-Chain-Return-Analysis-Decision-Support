"""
rl_agent.py  –  Q-learning agent for supply chain budget optimisation.

Algorithm: Q-Learning (tabular, epsilon-greedy exploration)

Q-Learning
----------
Maintains a Q-table: Q[state][action] = expected cumulative reward.

At each step:
  1. Observe current state s
  2. Choose action a (epsilon-greedy: explore or exploit)
  3. Execute action, observe reward r and next state s'
  4. Update: Q[s][a] ← Q[s][a] + α(r + γ·max Q[s'][a'] − Q[s][a])
  5. Decay epsilon (reduce exploration over time)

Why Q-Learning for this problem
--------------------------------
  - State space is small (budget bins × 2^6 bitmask = 640 states)
  - Tabular Q-learning converges exactly in this size
  - No neural network needed — keeps the implementation transparent
    and explainable (important for a dissertation)
  - Convergence can be demonstrated visually via episode reward curve

Academic framing
----------------
The Q-learning agent learns WHICH interventions to fund and IN WHAT ORDER
given budget constraints and uncertainty about intervention effectiveness.
This is fundamentally different from LP which assumes perfect knowledge.
The agent discovers through trial and error that funding high-ROI
interventions first is optimal — mirroring real supply chain decision-making
where managers learn from experience rather than solving equations.
"""

import numpy as np
from collections import defaultdict

from .rl_environment import SupplyChainEnv
from .utils import (
    ROOT_CAUSE_CLASSES,
    DEFAULT_N_EPISODES,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DISCOUNT,
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_EPSILON_MIN,
    get_logger,
)

logger = get_logger("phase7a.rl_agent")


class QLearningAgent:
    """
    Tabular Q-learning agent for supply chain budget allocation.

    Parameters
    ----------
    env            : SupplyChainEnv
    learning_rate  : float   — α (how fast to update Q-values)
    discount       : float   — γ (how much to value future rewards)
    epsilon        : float   — initial exploration rate
    epsilon_decay  : float   — exploration decay per episode
    epsilon_min    : float   — minimum exploration rate
    """

    def __init__(
        self,
        env: SupplyChainEnv,
        learning_rate:  float = DEFAULT_LEARNING_RATE,
        discount:       float = DEFAULT_DISCOUNT,
        epsilon:        float = DEFAULT_EPSILON,
        epsilon_decay:  float = DEFAULT_EPSILON_DECAY,
        epsilon_min:    float = DEFAULT_EPSILON_MIN,
    ):
        self.env           = env
        self.alpha         = learning_rate
        self.gamma         = discount
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.n_actions     = env.n_actions

        # Q-table: defaultdict so unseen states initialise to 0
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

        # Training history for plots
        self.episode_rewards   : list[float] = []
        self.episode_savings   : list[float] = []
        self.episode_prevented : list[float] = []
        self.epsilon_history   : list[float] = []

    # ---------------------------------------------------------------------------
    # Action selection (epsilon-greedy with action masking)
    # ---------------------------------------------------------------------------

    def select_action(self, state: tuple, valid_actions: list[int]) -> int:
        """
        Select action using epsilon-greedy policy.

        Exploration: choose randomly from valid actions (probability ε)
        Exploitation: choose action with highest Q-value (probability 1-ε)
        Action masking: never select invalid actions
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        q_values = self.q_table[state]
        # Mask invalid actions with -inf so they are never selected
        masked   = np.full(self.n_actions, -np.inf)
        for a in valid_actions:
            masked[a] = q_values[a]
        return int(np.argmax(masked))

    # ---------------------------------------------------------------------------
    # Q-value update
    # ---------------------------------------------------------------------------

    def update(
        self,
        state:      tuple,
        action:     int,
        reward:     float,
        next_state: tuple,
        done:       bool,
        next_valid: list[int],
    ) -> None:
        """
        Update Q-table using the Bellman equation.

        Q[s,a] ← Q[s,a] + α · (r + γ · max_a' Q[s',a'] − Q[s,a])
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            # Best Q-value for next state (masking invalid actions)
            next_q   = self.q_table[next_state]
            masked   = np.full(self.n_actions, -np.inf)
            for a in next_valid:
                masked[a] = next_q[a]
            best_next = np.max(masked) if np.any(np.isfinite(masked)) else 0.0
            target    = reward + self.gamma * best_next

        self.q_table[state][action] += self.alpha * (target - current_q)

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    def train(self, n_episodes: int = DEFAULT_N_EPISODES) -> dict:
        """
        Train the agent for n_episodes.

        Returns training history dict for visualisation.
        """
        logger.info(
            "Training Q-learning agent | episodes=%d | α=%.2f | γ=%.2f | ε=%.2f",
            n_episodes, self.alpha, self.gamma, self.epsilon,
        )

        for episode in range(n_episodes):
            state      = self.env.reset()
            total_reward = 0.0
            done       = False

            while not done:
                valid   = self.env.valid_actions()
                action  = self.select_action(state, valid)
                next_state, reward, done, _ = self.env.step(action)

                next_valid = self.env.valid_actions() if not done else []
                self.update(state, action, reward, next_state, done, next_valid)

                state        = next_state
                total_reward += reward

            # Record episode results
            summary = self.env.episode_summary()
            self.episode_rewards.append(total_reward)
            self.episode_savings.append(summary["total_saving"])
            self.episode_prevented.append(summary["total_returns_prevented"])
            self.epsilon_history.append(self.epsilon)

            # Decay exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Log every 200 episodes
            if (episode + 1) % 200 == 0:
                avg_saving = sum(self.episode_savings[-200:]) / 200
                logger.info(
                    "Episode %4d/%d | avg_saving=£%.0f | epsilon=%.3f | states=%d",
                    episode + 1, n_episodes, avg_saving,
                    self.epsilon, len(self.q_table),
                )

        logger.info("Training complete | Q-table states: %d", len(self.q_table))
        return {
            "episode_rewards":   self.episode_rewards,
            "episode_savings":   self.episode_savings,
            "episode_prevented": self.episode_prevented,
            "epsilon_history":   self.epsilon_history,
        }

    # ---------------------------------------------------------------------------
    # Extract optimal policy
    # ---------------------------------------------------------------------------

    def get_optimal_policy(self) -> dict:
        """
        Run one greedy episode (epsilon=0) to extract the learned policy.

        Returns a dict describing the optimal allocation.
        """
        original_epsilon = self.epsilon
        self.epsilon     = 0.0   # pure exploitation

        state = self.env.reset()
        done  = False
        steps = []

        while not done:
            valid  = self.env.valid_actions()
            action = self.select_action(state, valid)
            next_state, reward, done, info = self.env.step(action)
            steps.append({"action": action, "info": info, "reward": reward})
            state = next_state

        self.epsilon = original_epsilon
        summary      = self.env.episode_summary()

        # Build allocation table
        allocation = []
        for rc in summary["funded_interventions"]:
            rc_idx = ROOT_CAUSE_CLASSES.index(rc)
            cost   = self.env.intervention_costs.get(rc, 0)
            saving = self.env._savings[rc]
            prevented = self.env._return_counts[rc] * self.env.reduction_potential.get(rc, 0.3)
            allocation.append({
                "root_cause":        rc,
                "invested":          True,
                "cost":              cost,
                "returns_prevented": round(prevented),
                "financial_saving":  round(saving, 2),
                "roi":               round(saving / cost, 3) if cost > 0 else 0,
            })

        # Not funded
        for rc in ROOT_CAUSE_CLASSES:
            if rc not in summary["funded_interventions"]:
                allocation.append({
                    "root_cause":        rc,
                    "invested":          False,
                    "cost":              0,
                    "returns_prevented": 0,
                    "financial_saving":  0.0,
                    "roi":               0.0,
                })

        # Sort by ROI descending
        allocation.sort(key=lambda x: x["roi"], reverse=True)

        policy = {
            "allocation":              allocation,
            "budget_total":            self.env.budget,
            "budget_spent":            summary["budget_spent"],
            "budget_remaining":        summary["budget_remaining"],
            "total_returns_prevented": summary["total_returns_prevented"],
            "total_saving":            summary["total_saving"],
            "funded_count":            len(summary["funded_interventions"]),
        }

        logger.info("Optimal RL policy extracted:")
        logger.info("  Budget spent    : £%.0f / £%.0f", policy["budget_spent"], policy["budget_total"])
        logger.info("  Returns prevented: %d", policy["total_returns_prevented"])
        logger.info("  Total saving    : £%.0f", policy["total_saving"])
        for a in allocation:
            status = "✓ FUND" if a["invested"] else "✗ SKIP"
            logger.info("  %s  %-40s  £%.0f → saves £%.0f", status, a["root_cause"], a["cost"], a["financial_saving"])

        return policy
