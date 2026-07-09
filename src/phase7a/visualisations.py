"""
visualisations.py  –  Phase 7a optimisation plots.

Plots generated
---------------
  1. rl_training_curve.png      — episode reward and saving over training
  2. rl_convergence.png         — rolling average saving showing convergence
  3. rl_vs_lp_comparison.png    — RL policy vs LP optimum side by side
  4. budget_allocation.png      — funded interventions with cost and saving
  5. roi_ranking.png            — ROI per intervention (saving / cost)
  6. epsilon_decay.png          — exploration rate decay over episodes
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .utils import (
    PLOTS_DIR,
    ROOT_CAUSE_COLOURS,
    ROOT_CAUSE_SHORT,
    get_logger,
)

logger = get_logger("phase7a.visualisations")

FIGURE_DPI  = 150
ACCENT      = "#2C5F8A"
GREEN       = "#4CAF50"
ORANGE      = "#E8A838"
RED         = "#E8563A"

plt.rcParams.update({"font.family": "DejaVu Sans"})


def _save(fig, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved → %s", filename)


# ---------------------------------------------------------------------------
# 1. RL training curve
# ---------------------------------------------------------------------------

def plot_training_curve(history: dict) -> None:
    """Episode saving over all training episodes."""
    savings  = history["episode_savings"]
    episodes = range(1, len(savings) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episodes, savings, color=ACCENT, alpha=0.3, linewidth=0.5, label="Episode saving")

    # Rolling average
    window = max(50, len(savings) // 20)
    if len(savings) >= window:
        rolling = np.convolve(savings, np.ones(window)/window, mode="valid")
        ax.plot(range(window, len(savings)+1), rolling, color=GREEN,
                linewidth=2, label=f"Rolling avg ({window} eps)")

    ax.set_title("RL Agent Training — Financial Saving per Episode",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Financial Saving (£)", fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    _save(fig, "rl_training_curve.png")


# ---------------------------------------------------------------------------
# 2. Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(history: dict, lp_saving: float) -> None:
    """Rolling average saving vs LP optimum — shows convergence."""
    savings = history["episode_savings"]
    window  = max(100, len(savings) // 10)

    if len(savings) < window:
        return

    rolling = np.convolve(savings, np.ones(window)/window, mode="valid")
    x       = range(window, len(savings) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, rolling, color=ACCENT, linewidth=2, label=f"RL rolling avg ({window} eps)")
    ax.axhline(lp_saving, color=RED, linestyle="--", linewidth=2,
               label=f"LP optimum: £{lp_saving:,.0f}")

    # Convergence gap annotation
    final_rl = rolling[-1] if len(rolling) > 0 else 0
    gap_pct  = abs(lp_saving - final_rl) / lp_saving * 100 if lp_saving > 0 else 0
    ax.annotate(
        f"Convergence gap: {gap_pct:.1f}%",
        xy=(len(x), final_rl),
        xytext=(len(x) * 0.7, (final_rl + lp_saving) / 2),
        fontsize=10, color="#333333",
        arrowprops={"arrowstyle": "->", "color": "#333333"},
    )

    ax.set_title("RL Agent Convergence vs LP Mathematical Optimum",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Rolling Average Saving (£)", fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    _save(fig, "rl_convergence.png")


# ---------------------------------------------------------------------------
# 3. RL vs LP side-by-side comparison
# ---------------------------------------------------------------------------

def plot_rl_vs_lp(rl_policy: dict, lp_result: dict) -> None:
    """Side-by-side comparison of RL and LP allocation decisions."""
    labels = [ROOT_CAUSE_SHORT.get(rc, rc) for rc in
              [a["root_cause"] for a in lp_result["allocation"]]]

    lp_funded = [1 if a["invested"] else 0 for a in lp_result["allocation"]]
    rl_funded = {a["root_cause"]: a["invested"] for a in rl_policy["allocation"]}
    rl_vals   = [1 if rl_funded.get(
        lp_result["allocation"][i]["root_cause"], False) else 0
                 for i in range(len(lp_result["allocation"]))]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, lp_funded, width, label="LP decision",
                   color=[GREEN if v else RED for v in lp_funded], edgecolor="white", alpha=0.85)
    bars2 = ax.bar(x + width/2, rl_vals, width, label="RL decision",
                   color=[ACCENT if v else ORANGE for v in rl_vals], edgecolor="white", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Skip", "Fund"], fontsize=10)
    ax.set_title("RL vs LP Budget Allocation Decisions by Root Cause",
                 fontsize=13, fontweight="bold", pad=12)

    patches = [
        mpatches.Patch(color=GREEN,  label=f"LP: Fund (saving £{lp_result['total_saving']:,.0f})"),
        mpatches.Patch(color=ACCENT, label=f"RL: Fund (saving £{rl_policy['total_saving']:,.0f})"),
        mpatches.Patch(color=RED,    label="LP: Skip"),
        mpatches.Patch(color=ORANGE, label="RL: Skip"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="upper right")

    # Agreement annotation
    agreements = sum(1 for l, r in zip(lp_funded, rl_vals) if l == r)
    ax.text(0.02, 0.95, f"Agreement: {agreements}/{len(labels)} decisions",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5})

    _save(fig, "rl_vs_lp_comparison.png")


# ---------------------------------------------------------------------------
# 4. Budget allocation chart
# ---------------------------------------------------------------------------

def plot_budget_allocation(policy: dict, title_suffix: str = "RL") -> None:
    """Bar chart showing cost invested and saving returned per funded intervention."""
    funded = [a for a in policy["allocation"] if a["invested"]]
    if not funded:
        return

    labels  = [ROOT_CAUSE_SHORT.get(a["root_cause"], a["root_cause"]) for a in funded]
    costs   = [a["cost"]             / 1000 for a in funded]
    savings = [a["financial_saving"] / 1000 for a in funded]

    x     = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, costs,   width, label="Investment (£k)", color=RED,   edgecolor="white", alpha=0.85)
    ax.bar(x + width/2, savings, width, label="Saving (£k)",     color=GREEN, edgecolor="white", alpha=0.85)

    for i, (c, s) in enumerate(zip(costs, savings)):
        ax.text(i - width/2, c + 0.3, f"£{c:.0f}k", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, s + 0.3, f"£{s:.0f}k", ha="center", va="bottom",
                fontsize=8, color="#1B5E20", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Amount (£ thousands)", fontsize=11)
    ax.set_title(f"{title_suffix} Optimal Budget Allocation — Investment vs Financial Saving",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10)

    safe = title_suffix.lower().replace(" ", "_")
    _save(fig, f"budget_allocation_{safe}.png")


# ---------------------------------------------------------------------------
# 5. ROI ranking
# ---------------------------------------------------------------------------

def plot_roi_ranking(policy: dict) -> None:
    """Horizontal bar chart: ROI (saving / cost) per funded intervention."""
    funded = sorted(
        [a for a in policy["allocation"] if a["invested"]],
        key=lambda x: x["roi"]
    )
    if not funded:
        return

    labels = [ROOT_CAUSE_SHORT.get(a["root_cause"], a["root_cause"]) for a in funded]
    rois   = [a["roi"] for a in funded]
    colours = [ROOT_CAUSE_COLOURS.get(a["root_cause"], ACCENT) for a in funded]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels, rois, color=colours, edgecolor="white", height=0.6)
    for bar, roi in zip(bars, rois):
        ax.text(roi + 0.01, bar.get_y() + bar.get_height()/2,
                f"{roi:.2f}x", va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_title("Return on Investment per Funded Intervention (Saving / Cost)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("ROI Ratio (£ saving per £ invested)", fontsize=10)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Break-even (ROI=1)")
    ax.legend(fontsize=9)
    _save(fig, "roi_ranking.png")


# ---------------------------------------------------------------------------
# 6. Epsilon decay
# ---------------------------------------------------------------------------

def plot_epsilon_decay(history: dict) -> None:
    """Exploration rate decay showing shift from explore to exploit."""
    epsilons = history["epsilon_history"]
    fig, ax  = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(epsilons)+1), epsilons, color=ACCENT, linewidth=1.5)
    ax.fill_between(range(1, len(epsilons)+1), epsilons, alpha=0.2, color=ACCENT)
    ax.set_title("RL Agent Exploration Rate (ε) Decay over Training",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Epsilon (exploration rate)", fontsize=10)
    ax.text(0.7, 0.8, "Explore → Exploit", transform=ax.transAxes,
            fontsize=11, color=ACCENT, style="italic")
    _save(fig, "epsilon_decay.png")


# ---------------------------------------------------------------------------
# Master
# ---------------------------------------------------------------------------

def generate_all_plots(
    rl_history: dict,
    rl_policy:  dict,
    lp_result:  dict,
) -> None:
    logger.info("Generating Phase 7a visualisations …")
    plot_training_curve(rl_history)
    plot_convergence(rl_history, lp_result["total_saving"])
    plot_rl_vs_lp(rl_policy, lp_result)
    plot_budget_allocation(rl_policy, "RL")
    plot_budget_allocation(lp_result, "LP")
    plot_roi_ranking(rl_policy)
    plot_epsilon_decay(rl_history)
    logger.info("All Phase 7a plots saved to %s", PLOTS_DIR)
