"""
visualisations.py  –  Phase 7 recommendation visualisation plots.

Plots generated
---------------
  1. financial_impact.png       — bar chart: cost impact and potential saving per root cause
  2. recommendation_summary.png — table visual: all recommendations at a glance
  3. savings_waterfall.png      — waterfall: cumulative saving as each intervention is added
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .utils import (
    OUTPUTS_DIR,
    ROOT_CAUSE_COLOURS,
    STAKEHOLDER_MAP,
    get_logger,
)

logger = get_logger("phase7.visualisations")

FIGURE_DPI = 150
plt.rcParams.update({"font.family": "DejaVu Sans"})

# Colour constants
COST_COLOUR   = "#E8563A"
SAVING_COLOUR = "#4CAF50"


def _save(fig, filename: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved → %s", filename)


# ---------------------------------------------------------------------------
# 1. Financial impact bar chart
# ---------------------------------------------------------------------------

def plot_financial_impact(recommendations: dict) -> None:
    """
    Grouped bar chart showing annual cost exposure vs potential saving
    per root cause, sorted by priority.
    """
    sorted_recs = sorted(recommendations.values(), key=lambda x: x["priority"])
    labels      = [f"P{r['priority']} {r['root_cause'].replace(' / ', '/').replace(' & ', '/')}" for r in sorted_recs]
    costs       = [r["cost_impact"]      / 1000 for r in sorted_recs]   # £k
    savings     = [r["potential_saving"] / 1000 for r in sorted_recs]   # £k

    x     = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width/2, costs,   width, label="Annual cost exposure (£k)",
                   color=COST_COLOUR,   edgecolor="white", alpha=0.85)
    bars2 = ax.bar(x + width/2, savings, width, label="Potential annual saving (£k)",
                   color=SAVING_COLOUR, edgecolor="white", alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"£{bar.get_height():.0f}k", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"£{bar.get_height():.0f}k", ha="center", va="bottom",
                fontsize=8, color="#1B5E20", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Amount (£ thousands)", fontsize=11)
    ax.set_title("Annual Cost Exposure vs Potential Saving by Root Cause",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    _save(fig, "financial_impact.png")


# ---------------------------------------------------------------------------
# 2. Recommendation summary table
# ---------------------------------------------------------------------------

def plot_recommendation_summary(recommendations: dict) -> None:
    """Formatted table showing all recommendations at a glance."""
    sorted_recs = sorted(recommendations.values(), key=lambda x: x["priority"])

    cols = ["Priority", "Root Cause", "Stakeholder", "Returns",
            "Risk Score", "Risk Level", "Saving (£)", "Reduction %"]
    rows = []
    for r in sorted_recs:
        rows.append([
            f"P{r['priority']}",
            r["root_cause"],
            r["stakeholder"].replace(" Manager", "").replace(" Team", ""),
            f"{r['returns']:,}",
            f"{r['risk_score']:.1f}",
            r["risk_level"],
            f"£{r['potential_saving']:,.0f}",
            f"{r['reduction_pct']:.0f}%",
        ])

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")

    risk_colours = {"High": "#FFF3CD", "Medium": "#D1ECF1", "Low": "#D4EDDA", "Critical": "#F8D7DA"}
    cell_colours = []
    for r in sorted_recs:
        colour = risk_colours.get(r["risk_level"], "#FFFFFF") 
        cell_colours.append([colour] * len(cols))

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="center",
        loc="center",
        cellColours=cell_colours,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for j in range(len(cols)):
        table[(0, j)].set_facecolor("#2C5F8A")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Phase 7 — AI-Generated Recommendation Summary",
                 fontsize=13, fontweight="bold", pad=20, y=0.98)
    _save(fig, "recommendation_summary.png")


# ---------------------------------------------------------------------------
# 3. Cumulative savings waterfall
# ---------------------------------------------------------------------------

def plot_savings_waterfall(recommendations: dict) -> None:
    """
    Waterfall chart showing cumulative potential saving as each
    intervention (P1 → P6) is implemented in priority order.
    """
    sorted_recs = sorted(recommendations.values(), key=lambda x: x["priority"])
    labels      = [f"P{r['priority']}\n{r['root_cause'].split('/')[0].strip()}" for r in sorted_recs]
    savings     = [r["potential_saving"] / 1000 for r in sorted_recs]

    cumulative = []
    running = 0
    for s in savings:
        running += s
        cumulative.append(running)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw bars as stacked increments
    bottoms = [0] + cumulative[:-1]
    colours = [ROOT_CAUSE_COLOURS.get(r["root_cause"], "#2C5F8A") for r in sorted_recs]

    for i, (label, saving, bottom, colour) in enumerate(zip(labels, savings, bottoms, colours)):
        ax.bar(i, saving, bottom=bottom, color=colour, edgecolor="white",
               width=0.6, alpha=0.88)
        ax.text(i, bottom + saving/2, f"+£{saving:.0f}k",
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        ax.text(i, bottom + saving + 1, f"Total: £{cumulative[i]:.0f}k",
                ha="center", va="bottom", fontsize=8, color="#111111")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cumulative Potential Annual Saving (£ thousands)", fontsize=11)
    ax.set_title("Cumulative Saving by Implementing Interventions in Priority Order\n"
                 "(P1 first delivers highest return on investment)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, max(cumulative) * 1.18)

    patches = [mpatches.Patch(color=ROOT_CAUSE_COLOURS.get(r["root_cause"], "#2C5F8A"),
                               label=r["root_cause"]) for r in sorted_recs]
    ax.legend(handles=patches, fontsize=8, loc="upper left")
    _save(fig, "savings_waterfall.png")


# ---------------------------------------------------------------------------
# Master
# ---------------------------------------------------------------------------

def generate_all_plots(recommendations: dict) -> None:
    """Generate and save all Phase 7 visualisations."""
    logger.info("Generating Phase 7 visualisations …")
    plot_financial_impact(recommendations)
    plot_recommendation_summary(recommendations)
    plot_savings_waterfall(recommendations)
    logger.info("All Phase 7 plots saved to %s", OUTPUTS_DIR)
