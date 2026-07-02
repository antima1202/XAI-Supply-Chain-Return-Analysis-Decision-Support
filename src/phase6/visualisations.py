"""
visualisations.py  –  Phase 6 Risk Assessment plots.

Plots generated
---------------
  1.  risk_priority_ranking.png     — horizontal bar chart ranked by risk score
  2.  risk_matrix.png               — 2×2 bubble plot: frequency vs impact
  3.  risk_score_breakdown.png      — stacked bar: freq + impact + trend components
  4.  risk_level_donut.png          — donut chart: % of returns by risk level
  5.  risk_vs_sentiment.png         — scatter: risk score vs avg sentiment
  6.  risk_score_table.png          — table visual for dissertation
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .utils import (
    PLOTS_DIR,
    ROOT_CAUSE_SHORT,
    ROOT_CAUSE_COLOURS,
    RISK_LEVEL_COLOURS,
    get_logger,
)

logger = get_logger("phase6.visualisations")

FIGURE_DPI = 150
FONT_TITLE = {"fontsize": 13, "fontweight": "bold", "pad": 12}
FONT_AXIS  = {"fontsize": 11}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans"})


def _save(fig, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved → %s", filename)


# ---------------------------------------------------------------------------
# 1. Priority ranking bar chart
# ---------------------------------------------------------------------------

def plot_priority_ranking(risk_df: pd.DataFrame) -> None:
    """
    Horizontal bar chart ranked by risk score with risk level colour coding
    and priority label (P1–P6).
    """
    df = risk_df.sort_values("risk_score", ascending=True)   # ascending for barh
    colours = [RISK_LEVEL_COLOURS[lvl] for lvl in df["risk_level"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(
        df["root_cause_category"],
        df["risk_score"],
        color=colours,
        edgecolor="white",
        height=0.65,
    )

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"P{row['priority']}  {row['risk_score']:.1f}  [{row['risk_level']}]",
            va="center", ha="left", fontsize=9, fontweight="bold",
        )

    # Legend for risk levels
    patches = [
        mpatches.Patch(color=c, label=lvl)
        for lvl, c in RISK_LEVEL_COLOURS.items()
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9, title="Risk Level")

    ax.set_title("Supply Chain Risk Priority Ranking by Root Cause Category", **FONT_TITLE)
    ax.set_xlabel("Composite Risk Score (0–100)", **FONT_AXIS)
    ax.set_xlim(0, 115)
    _save(fig, "risk_priority_ranking.png")


# ---------------------------------------------------------------------------
# 2. Risk matrix — 2×2 bubble plot
# ---------------------------------------------------------------------------

def plot_risk_matrix(risk_df: pd.DataFrame) -> None:
    """
    2×2 risk matrix: X = frequency score, Y = impact score.
    Bubble size = trend score. Colour = risk level.
    Quadrant labels: Low / Medium / High / Critical risk zones.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Quadrant shading
    ax.fill_between([0, 50],  [0, 0],   [50, 50],  alpha=0.06, color="#7ED321")   # Low
    ax.fill_between([50, 100],[0, 0],   [50, 50],  alpha=0.06, color="#4A90D9")   # Medium
    ax.fill_between([0, 50],  [50, 50], [100, 100],alpha=0.06, color="#E8A838")   # High
    ax.fill_between([50, 100],[50, 50], [100, 100],alpha=0.06, color="#D94F3D")   # Critical

    # Quadrant labels
    for (x, y, label) in [(25, 25, "LOW RISK"), (75, 25, "MEDIUM RISK"),
                           (25, 75, "HIGH RISK"), (75, 75, "CRITICAL RISK")]:
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                color="grey", alpha=0.6, fontweight="bold")

    # Bubbles
    for _, row in risk_df.iterrows():
        cls   = row["root_cause_category"]
        colour = RISK_LEVEL_COLOURS[row["risk_level"]]
        size   = (row["trend_score"] / 100) * 1200 + 300   # bubble 300–1500

        ax.scatter(
            row["frequency_score"],
            row["impact_score"],
            s=size,
            color=colour,
            edgecolors="white",
            linewidth=1.5,
            alpha=0.88,
            zorder=3,
        )
        ax.annotate(
            ROOT_CAUSE_SHORT[cls],
            (row["frequency_score"], row["impact_score"]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=8.5,
            fontweight="bold",
        )

    ax.axhline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_title(
        "Risk Matrix — Frequency vs Operational Impact\n(bubble size = trend score)",
        **FONT_TITLE,
    )
    ax.set_xlabel("Frequency Score (normalised return count, 0–100)", **FONT_AXIS)
    ax.set_ylabel("Operational Impact Score (domain-defined, 0–100)", **FONT_AXIS)
    ax.set_xlim(-5, 110)
    ax.set_ylim(-5, 110)

    patches = [mpatches.Patch(color=c, label=lvl) for lvl, c in RISK_LEVEL_COLOURS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9, title="Risk Level")
    _save(fig, "risk_matrix.png")


# ---------------------------------------------------------------------------
# 3. Score component breakdown — stacked bar
# ---------------------------------------------------------------------------

def plot_score_breakdown(risk_df: pd.DataFrame) -> None:
    """
    Stacked bar chart showing how each component contributes to the total
    risk score. Weighted contributions: freq×0.4, impact×0.4, trend×0.2.
    """
    df = risk_df.sort_values("risk_score", ascending=False)
    labels = [ROOT_CAUSE_SHORT[c] for c in df["root_cause_category"]]

    freq_contrib   = df["frequency_score"] * 0.40
    impact_contrib = df["impact_score"]    * 0.40
    trend_contrib  = df["trend_score"]     * 0.20

    x = np.arange(len(labels))
    width = 0.55

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x, freq_contrib,                    width, label="Frequency (×0.4)", color="#4A90D9", edgecolor="white")
    b2 = ax.bar(x, impact_contrib, freq_contrib,
                width, label="Impact (×0.4)",     color="#E8563A", edgecolor="white")
    b3 = ax.bar(x, trend_contrib,  freq_contrib + impact_contrib,
                width, label="Trend (×0.2)",      color="#F5A623", edgecolor="white")

    # Total score annotation on top
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row["risk_score"] + 0.8, f"{row['risk_score']:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Risk Score Component Breakdown by Root Cause Category", **FONT_TITLE)
    ax.set_ylabel("Weighted Score Contribution (0–100)", **FONT_AXIS)
    ax.legend(fontsize=10, loc="upper right")
    _save(fig, "risk_score_breakdown.png")


# ---------------------------------------------------------------------------
# 4. Risk level donut — % of returns by risk level
# ---------------------------------------------------------------------------

def plot_risk_level_donut(risk_df: pd.DataFrame) -> None:
    """Donut chart: share of total returns falling in each risk level."""
    level_counts = (
        risk_df.groupby("risk_level")["count"].sum()
        .reindex(["Critical", "High", "Medium", "Low"])
        .dropna()
    )
    total = level_counts.sum()

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        level_counts,
        labels=level_counts.index,
        colors=[RISK_LEVEL_COLOURS[lvl] for lvl in level_counts.index],
        autopct=lambda p: f"{p:.1f}%\n({int(p/100*total):,})",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")

    ax.set_title(
        "Confirmed Returns by Risk Level\n(% of 14,000 returned orders)",
        **FONT_TITLE,
    )
    # Centre annotation
    ax.text(0, 0, f"14,000\nReturns", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#333333")
    _save(fig, "risk_level_donut.png")


# ---------------------------------------------------------------------------
# 5. Risk score vs sentiment scatter
# ---------------------------------------------------------------------------

def plot_risk_vs_sentiment(risk_df: pd.DataFrame) -> None:
    """
    Scatter plot: risk score (X) vs average sentiment score (Y).
    Shows the relationship between operational risk and customer tone.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in risk_df.iterrows():
        cls    = row["root_cause_category"]
        colour = RISK_LEVEL_COLOURS[row["risk_level"]]
        ax.scatter(
            row["risk_score"],
            row["avg_sentiment"],
            s=row["count"] / 10,       # bubble size = return count / 10
            color=colour,
            edgecolors="white",
            linewidth=1.5,
            alpha=0.88,
            zorder=3,
        )
        ax.annotate(
            ROOT_CAUSE_SHORT[cls],
            (row["risk_score"], row["avg_sentiment"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8.5,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4, label="Neutral sentiment")
    ax.set_title("Risk Score vs Average Customer Sentiment by Root Cause", **FONT_TITLE)
    ax.set_xlabel("Composite Risk Score (0–100)", **FONT_AXIS)
    ax.set_ylabel("Average VADER Sentiment Score", **FONT_AXIS)
    ax.legend(fontsize=9)
    ax.text(0.98, 0.02, "Bubble size ∝ number of returns",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="grey")
    _save(fig, "risk_vs_sentiment.png")


# ---------------------------------------------------------------------------
# 6. Risk score summary table plot
# ---------------------------------------------------------------------------

def plot_risk_table(risk_df: pd.DataFrame) -> None:
    """
    Render a formatted table as a matplotlib figure — dissertation-ready.
    Rows are colour-coded by risk level.
    """
    cols = ["Priority", "Root Cause", "Returns", "% Share",
            "Risk Score", "Risk Level", "Freq Score", "Impact Score", "Trend Score"]

    rows = []
    for _, r in risk_df.iterrows():
        rows.append([
            f"P{r['priority']}",
            r["root_cause_category"],
            f"{r['count']:,}",
            f"{r['pct_of_returns']:.1f}%",
            f"{r['risk_score']:.1f}",
            r["risk_level"],
            f"{r['frequency_score']:.1f}",
            f"{r['impact_score']:.1f}",
            f"{r['trend_score']:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.axis("off")

    cell_colours = []
    for r in risk_df.itertuples():
        lvl_colour = RISK_LEVEL_COLOURS[r.risk_level] + "33"   # 20% opacity
        cell_colours.append([lvl_colour] * len(cols))

    table = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="center",
        loc="center",
        cellColours=cell_colours,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Bold headers
    for j in range(len(cols)):
        table[(0, j)].set_facecolor("#2C5F8A")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Supply Chain Risk Assessment Summary Table",
                 fontsize=13, fontweight="bold", pad=20, y=0.98)
    _save(fig, "risk_score_table.png")


# ---------------------------------------------------------------------------
# Master plot function
# ---------------------------------------------------------------------------

def generate_all_plots(risk_df: pd.DataFrame) -> None:
    """Generate and save all Phase 6 risk visualisations."""
    logger.info("Generating Phase 6 risk visualisations …")
    plot_priority_ranking(risk_df)
    plot_risk_matrix(risk_df)
    plot_score_breakdown(risk_df)
    plot_risk_level_donut(risk_df)
    plot_risk_vs_sentiment(risk_df)
    plot_risk_table(risk_df)
    logger.info("All Phase 6 plots saved to %s", PLOTS_DIR)
