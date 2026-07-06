"""
visualisations.py  –  Phase 5 Root Cause Analysis plots.

Plots generated
---------------
  1.  root_cause_distribution.png      — horizontal bar chart with counts + %
  2.  monthly_trend.png                — stacked area chart, 2-year window
  3.  monthly_trend_lines.png          — line chart per root cause (easier to read)
  4.  product_heatmap.png              — heatmap: subcategory × root cause
  5.  brand_comparison.png             — grouped bar: returns per brand by root cause
  6.  region_heatmap.png               — heatmap: region × root cause %
  7.  sentiment_by_root_cause.png      — box plot: sentiment score per root cause
  8.  sentiment_profile_bar.png        — bar chart: avg sentiment per root cause
  9.  return_lag_by_root_cause.png     — box plot: return lag days per root cause
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import (
    PLOTS_DIR,
    ROOT_CAUSE_CLASSES,
    ROOT_CAUSE_SHORT,
    ROOT_CAUSE_COLOURS,
    get_logger,
)

logger = get_logger("phase5.visualisations")

FIGURE_DPI  = 150
FONT_TITLE  = {"fontsize": 13, "fontweight": "bold", "pad": 12}
FONT_AXIS   = {"fontsize": 11}

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"font.family": "DejaVu Sans"})

COLOUR_LIST = [ROOT_CAUSE_COLOURS[c] for c in ROOT_CAUSE_CLASSES]


def _save(fig, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved → %s", filename)


# ---------------------------------------------------------------------------
# 1. Root cause distribution bar chart
# ---------------------------------------------------------------------------

def plot_root_cause_distribution(summary: pd.DataFrame) -> None:
    """Horizontal bar chart: count + percentage label per root cause."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colours = [ROOT_CAUSE_COLOURS.get(rc, "#2C5F8A") for rc in summary["root_cause_category"]]
    bars = ax.barh(
        summary["root_cause_category"],
        summary["count"],
        color=colours,
        edgecolor="white",
        height=0.65,
    )

    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(
            bar.get_width() + 20,
            bar.get_y() + bar.get_height() / 2,
            f"{row['count']:,}  ({row['pct_of_returns']:.1f}%)",
            va="center", ha="left", fontsize=9,
        )

    ax.set_title("Root Cause Category Distribution — Confirmed Returns", **FONT_TITLE)
    ax.set_xlabel("Number of Returns", **FONT_AXIS)
    ax.set_xlim(0, summary["count"].max() * 1.22)
    ax.invert_yaxis()
    _save(fig, "root_cause_distribution.png")


# ---------------------------------------------------------------------------
# 2. Monthly trend — stacked area
# ---------------------------------------------------------------------------

def plot_monthly_trend_stacked(monthly_trend: pd.DataFrame) -> None:
    """Stacked area chart showing volume of each root cause over time."""
    fig, ax = plt.subplots(figsize=(14, 6))

    months = [str(p) for p in monthly_trend.index]
    x = np.arange(len(months))

    bottom = np.zeros(len(months))
    for cls in ROOT_CAUSE_CLASSES:
        if cls in monthly_trend.columns:
            vals = monthly_trend[cls].values.astype(float)
            ax.fill_between(x, bottom, bottom + vals,
                            label=ROOT_CAUSE_SHORT[cls],
                            color=ROOT_CAUSE_COLOURS[cls],
                            alpha=0.85)
            bottom += vals

    # X-axis: show every 3rd month to avoid crowding
    tick_step = max(1, len(months) // 10)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(months[::tick_step], rotation=45, ha="right", fontsize=8)

    ax.set_title("Monthly Return Volume by Root Cause Category (2-Year Trend)", **FONT_TITLE)
    ax.set_xlabel("Month", **FONT_AXIS)
    ax.set_ylabel("Number of Returns", **FONT_AXIS)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    _save(fig, "monthly_trend_stacked.png")


# ---------------------------------------------------------------------------
# 3. Monthly trend — line chart
# ---------------------------------------------------------------------------

def plot_monthly_trend_lines(monthly_trend: pd.DataFrame) -> None:
    """Line chart — one line per root cause, easier to read individual trends."""
    fig, ax = plt.subplots(figsize=(14, 6))

    months = [str(p) for p in monthly_trend.index]
    x = np.arange(len(months))

    for cls in ROOT_CAUSE_CLASSES:
        if cls in monthly_trend.columns:
            ax.plot(
                x,
                monthly_trend[cls].values,
                label=ROOT_CAUSE_SHORT[cls],
                color=ROOT_CAUSE_COLOURS[cls],
                linewidth=2,
                marker="o",
                markersize=3,
            )

    tick_step = max(1, len(months) // 10)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(months[::tick_step], rotation=45, ha="right", fontsize=8)

    ax.set_title("Monthly Return Trend by Root Cause Category", **FONT_TITLE)
    ax.set_xlabel("Month", **FONT_AXIS)
    ax.set_ylabel("Number of Returns", **FONT_AXIS)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    _save(fig, "monthly_trend_lines.png")


# ---------------------------------------------------------------------------
# 4. Product subcategory heatmap
# ---------------------------------------------------------------------------

def plot_product_heatmap(product: pd.DataFrame) -> None:
    """Heatmap: product subcategory (rows) × root cause (cols), % within subcategory."""
    pivot = product.pivot(
        index="product_subcategory",
        columns="root_cause_category",
        values="pct_within_subcategory",
    ).fillna(0)

    # Reorder columns to consistent order
    cols = [c for c in ROOT_CAUSE_CLASSES if c in pivot.columns]
    pivot = pivot[cols]
    pivot.columns = [ROOT_CAUSE_SHORT[c] for c in cols]

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "% of returns within subcategory"},
    )
    ax.set_title("Root Cause Distribution by Product Subcategory (%)", **FONT_TITLE)
    ax.set_xlabel("Root Cause Category", **FONT_AXIS)
    ax.set_ylabel("Product Subcategory", **FONT_AXIS)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    _save(fig, "product_heatmap.png")


# ---------------------------------------------------------------------------
# 5. Brand comparison grouped bar
# ---------------------------------------------------------------------------

def plot_brand_comparison(brand_detail: pd.DataFrame) -> None:
    """Grouped bar chart: returns per brand coloured by root cause."""
    pivot = brand_detail.pivot(
        index="brand",
        columns="root_cause_category",
        values="count",
    ).fillna(0)

    cols = [c for c in ROOT_CAUSE_CLASSES if c in pivot.columns]
    pivot = pivot[cols]

    n_brands = len(pivot)
    n_causes = len(cols)
    x = np.arange(n_brands)
    width = 0.12

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, cls in enumerate(cols):
        offset = (i - n_causes / 2) * width + width / 2
        ax.bar(
            x + offset,
            pivot[cls].values,
            width=width,
            label=ROOT_CAUSE_SHORT[cls],
            color=ROOT_CAUSE_COLOURS[cls],
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=10)
    ax.set_title("Return Count by Brand and Root Cause Category", **FONT_TITLE)
    ax.set_xlabel("Brand", **FONT_AXIS)
    ax.set_ylabel("Number of Returns", **FONT_AXIS)
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    _save(fig, "brand_comparison.png")


# ---------------------------------------------------------------------------
# 6. Region heatmap
# ---------------------------------------------------------------------------

def plot_region_heatmap(region: pd.DataFrame) -> None:
    """Heatmap: region (rows) × root cause (cols), % within region."""
    pivot = region.pivot(
        index="region",
        columns="root_cause_category",
        values="pct_within_region",
    ).fillna(0)

    cols = [c for c in ROOT_CAUSE_CLASSES if c in pivot.columns]
    pivot = pivot[cols]
    pivot.columns = [ROOT_CAUSE_SHORT[c] for c in cols]

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "% of returns within region"},
    )
    ax.set_title("Root Cause Distribution by Region (%)", **FONT_TITLE)
    ax.set_xlabel("Root Cause Category", **FONT_AXIS)
    ax.set_ylabel("Region", **FONT_AXIS)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    _save(fig, "region_heatmap.png")


# ---------------------------------------------------------------------------
# 7. Sentiment boxplot by root cause
# ---------------------------------------------------------------------------

def plot_sentiment_boxplot(rich: pd.DataFrame) -> None:
    """Box plot: VADER sentiment score distribution per root cause."""
    order = (
        rich.groupby("root_cause_category")["sentiment_score"]
        .median()
        .sort_values()
        .index.tolist()
    )
    palette = {rc: ROOT_CAUSE_COLOURS[rc] for rc in order if rc in ROOT_CAUSE_COLOURS}

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=rich,
        x="sentiment_score",
        y="root_cause_category",
        order=order,
        hue="root_cause_category",
        palette=palette,
        width=0.6,
        linewidth=1.3,
        legend=False,
        ax=ax,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Neutral (0)")
    ax.set_title("VADER Sentiment Score Distribution by Root Cause Category", **FONT_TITLE)
    ax.set_xlabel("VADER Compound Sentiment Score (−1 = most negative, +1 = most positive)", **FONT_AXIS)
    ax.set_ylabel("")
    ax.legend(fontsize=9)
    _save(fig, "sentiment_by_root_cause.png")


# ---------------------------------------------------------------------------
# 8. Average sentiment bar
# ---------------------------------------------------------------------------

def plot_sentiment_profile(sentiment: pd.DataFrame) -> None:
    """Horizontal bar chart: avg sentiment per root cause with std error bars."""
    df = sentiment.sort_values("avg_sentiment")
    colours = [ROOT_CAUSE_COLOURS.get(rc, "#2C5F8A") for rc in df["root_cause_category"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        df["root_cause_category"],
        df["avg_sentiment"],
        xerr=df["std_sentiment"],
        color=colours,
        edgecolor="white",
        height=0.6,
        capsize=4,
        error_kw={"linewidth": 1.2, "ecolor": "grey"},
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("Average VADER Sentiment Score by Root Cause Category (± std)", **FONT_TITLE)
    ax.set_xlabel("Mean VADER Compound Score", **FONT_AXIS)
    ax.set_ylabel("")

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["avg_sentiment"] + (0.02 if row["avg_sentiment"] >= 0 else -0.02),
            i,
            f"{row['avg_sentiment']:.3f}",
            va="center",
            ha="left" if row["avg_sentiment"] >= 0 else "right",
            fontsize=9,
        )
    _save(fig, "sentiment_profile_bar.png")


# ---------------------------------------------------------------------------
# 9. Return lag boxplot
# ---------------------------------------------------------------------------

def plot_return_lag(rich: pd.DataFrame) -> None:
    """Box plot: days from delivery to return request per root cause."""
    order = (
        rich.groupby("root_cause_category")["return_lag_days"]
        .median()
        .sort_values()
        .index.tolist()
    )
    palette = {rc: ROOT_CAUSE_COLOURS[rc] for rc in order if rc in ROOT_CAUSE_COLOURS}

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=rich,
        x="return_lag_days",
        y="root_cause_category",
        order=order,
        hue="root_cause_category",
        palette=palette,
        width=0.6,
        linewidth=1.3,
        legend=False,
        ax=ax,
    )
    ax.set_title("Days from Delivery to Return Request by Root Cause Category", **FONT_TITLE)
    ax.set_xlabel("Return Lag (days)", **FONT_AXIS)
    ax.set_ylabel("")
    _save(fig, "return_lag_by_root_cause.png")


# ---------------------------------------------------------------------------
# Master plot function
# ---------------------------------------------------------------------------

def generate_all_plots(results: dict) -> None:
    """Generate and save all Phase 5 visualisations."""
    logger.info("Generating Phase 5 visualisations …")
    plot_root_cause_distribution(results["summary"])
    plot_monthly_trend_stacked(results["monthly_trend"])
    plot_monthly_trend_lines(results["monthly_trend"])
    plot_product_heatmap(results["product"])
    plot_brand_comparison(results["brand_detail"])
    plot_region_heatmap(results["region"])
    plot_sentiment_boxplot(results["rich"])
    plot_sentiment_profile(results["sentiment"])
    plot_return_lag(results["rich"])
    logger.info("All Phase 5 plots saved to %s", PLOTS_DIR)
