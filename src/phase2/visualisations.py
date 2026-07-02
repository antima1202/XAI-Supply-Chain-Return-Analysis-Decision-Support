"""
visualisations.py  –  Dissertation-quality plots for the Phase 2 NLP pipeline.

Generates and saves:
  1.  sentiment_distribution.png       — VADER score histogram by return status
  2.  sentiment_boxplot.png            — Box-plot RICH vs LEAN sentiment
  3.  subjectivity_distribution.png    — TextBlob subjectivity histogram (RICH)
  4.  topic_distribution.png           — LDA dominant topic bar chart
  5.  complaint_category_distribution.png — Complaint category bar chart
  6.  keyword_frequency.png            — Top-30 TF-IDF keywords (RICH corpus)
  7.  wordcloud_rich.png               — Word cloud from RICH review text
  8.  sentiment_vs_rating.png          — VADER score by review_rating
  9.  topic_by_root_cause.png          — Heatmap: LDA topic × root_cause_category

All figures use a consistent academic style suitable for dissertation inclusion.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from .utils import PLOTS_DIR, get_logger

logger = get_logger("phase2.visualisations")

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

PALETTE_RETURN  = {"Returned": "#E8563A", "Not Returned": "#4A90D9"}
ACCENT_COLOUR   = "#2C5F8A"
FIGURE_DPI      = 150
FIGURE_SIZE_STD = (10, 5)
FIGURE_SIZE_SQ  = (8, 8)
FONT_TITLE      = {"fontsize": 14, "fontweight": "bold", "pad": 12}
FONT_AXIS       = {"fontsize": 11}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlepad": 12,
    "figure.dpi": FIGURE_DPI,
})


def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure to the Phase 2 plots directory."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved plot → %s", path.name)


# ---------------------------------------------------------------------------
# 1. Sentiment histogram
# ---------------------------------------------------------------------------

def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    """VADER compound score distribution split by return status."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STD)

    returned     = df[df["is_returned"] == 1]["sentiment_score"].astype(float)
    not_returned = df[df["is_returned"] == 0]["sentiment_score"].astype(float)

    ax.hist(not_returned, bins=50, alpha=0.6, color=PALETTE_RETURN["Not Returned"],
            label="Not Returned", density=True)
    ax.hist(returned,     bins=50, alpha=0.6, color=PALETTE_RETURN["Returned"],
            label="Returned",     density=True)

    ax.axvline(returned.mean(),     color=PALETTE_RETURN["Returned"],
               linestyle="--", linewidth=1.5, label=f"Mean (Returned) = {returned.mean():.3f}")
    ax.axvline(not_returned.mean(), color=PALETTE_RETURN["Not Returned"],
               linestyle="--", linewidth=1.5, label=f"Mean (Not Returned) = {not_returned.mean():.3f}")

    ax.set_title("VADER Sentiment Score Distribution by Return Status", **FONT_TITLE)
    ax.set_xlabel("VADER Compound Score (−1 = most negative, +1 = most positive)", **FONT_AXIS)
    ax.set_ylabel("Density", **FONT_AXIS)
    ax.legend(fontsize=10)
    _save(fig, "sentiment_distribution.png")


# ---------------------------------------------------------------------------
# 2. Sentiment box-plot
# ---------------------------------------------------------------------------

def plot_sentiment_boxplot(df: pd.DataFrame) -> None:
    """Box-plot comparing VADER scores for returned vs non-returned orders."""
    plot_df = df.copy()
    plot_df["Return Status"] = plot_df["is_returned"].map({1: "Returned", 0: "Not Returned"})
    plot_df["sentiment_score"] = plot_df["sentiment_score"].astype(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=plot_df, x="Return Status", y="sentiment_score",
        hue="Return Status", palette=PALETTE_RETURN,
        width=0.5, linewidth=1.5, legend=False, ax=ax,
    )
    ax.set_title("VADER Sentiment Score: Returned vs Non-Returned Orders", **FONT_TITLE)
    ax.set_xlabel("Return Status", **FONT_AXIS)
    ax.set_ylabel("VADER Compound Score", **FONT_AXIS)
    _save(fig, "sentiment_boxplot.png")


# ---------------------------------------------------------------------------
# 3. Subjectivity histogram (RICH only)
# ---------------------------------------------------------------------------

def plot_subjectivity_distribution(df: pd.DataFrame) -> None:
    """TextBlob subjectivity distribution for returned reviews."""
    rich = df[df["is_returned"] == 1]["subjectivity_score"].astype(float)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STD)
    ax.hist(rich, bins=40, color=ACCENT_COLOUR, alpha=0.85, edgecolor="white")
    ax.axvline(rich.mean(), color="#E8563A", linestyle="--", linewidth=1.8,
               label=f"Mean = {rich.mean():.3f}")
    ax.set_title("TextBlob Subjectivity Score — Returned Reviews", **FONT_TITLE)
    ax.set_xlabel("Subjectivity Score (0 = Objective, 1 = Subjective)", **FONT_AXIS)
    ax.set_ylabel("Count", **FONT_AXIS)
    ax.legend(fontsize=10)
    _save(fig, "subjectivity_distribution.png")


# ---------------------------------------------------------------------------
# 4. Topic distribution bar chart
# ---------------------------------------------------------------------------

def plot_topic_distribution(df: pd.DataFrame) -> None:
    """Dominant LDA topic counts for returned reviews."""
    rich = df[df["is_returned"] == 1]
    counts = rich["dominant_topic"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(counts.index, counts.values, color=ACCENT_COLOUR, edgecolor="white", height=0.6)

    for bar, val in zip(bars, counts.values):
        ax.text(val + 40, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left", fontsize=10)

    ax.set_title("LDA Dominant Topic Distribution — Returned Reviews", **FONT_TITLE)
    ax.set_xlabel("Number of Reviews", **FONT_AXIS)
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(0, counts.max() * 1.15)
    _save(fig, "topic_distribution.png")


# ---------------------------------------------------------------------------
# 5. Complaint category distribution
# ---------------------------------------------------------------------------

def plot_complaint_distribution(df: pd.DataFrame) -> None:
    """Bar chart of inferred complaint categories for returned reviews."""
    rich = df[df["is_returned"] == 1]
    counts = rich["complaint_category"].value_counts().sort_values(ascending=True)

    colours = plt.cm.RdYlBu(np.linspace(0.15, 0.85, len(counts)))

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(counts.index, counts.values, color=colours, edgecolor="white", height=0.65)

    for bar, val in zip(bars, counts.values):
        ax.text(val + 20, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left", fontsize=9)

    ax.set_title("Inferred Complaint Category Distribution — Returned Reviews", **FONT_TITLE)
    ax.set_xlabel("Number of Reviews", **FONT_AXIS)
    ax.set_xlim(0, counts.max() * 1.15)
    _save(fig, "complaint_category_distribution.png")


# ---------------------------------------------------------------------------
# 6. Top-30 TF-IDF keywords (RICH corpus)
# ---------------------------------------------------------------------------

def plot_keyword_frequency(df: pd.DataFrame, top_n: int = 30) -> None:
    """Horizontal bar chart of the most frequent TF-IDF keywords."""
    rich = df[df["is_returned"] == 1]

    # Flatten keyword lists (stored as Python lists after pipeline)
    all_keywords: list[str] = []
    for kw_entry in rich["extracted_keywords"]:
        if isinstance(kw_entry, list):
            all_keywords.extend(kw_entry)
        elif isinstance(kw_entry, str) and kw_entry.startswith("["):
            try:
                import ast
                all_keywords.extend(ast.literal_eval(kw_entry))
            except Exception:
                pass

    if not all_keywords:
        logger.warning("No keywords found — skipping keyword frequency plot.")
        return

    from collections import Counter
    freq = Counter(all_keywords).most_common(top_n)
    labels, values = zip(*freq)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=ACCENT_COLOUR, edgecolor="white", height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} TF-IDF Keywords — Returned Reviews", **FONT_TITLE)
    ax.set_xlabel("Frequency Across All Reviews", **FONT_AXIS)
    _save(fig, "keyword_frequency.png")


# ---------------------------------------------------------------------------
# 7. Word cloud — RICH corpus
# ---------------------------------------------------------------------------

def plot_wordcloud(df: pd.DataFrame) -> None:
    """Word cloud generated from RICH review text."""
    rich_text = " ".join(df[df["is_returned"] == 1]["review_text"].astype(str).tolist())

    wc = WordCloud(
        width=1400,
        height=700,
        background_color="white",
        colormap="RdYlBu",
        max_words=200,
        collocations=True,
        min_font_size=8,
    ).generate(rich_text)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud — Returned Customer Reviews", **FONT_TITLE)
    _save(fig, "wordcloud_rich.png")


# ---------------------------------------------------------------------------
# 8. Sentiment score vs review rating
# ---------------------------------------------------------------------------

def plot_sentiment_vs_rating(df: pd.DataFrame) -> None:
    """Box-plot: VADER sentiment score by review star rating."""
    plot_df = df.copy()
    plot_df["sentiment_score"] = plot_df["sentiment_score"].astype(float)
    plot_df["review_rating"] = plot_df["review_rating"].astype(int)

    palette = ["#D94F3D", "#E8A838", "#6BAB8B", "#4A90D9", "#2C5F8A"]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=plot_df, x="review_rating", y="sentiment_score",
        hue="review_rating", palette=palette,
        width=0.55, linewidth=1.4, legend=False, ax=ax,
    )
    ax.set_title("VADER Sentiment Score by Review Star Rating", **FONT_TITLE)
    ax.set_xlabel("Review Star Rating", **FONT_AXIS)
    ax.set_ylabel("VADER Compound Score", **FONT_AXIS)
    _save(fig, "sentiment_vs_rating.png")


# ---------------------------------------------------------------------------
# 9. Topic × root_cause heatmap
# ---------------------------------------------------------------------------

def plot_topic_by_root_cause(df: pd.DataFrame) -> None:
    """Heatmap: LDA dominant topic vs root_cause_category (RICH rows)."""
    rich = df[df["is_returned"] == 1].copy()

    if rich["dominant_topic"].isna().all() or rich["root_cause_category"].isna().all():
        logger.warning("Insufficient data for topic × root_cause heatmap — skipping.")
        return

    pivot = (
        rich.groupby(["dominant_topic", "root_cause_category"])
        .size()
        .unstack(fill_value=0)
    )

    # Normalise by row (topic) to show proportions
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        pivot_norm,
        annot=pivot.values,      # raw counts as annotations
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Proportion within topic"},
    )
    ax.set_title("LDA Topic vs Root Cause Category — Returned Reviews", **FONT_TITLE)
    ax.set_xlabel("Root Cause Category", **FONT_AXIS)
    ax.set_ylabel("Dominant LDA Topic", **FONT_AXIS)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    _save(fig, "topic_by_root_cause.png")


# ---------------------------------------------------------------------------
# Master plot function
# ---------------------------------------------------------------------------

def generate_all_plots(df: pd.DataFrame) -> None:
    """Generate and save all Phase 2 visualisations."""
    logger.info("Generating Phase 2 visualisations …")
    plot_sentiment_distribution(df)
    plot_sentiment_boxplot(df)
    plot_subjectivity_distribution(df)
    plot_topic_distribution(df)
    plot_complaint_distribution(df)
    plot_keyword_frequency(df)
    plot_wordcloud(df)
    plot_sentiment_vs_rating(df)
    plot_topic_by_root_cause(df)
    logger.info("All plots saved to %s", PLOTS_DIR)
