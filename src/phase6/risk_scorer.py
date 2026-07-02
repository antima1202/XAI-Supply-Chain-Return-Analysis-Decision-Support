"""
risk_scorer.py  –  Quantified Risk Assessment for Phase 6.

Risk Score Formula
------------------
Each root cause receives a composite risk score in [0, 100]:

    Risk Score = (frequency_score × 0.40)
               + (impact_score   × 0.40)
               + (trend_score    × 0.20)

Component definitions
---------------------

frequency_score (0–100):
    Min-max normalised return count across the 6 root causes.
    Root cause with the most returns → 100. Fewest → 0.
    Rationale: higher volume = greater operational exposure.

impact_score (0–100):
    Domain-defined operational impact weight (see utils.OPERATIONAL_IMPACT).
    Reflects the cost, complexity, and strategic risk of each failure type.
    These weights are fixed domain knowledge — not derived from the data.

trend_score (0–100):
    Derived from the second-half vs first-half monthly trend change.
    A root cause growing over time scores higher; one declining scores lower.
    Formula:
        raw = pct_change_second_half_vs_first_half
        trend_score = 50 + clip(raw × 2, -50, 50)
    Interpretation:
        +25% growth  → trend_score = 100
        Flat (0%)    → trend_score = 50
        -25% decline → trend_score = 0

Risk level thresholds:
    Critical: 75–100
    High:     50–74
    Medium:   25–49
    Low:       0–24

Priority ranking:
    1 = highest risk, 6 = lowest risk (sorted by risk score descending).
"""

import numpy as np
import pandas as pd

from .utils import (
    ROOT_CAUSE_CLASSES,
    OPERATIONAL_IMPACT,
    RISK_LEVELS,
    get_logger,
)

logger = get_logger("phase6.scorer")


# ---------------------------------------------------------------------------
# Component 1: Frequency score
# ---------------------------------------------------------------------------

def compute_frequency_score(summary: pd.DataFrame) -> pd.Series:
    """
    Min-max normalise return counts to [0, 100].
    Returns a Series indexed by root_cause_category.
    """
    counts = summary.set_index("root_cause_category")["count"]
    min_c, max_c = counts.min(), counts.max()
    score = ((counts - min_c) / (max_c - min_c) * 100).round(2)
    return score


# ---------------------------------------------------------------------------
# Component 2: Impact score
# ---------------------------------------------------------------------------

def compute_impact_score() -> pd.Series:
    """
    Return the domain-defined operational impact weights as a Series.
    Already on [0, 100] scale — no normalisation needed.
    """
    return pd.Series(OPERATIONAL_IMPACT, name="impact_score")


# ---------------------------------------------------------------------------
# Component 3: Trend score
# ---------------------------------------------------------------------------

def compute_trend_score(trend: pd.DataFrame) -> pd.Series:
    """
    Score each root cause's trend direction from monthly data.

    Uses the full trend window, excluding the last partial month (Sep 2025)
    and the first 3 months (dataset warm-up period).

    Returns a Series indexed by root_cause_category, values in [0, 100].
    """
    # Exclude warm-up and partial month
    core = trend.iloc[3:-1]
    half = len(core) // 2
    first_half  = core.iloc[:half].mean()
    second_half = core.iloc[half:].mean()

    # Percentage change: positive = growing (worse), negative = declining (better)
    pct_change = ((second_half - first_half) / first_half.replace(0, np.nan) * 100).fillna(0)

    # Map to [0, 100]: flat = 50, +25% = 100, -25% = 0
    trend_score = (50 + np.clip(pct_change * 2, -50, 50)).round(2)

    logger.info("Trend scores (% change → score):")
    for cls in ROOT_CAUSE_CLASSES:
        if cls in pct_change.index:
            logger.info(
                "  %-40s  pct_change=%+.1f%%  trend_score=%.1f",
                cls, pct_change[cls], trend_score[cls],
            )

    return trend_score


# ---------------------------------------------------------------------------
# Composite risk score
# ---------------------------------------------------------------------------

def compute_risk_scores(summary: pd.DataFrame, trend: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the composite risk score for each root cause.

    Parameters
    ----------
    summary : DataFrame from Phase 5 root_cause_summary.joblib
    trend   : DataFrame from Phase 5 monthly_trend.joblib

    Returns
    -------
    DataFrame with columns:
      root_cause_category, count, pct_of_returns,
      frequency_score, impact_score, trend_score,
      risk_score, risk_level, priority,
      avg_sentiment, avg_return_lag
    """
    freq_score   = compute_frequency_score(summary)
    impact_score = compute_impact_score()
    trend_score  = compute_trend_score(trend)

    rows = []
    for cls in ROOT_CAUSE_CLASSES:
        row = summary[summary["root_cause_category"] == cls].iloc[0]

        fs = freq_score.get(cls, 0)
        is_ = impact_score.get(cls, 50)
        ts  = trend_score.get(cls, 50)

        composite = round(fs * 0.40 + is_ * 0.40 + ts * 0.20, 2)

        # Determine risk level
        level = "Low"
        for lvl, (lo, hi) in RISK_LEVELS.items():
            if lo <= composite <= hi:
                level = lvl
                break

        rows.append({
            "root_cause_category": cls,
            "count":               int(row["count"]),
            "pct_of_returns":      float(row["pct_of_returns"]),
            "frequency_score":     float(fs),
            "impact_score":        float(is_),
            "trend_score":         float(ts),
            "risk_score":          composite,
            "risk_level":          level,
            "avg_sentiment":       float(row["avg_sentiment"]),
            "avg_return_lag":      float(row["avg_return_lag"]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    df["priority"] = df.index + 1          # P1 = highest risk

    logger.info("\n--- Risk Score Results ---")
    for _, r in df.iterrows():
        logger.info(
            "  P%d  %-40s  score=%.1f  level=%s",
            r["priority"], r["root_cause_category"], r["risk_score"], r["risk_level"],
        )

    return df
