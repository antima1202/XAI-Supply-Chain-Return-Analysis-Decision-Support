"""
analyser.py  –  Root Cause Analysis computations for Phase 5.

Produces six analytical outputs from the Phase 2 enriched dataset:

  1.  Root cause summary      — counts, percentages, avg sentiment per class
  2.  Monthly trend           — returns per root cause per month (2-year window)
  3.  Product breakdown       — root cause distribution per product subcategory
  4.  Brand breakdown         — return rate and root cause mix per brand
  5.  Region breakdown        — root cause distribution per region
  6.  Sentiment profile       — avg sentiment and subjectivity per root cause

All outputs are plain pandas DataFrames and dicts — no side effects.
Saving is handled by the pipeline.
"""

import pandas as pd
import numpy as np
from .utils import get_logger, ROOT_CAUSE_CLASSES

logger = get_logger("phase5.analyser")


# ---------------------------------------------------------------------------
# Step 1: Load and prepare RICH rows
# ---------------------------------------------------------------------------

def load_and_prepare(path) -> pd.DataFrame:
    """Load the NLP-enriched CSV and return RICH rows with parsed dates."""
    df = pd.read_csv(path, low_memory=False)
    rich = df[df["is_returned"] == 1].copy()

    # Parse dates
    rich["order_date"] = pd.to_datetime(rich["order_date"], errors="coerce")
    rich["year_month"] = rich["order_date"].dt.to_period("M")

    # Numeric coercions
    rich["sentiment_score"]    = pd.to_numeric(rich["sentiment_score"],    errors="coerce")
    rich["subjectivity_score"] = pd.to_numeric(rich["subjectivity_score"], errors="coerce")
    rich["return_lag_days"]    = pd.to_numeric(rich["return_lag_days"],    errors="coerce")
    rich["delivery_days"]      = pd.to_numeric(rich["delivery_days"],      errors="coerce")

    logger.info(
        "Loaded %d RICH rows  |  date range: %s → %s",
        len(rich),
        rich["order_date"].min().strftime("%Y-%m"),
        rich["order_date"].max().strftime("%Y-%m"),
    )
    return rich


# ---------------------------------------------------------------------------
# Step 2: Root cause summary
# ---------------------------------------------------------------------------

def compute_root_cause_summary(rich: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-class summary statistics.

    Returns a DataFrame with columns:
      root_cause_category, count, pct_of_returns,
      avg_sentiment, avg_subjectivity, avg_return_lag_days,
      avg_delivery_days
    """
    total = len(rich)
    groups = rich.groupby("root_cause_category")

    summary = groups.agg(
        count              = ("root_cause_category", "count"),
        avg_sentiment      = ("sentiment_score",    "mean"),
        avg_subjectivity   = ("subjectivity_score", "mean"),
        avg_return_lag     = ("return_lag_days",    "mean"),
        avg_delivery_days  = ("delivery_days",      "mean"),
    ).reset_index()

    summary["pct_of_returns"] = (summary["count"] / total * 100).round(2)
    summary = summary.sort_values("count", ascending=False).reset_index(drop=True)

    # Round numeric columns
    for col in ["avg_sentiment", "avg_subjectivity", "avg_return_lag", "avg_delivery_days"]:
        summary[col] = summary[col].round(4)

    logger.info("Root cause summary computed (%d classes)", len(summary))
    return summary


# ---------------------------------------------------------------------------
# Step 3: Monthly trend
# ---------------------------------------------------------------------------

def compute_monthly_trend(rich: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly return counts per root cause category.

    Returns a pivoted DataFrame:
      index = year_month (Period)
      columns = root cause categories
      values = return counts
    """
    monthly = (
        rich.groupby(["year_month", "root_cause_category"])
        .size()
        .reset_index(name="count")
    )

    pivot = monthly.pivot(
        index="year_month",
        columns="root_cause_category",
        values="count",
    ).fillna(0).astype(int)

    # Ensure all classes present
    for cls in ROOT_CAUSE_CLASSES:
        if cls not in pivot.columns:
            pivot[cls] = 0

    pivot = pivot.sort_index()
    logger.info(
        "Monthly trend computed  |  %d months  |  %d root causes",
        len(pivot), len(pivot.columns),
    )
    return pivot


# ---------------------------------------------------------------------------
# Step 4: Product subcategory breakdown
# ---------------------------------------------------------------------------

def compute_product_breakdown(rich: pd.DataFrame) -> pd.DataFrame:
    """
    Return count and percentage of each root cause per product subcategory.

    Returns a DataFrame with MultiIndex (product_subcategory, root_cause_category)
    and columns: count, pct_within_subcategory.
    """
    product = (
        rich.groupby(["product_subcategory", "root_cause_category"])
        .size()
        .reset_index(name="count")
    )

    # Percentage within each subcategory
    totals = product.groupby("product_subcategory")["count"].transform("sum")
    product["pct_within_subcategory"] = (product["count"] / totals * 100).round(2)
    product = product.sort_values(["product_subcategory", "count"], ascending=[True, False])

    logger.info("Product breakdown computed  (%d subcategories)", rich["product_subcategory"].nunique())
    return product


# ---------------------------------------------------------------------------
# Step 5: Brand breakdown
# ---------------------------------------------------------------------------

def compute_brand_breakdown(rich: pd.DataFrame) -> pd.DataFrame:
    """
    Return total returns, return rate, and dominant root cause per brand.

    Return rate is computed against all orders (RICH + LEAN) for that brand,
    so we need the full dataset. Since only RICH is passed here, we compute
    relative shares — the pipeline passes the full df for rate calculation.
    """
    brand = (
        rich.groupby(["brand", "root_cause_category"])
        .size()
        .reset_index(name="count")
    )

    totals = brand.groupby("brand")["count"].transform("sum")
    brand["pct_within_brand"] = (brand["count"] / totals * 100).round(2)

    # Dominant root cause per brand
    dominant = (
        brand.loc[brand.groupby("brand")["count"].idxmax()]
        .rename(columns={"root_cause_category": "dominant_root_cause",
                          "count": "dominant_count"})[["brand", "dominant_root_cause"]]
    )

    brand_summary = (
        brand.groupby("brand")["count"].sum()
        .reset_index(name="total_returns")
        .merge(dominant, on="brand")
    )

    brand_summary = brand_summary.sort_values("total_returns", ascending=False)
    logger.info("Brand breakdown computed (%d brands)", rich["brand"].nunique())
    return brand_summary, brand


# ---------------------------------------------------------------------------
# Step 6: Region breakdown
# ---------------------------------------------------------------------------

def compute_region_breakdown(rich: pd.DataFrame) -> pd.DataFrame:
    """Root cause distribution per region."""
    region = (
        rich.groupby(["region", "root_cause_category"])
        .size()
        .reset_index(name="count")
    )
    totals = region.groupby("region")["count"].transform("sum")
    region["pct_within_region"] = (region["count"] / totals * 100).round(2)
    region = region.sort_values(["region", "count"], ascending=[True, False])

    logger.info("Region breakdown computed (%d regions)", rich["region"].nunique())
    return region


# ---------------------------------------------------------------------------
# Step 7: Sentiment profile per root cause
# ---------------------------------------------------------------------------

def compute_sentiment_profile(rich: pd.DataFrame) -> pd.DataFrame:
    """
    Average sentiment score and subjectivity score per root cause.
    Used to characterise how customers write about each type of failure.
    """
    profile = (
        rich.groupby("root_cause_category")
        .agg(
            avg_sentiment    = ("sentiment_score",    "mean"),
            std_sentiment    = ("sentiment_score",    "std"),
            avg_subjectivity = ("subjectivity_score", "mean"),
            count            = ("root_cause_category", "count"),
        )
        .reset_index()
    )
    for col in ["avg_sentiment", "std_sentiment", "avg_subjectivity"]:
        profile[col] = profile[col].round(4)
    profile = profile.sort_values("avg_sentiment")

    logger.info("Sentiment profile computed")
    return profile


# ---------------------------------------------------------------------------
# Master analysis function
# ---------------------------------------------------------------------------

def run_analysis(path) -> dict:
    """
    Run all Phase 5 analysis steps and return results as a dict.

    Returns
    -------
    dict with keys:
      rich             : the prepared RICH DataFrame
      summary          : root cause summary table
      monthly_trend    : pivoted monthly trend DataFrame
      product          : product subcategory breakdown
      brand_summary    : brand-level summary
      brand_detail     : brand × root cause detail
      region           : region breakdown
      sentiment        : sentiment profile per root cause
    """
    rich = load_and_prepare(path)

    return {
        "rich":          rich,
        "summary":       compute_root_cause_summary(rich),
        "monthly_trend": compute_monthly_trend(rich),
        "product":       compute_product_breakdown(rich),
        "brand_summary": compute_brand_breakdown(rich)[0],
        "brand_detail":  compute_brand_breakdown(rich)[1],
        "region":        compute_region_breakdown(rich),
        "sentiment":     compute_sentiment_profile(rich),
    }
