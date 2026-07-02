"""
pipeline.py  –  Phase 2 NLP pipeline orchestrator.

Execution order
---------------
  1.  Load the Phase 1 dataset (fashion_returns_dataset.csv)
  2.  Preprocess review_text  →  cleaned_text + token_lists
  3.  Sentiment analysis
        • VADER compound score      → all 50 000 rows
        • TextBlob subjectivity     → RICH rows only (14 000)
  4.  TF-IDF keyword extraction     → RICH rows only
  5.  LDA topic modelling           → RICH rows only
  6.  Complaint classification      → RICH rows only
  7.  Populate NLP columns in-place (replace every NLP_PENDING / null)
  8.  Save enriched dataset         → data/processed/fashion_returns_dataset_nlp.csv
  9.  Generate dissertation plots   → outputs/phase2/plots/

LEAN rows (is_returned == 0) receive:
  • sentiment_score   : VADER compound (genuine value — LEAN reviews are real text)
  • subjectivity_score: None
  • dominant_topic    : None
  • extracted_keywords: None  (stored as pd.NA)
  • complaint_category: None

This is consistent with the data dictionary specification and the
dissertation methodology: subjectivity, topic, keywords, and complaint
classification are complaint-analysis features with no meaning for
non-returned orders.
"""

import ast

import numpy as np
import pandas as pd
from tqdm import tqdm

from .complaint_classifier import classify_complaints_batch
from .keyword_extraction   import extract_keywords_batch, fit_tfidf
from .preprocessing        import preprocess_series
from .sentiment_analysis   import compute_vader_batch, compute_subjectivity_batch
from .topic_modelling      import (
    assign_dominant_topics,
    build_corpus,
    compute_coherence,
    train_lda,
)
from .utils import (
    INPUT_CSV,
    OUTPUT_CSV,
    get_logger,
    timer,
)
from .visualisations import generate_all_plots

logger = get_logger("phase2.pipeline")


# ---------------------------------------------------------------------------
# Step 1: Load dataset
# ---------------------------------------------------------------------------

def load_dataset(path=INPUT_CSV) -> pd.DataFrame:
    """Load the Phase 1 CSV with consistent dtype handling."""
    logger.info("Loading dataset from %s …", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded  %d rows × %d columns", *df.shape)

    # Validate expected columns exist
    required = {
        "is_returned", "review_text", "sentiment_score",
        "subjectivity_score", "complaint_category",
        "dominant_topic", "extracted_keywords", "root_cause_category",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    logger.info(
        "RICH rows (returned): %d  |  LEAN rows: %d",
        (df["is_returned"] == 1).sum(),
        (df["is_returned"] == 0).sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 2: Preprocessing
# ---------------------------------------------------------------------------

def run_preprocessing(df: pd.DataFrame) -> tuple[list[str], list[list[str]],
                                                   list[str], list[list[str]]]:
    """
    Preprocess review_text for all rows and for RICH rows separately.

    Returns
    -------
    all_cleaned     : list[str]          — for VADER (all 50 k rows)
    all_tokens      : list[list[str]]    — for LDA later (not used for LEAN)
    rich_cleaned    : list[str]          — cleaned text for RICH rows
    rich_tokens     : list[list[str]]    — token lists for RICH rows
    """
    with timer("Preprocessing — all rows"):
        all_cleaned, all_tokens = preprocess_series(df["review_text"].tolist())

    rich_mask   = df["is_returned"] == 1
    rich_texts  = df.loc[rich_mask, "review_text"].tolist()

    with timer("Preprocessing — RICH rows"):
        rich_cleaned, rich_tokens = preprocess_series(rich_texts)

    return all_cleaned, all_tokens, rich_cleaned, rich_tokens


# ---------------------------------------------------------------------------
# Step 3: Sentiment analysis
# ---------------------------------------------------------------------------

def run_sentiment(df: pd.DataFrame, all_cleaned: list[str]) -> pd.DataFrame:
    """Compute VADER (all rows) and TextBlob subjectivity (RICH only)."""

    with timer("VADER sentiment — all rows"):
        vader_scores = compute_vader_batch(all_cleaned)
    df["sentiment_score"] = [float(s) for s in vader_scores]

    rich_mask = df["is_returned"] == 1
    rich_cleaned_for_subj = [all_cleaned[i] for i in df.index[rich_mask]]

    with timer("TextBlob subjectivity — RICH rows"):
        subj_scores = compute_subjectivity_batch(rich_cleaned_for_subj)

    df["subjectivity_score"] = np.nan
    df.loc[rich_mask, "subjectivity_score"] = [float(s) for s in subj_scores]

    logger.info(
        "VADER   | mean=%.3f  std=%.3f",
        df["sentiment_score"].mean(),
        df["sentiment_score"].std(),
    )
    logger.info(
        "Subjectivity (RICH) | mean=%.3f  std=%.3f",
        df.loc[rich_mask, "subjectivity_score"].mean(),
        df.loc[rich_mask, "subjectivity_score"].std(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 4: TF-IDF keyword extraction
# ---------------------------------------------------------------------------

def run_keyword_extraction(
    df: pd.DataFrame,
    rich_cleaned: list[str],
) -> tuple[pd.DataFrame, object]:
    """Fit TF-IDF on RICH reviews and extract keywords per row."""

    with timer("TF-IDF fit — RICH corpus"):
        vectorizer = fit_tfidf(rich_cleaned)

    with timer("TF-IDF keyword extraction — RICH rows"):
        keyword_lists = extract_keywords_batch(rich_cleaned, vectorizer)

    # Store as Python list objects in the DataFrame (serialised as string in CSV)
    # Use object-dtype Series to avoid pandas rejecting list values in float columns
    rich_mask = df["is_returned"] == 1
    kw_series = pd.Series([None] * len(df), dtype=object)
    rich_indices = df.index[rich_mask].tolist()

    for idx, keywords in zip(rich_indices, keyword_lists):
        kw_series[idx] = keywords   # keep as list; _finalise_columns converts to str

    df["extracted_keywords"] = kw_series

    return df, vectorizer


# ---------------------------------------------------------------------------
# Step 5: LDA topic modelling
# ---------------------------------------------------------------------------

def run_topic_modelling(
    df: pd.DataFrame,
    rich_tokens: list[list[str]],
) -> pd.DataFrame:
    """Train LDA on RICH token lists and assign dominant topics."""

    with timer("Building Gensim corpus"):
        dictionary, corpus = build_corpus(rich_tokens)

    with timer("Training LDA model"):
        lda_model = train_lda(corpus, dictionary)

    # Compute coherence score for dissertation reporting
    try:
        with timer("LDA coherence score"):
            compute_coherence(lda_model, rich_tokens, dictionary)
    except Exception as exc:
        logger.warning("Coherence computation skipped: %s", exc)

    with timer("Assigning dominant topics — RICH rows"):
        topic_labels = assign_dominant_topics(lda_model, dictionary, rich_tokens)

    topic_series = pd.Series([None] * len(df), dtype=object, index=df.index)
    rich_mask   = df["is_returned"] == 1
    rich_indices = df.index[rich_mask].tolist()

    for idx, label in zip(rich_indices, topic_labels):
        topic_series[idx] = label

    df["dominant_topic"] = topic_series

    return df


# ---------------------------------------------------------------------------
# Step 6: Complaint classification
# ---------------------------------------------------------------------------

def run_complaint_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Classify complaint categories for RICH rows using NLP signals."""

    rich_mask    = df["is_returned"] == 1
    rich_indices = df.index[rich_mask].tolist()

    review_texts      = df.loc[rich_mask, "review_text"].tolist()
    dominant_topics   = df.loc[rich_mask, "dominant_topic"].tolist()

    # Retrieve keyword lists stored as Python lists in DataFrame
    keyword_lists = []
    for idx in rich_indices:
        kw = df.at[idx, "extracted_keywords"]
        if isinstance(kw, list):
            keyword_lists.append(kw)
        elif isinstance(kw, str):
            try:
                keyword_lists.append(ast.literal_eval(kw))
            except Exception:
                keyword_lists.append([])
        else:
            keyword_lists.append([])

    with timer("Complaint classification — RICH rows"):
        categories = classify_complaints_batch(
            review_texts=review_texts,
            extracted_keywords_list=keyword_lists,
            dominant_topics=[str(t) for t in dominant_topics],
        )

    cat_series = pd.Series([None] * len(df), dtype=object)
    for idx, cat in zip(rich_indices, categories):
        cat_series[idx] = cat
    df["complaint_category"] = cat_series

    return df


# ---------------------------------------------------------------------------
# Step 7: Final column type cleanup
# ---------------------------------------------------------------------------

def _finalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure numeric NLP columns have correct dtypes and LEAN rows have
    explicit None / NaN rather than the original 'NLP_PENDING' string.
    """
    # sentiment_score: float for all rows
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")

    # subjectivity_score: float for RICH, NaN for LEAN
    df["subjectivity_score"] = pd.to_numeric(df["subjectivity_score"], errors="coerce")

    # extracted_keywords: store as string representation of list for CSV compatibility
    def _kw_to_str(val):
        if isinstance(val, list):
            return str(val)
        return val  # already None/NaN for LEAN

    df["extracted_keywords"] = df["extracted_keywords"].apply(_kw_to_str)

    # Confirm no NLP_PENDING values remain
    for col in ["sentiment_score", "subjectivity_score", "complaint_category", "dominant_topic"]:
        pending = (df[col].astype(str) == "NLP_PENDING").sum()
        if pending > 0:
            logger.warning("Column '%s' still has %d NLP_PENDING values!", col, pending)
        else:
            logger.info("Column '%s' — NLP_PENDING fully replaced ✓", col)

    return df


# ---------------------------------------------------------------------------
# Step 8: Save output CSV
# ---------------------------------------------------------------------------

def save_dataset(df: pd.DataFrame, path=OUTPUT_CSV) -> None:
    """Write the enriched dataset to disk without overwriting the original."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Enriched dataset saved → %s  (%d rows × %d cols)", path, *df.shape)


# ---------------------------------------------------------------------------
# Master pipeline entry-point
# ---------------------------------------------------------------------------

def run_phase2_pipeline() -> pd.DataFrame:
    """
    Execute the complete Phase 2 NLP pipeline end-to-end.

    Returns the enriched DataFrame (also saved to disk).
    """
    logger.info("=" * 60)
    logger.info("Phase 2 NLP Pipeline — START")
    logger.info("=" * 60)

    # 1. Load
    df = load_dataset()

    # 2. Preprocess
    all_cleaned, all_tokens, rich_cleaned, rich_tokens = run_preprocessing(df)

    # 3. Sentiment
    df = run_sentiment(df, all_cleaned)

    # 4. Keywords
    df, _vectorizer = run_keyword_extraction(df, rich_cleaned)

    # 5. Topics
    df = run_topic_modelling(df, rich_tokens)

    # 6. Complaint classification
    df = run_complaint_classification(df)

    # 7. Finalise
    df = _finalise_columns(df)

    # 8. Save CSV
    save_dataset(df)

    # 9. Visualisations
    with timer("Generating visualisations"):
        generate_all_plots(df)

    logger.info("=" * 60)
    logger.info("Phase 2 NLP Pipeline — COMPLETE")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    run_phase2_pipeline()
