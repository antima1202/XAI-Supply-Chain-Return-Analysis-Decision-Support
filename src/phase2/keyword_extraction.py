"""
keyword_extraction.py  –  TF-IDF keyword extraction for the Phase 2 pipeline.

Design
------
  • Fit a TF-IDF vectorizer on the RICH (returned) review corpus only.
    Non-returned reviews do not carry complaint signals, so including them
    would dilute term weights.
  • For each RICH review, extract the top-N highest-scoring terms.
  • The fitted vectorizer is persisted to disk so the dashboard (Phase 5)
    can transform new reviews without re-training.
  • LEAN rows receive an empty list for extracted_keywords.

TF-IDF configuration
--------------------
  - min_df = 3     : ignore terms appearing in fewer than 3 documents
  - max_df = 0.85  : ignore terms appearing in more than 85 % of documents
                     (quasi stop-words in this corpus)
  - ngram_range = (1, 2) : unigrams and bigrams capture phrases like
                           "wrong size", "damaged packaging", "late delivery"
  - sublinear_tf = True  : apply log(1 + tf) to dampen high raw frequencies
  - max_features = 5000  : vocabulary ceiling for memory efficiency
"""

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import (
    TFIDF_PATH,
    get_logger,
    save_artefact,
)

logger = get_logger("phase2.keywords")

# Number of top keywords to extract per review
TOP_K = 8


# ---------------------------------------------------------------------------
# Vectorizer construction
# ---------------------------------------------------------------------------

def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Return a freshly configured (unfitted) TF-IDF vectorizer."""
    return TfidfVectorizer(
        min_df=3,
        max_df=0.85,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=5000,
        token_pattern=r"(?u)\b[a-z][a-z]{2,}\b",  # min 3-char alphabetic tokens
    )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_tfidf(cleaned_rich_texts: list[str]) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the cleaned RICH review corpus.

    Parameters
    ----------
    cleaned_rich_texts : list[str]
        Preprocessed (lowercased, punctuation-stripped) review strings
        from returned orders only.

    Returns
    -------
    TfidfVectorizer
        The fitted vectorizer.
    """
    vectorizer = build_tfidf_vectorizer()
    vectorizer.fit(cleaned_rich_texts)
    vocab_size = len(vectorizer.vocabulary_)
    logger.info("TF-IDF fitted on %d docs  (vocab size: %d)", len(cleaned_rich_texts), vocab_size)
    save_artefact(vectorizer, TFIDF_PATH)
    return vectorizer


# ---------------------------------------------------------------------------
# Extract keywords for a single document
# ---------------------------------------------------------------------------

def extract_keywords_single(text: str, vectorizer: TfidfVectorizer, top_k: int = TOP_K) -> list[str]:
    """
    Return the top-*k* TF-IDF keywords for a single cleaned text string.

    Returns an empty list for blank / very short texts.
    """
    if not isinstance(text, str) or len(text.split()) < 3:
        return []

    tfidf_matrix = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    # Convert the single-row sparse matrix to a dense 1-D array
    row_dense = tfidf_matrix[0].toarray().flatten()   # shape: (vocab_size,)
    nonzero_indices = np.where(row_dense > 0)[0]

    if len(nonzero_indices) == 0:
        return []

    scores = row_dense[nonzero_indices]
    top_indices = nonzero_indices[np.argsort(scores)[::-1][:top_k]]
    keywords = [feature_names[i] for i in top_indices]
    return keywords


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_keywords_batch(
    cleaned_texts: list[str],
    vectorizer: TfidfVectorizer,
    top_k: int = TOP_K,
) -> list[list[str]]:
    """
    Extract top-*k* keywords for every document in *cleaned_texts*.

    Parameters
    ----------
    cleaned_texts : list[str]
        Cleaned review strings (RICH rows only; function is only called
        for returned orders).
    vectorizer : TfidfVectorizer
        A **fitted** TF-IDF vectorizer.
    top_k : int
        Number of keywords to extract per document.

    Returns
    -------
    list[list[str]]
        Keyword lists aligned with *cleaned_texts*.
    """
    if not cleaned_texts:
        return []

    tfidf_matrix = vectorizer.transform(cleaned_texts)
    feature_names = vectorizer.get_feature_names_out()

    # Convert full matrix to dense once — more efficient than per-row sparse ops
    dense_matrix = tfidf_matrix.toarray()   # shape: (n_docs, vocab_size)
    results = []

    for i in range(dense_matrix.shape[0]):
        row_dense = dense_matrix[i]
        nonzero_indices = np.where(row_dense > 0)[0]
        if len(nonzero_indices) == 0:
            results.append([])
            continue
        scores = row_dense[nonzero_indices]
        top_idx = nonzero_indices[np.argsort(scores)[::-1][:top_k]]
        results.append([feature_names[j] for j in top_idx])

    logger.info(
        "Keywords extracted for %d docs  (avg keywords/doc: %.1f)",
        len(results),
        sum(len(k) for k in results) / max(len(results), 1),
    )
    return results
