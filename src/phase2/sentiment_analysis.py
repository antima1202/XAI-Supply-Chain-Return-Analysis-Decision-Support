"""
sentiment_analysis.py  –  Sentiment and subjectivity scoring for the Phase 2 pipeline.

VADER
-----
  • Applied to ALL rows (RICH and LEAN) using the lightly-cleaned review text.
  • Returns the compound score in [-1.0, +1.0].
  • VADER is rule-based and needs minimal preprocessing; it exploits
    capitalisation and punctuation, so we pass the cleaned-but-not-lemmatised
    text rather than the token list.

TextBlob Subjectivity
---------------------
  • Applied to RICH rows only (is_returned == 1).
  • Returns a score in [0.0, 1.0]:  0 = objective, 1 = highly subjective.
  • Set to None for LEAN rows — consistent with the data dictionary spec.
"""

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .utils import get_logger

logger = get_logger("phase2.sentiment")

# Module-level singleton — SentimentIntensityAnalyzer is expensive to initialise
_VADER = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# VADER
# ---------------------------------------------------------------------------

def compute_vader_score(text: str) -> float:
    """
    Compute the VADER compound sentiment score for a single review string.

    Parameters
    ----------
    text : str
        Lightly-cleaned (lowercased, punctuation-stripped) review text.

    Returns
    -------
    float
        Compound score in [-1.0, +1.0], rounded to 4 decimal places.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    scores = _VADER.polarity_scores(text)
    return round(scores["compound"], 4)


def compute_vader_batch(texts) -> list[float]:
    """
    Compute VADER compound scores for an iterable of review strings.

    Returns a list of floats aligned with the input iterable.
    """
    scores = [compute_vader_score(str(t)) for t in texts]
    logger.info(
        "VADER scored %d texts  (mean=%.3f, min=%.3f, max=%.3f)",
        len(scores),
        sum(scores) / max(len(scores), 1),
        min(scores),
        max(scores),
    )
    return scores


# ---------------------------------------------------------------------------
# TextBlob subjectivity
# ---------------------------------------------------------------------------

def compute_subjectivity_score(text: str) -> float:
    """
    Compute TextBlob subjectivity for a single review string.

    Parameters
    ----------
    text : str
        Lightly-cleaned review text.

    Returns
    -------
    float
        Subjectivity in [0.0, 1.0], rounded to 4 decimal places.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    blob = TextBlob(text)
    return round(blob.sentiment.subjectivity, 4)


def compute_subjectivity_batch(texts) -> list[float]:
    """
    Compute TextBlob subjectivity for an iterable of review strings.

    Returns a list of floats aligned with the input iterable.
    """
    scores = [compute_subjectivity_score(str(t)) for t in texts]
    logger.info(
        "TextBlob subjectivity scored %d texts  (mean=%.3f)",
        len(scores),
        sum(scores) / max(len(scores), 1),
    )
    return scores
