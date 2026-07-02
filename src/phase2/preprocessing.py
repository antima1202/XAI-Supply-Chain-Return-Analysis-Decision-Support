"""
preprocessing.py  –  Text cleaning and normalisation for the Phase 2 NLP pipeline.

Responsibilities
----------------
  1. Strip HTML tags
  2. Lowercase
  3. Remove punctuation / special characters
  4. Tokenise
  5. Remove English stop-words
  6. Lemmatise (WordNet)

The cleaned text is stored as:
  - A single string (``cleaned_text``) used by VADER / TextBlob / TF-IDF
  - A list of tokens (``tokens``) used by the Gensim LDA pipeline

All functions operate on plain strings; the pipeline module handles DataFrame
iteration so each step here stays stateless and unit-testable.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .utils import get_logger

# ---------------------------------------------------------------------------
# Ensure required NLTK data is present
# ---------------------------------------------------------------------------

_REQUIRED_NLTK = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords",    "stopwords"),
    ("corpora/wordnet",      "wordnet"),
    ("corpora/omw-1.4",      "omw-1.4"),
]

for _resource_path, _resource_id in _REQUIRED_NLTK:
    try:
        nltk.data.find(_resource_path)
    except LookupError:
        nltk.download(_resource_id, quiet=True)

# ---------------------------------------------------------------------------
# Module-level singletons (created once, reused per call)
# ---------------------------------------------------------------------------

_LEMMATIZER  = WordNetLemmatizer()
_STOP_WORDS  = set(stopwords.words("english"))

# Fashion-domain stop-words that add noise without meaning
_DOMAIN_NOISE = {
    "item", "product", "order", "purchase", "bought", "buy", "get",
    "got", "would", "one", "also", "even", "much", "really", "quite",
    "just", "like", "bit", "thing", "lot", "use", "used",
}
_STOP_WORDS.update(_DOMAIN_NOISE)

logger = get_logger("phase2.preprocessing")

# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def _remove_html(text: str) -> str:
    """Strip HTML / XML tags."""
    return re.sub(r"<[^>]+>", " ", text)


def _remove_urls(text: str) -> str:
    """Remove HTTP(S) URLs."""
    return re.sub(r"https?://\S+", " ", text)


def _remove_punctuation(text: str) -> str:
    """Replace punctuation and special characters with a space."""
    # Keep apostrophes inside words (e.g. "don't") temporarily,
    # then strip everything else.
    text = re.sub(r"[^\w\s']", " ", text)   # keep word-chars, whitespace, apostrophe
    text = re.sub(r"'", " ", text)           # then remove apostrophes
    return text


def _normalise_whitespace(text: str) -> str:
    """Collapse multiple spaces / newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Return a cleaned, lower-cased string suitable for VADER / TextBlob / TF-IDF.

    Note: VADER works better on lightly-cleaned text (it uses capitalisation
    and punctuation as sentiment signals), so this function produces a
    *moderately* cleaned string rather than aggressively stripping everything.
    For VADER we therefore call this but NOT the tokenise/lemmatise step.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _remove_html(text)
    text = _remove_urls(text)
    text = text.lower()
    text = _remove_punctuation(text)
    text = _normalise_whitespace(text)
    return text


def tokenise_and_lemmatise(cleaned_text: str) -> list[str]:
    """
    Tokenise *cleaned_text*, remove stop-words, and lemmatise.

    Returns a list of meaningful tokens used by TF-IDF and LDA.
    Minimum token length is 3 characters to filter out noise.
    """
    if not cleaned_text:
        return []

    tokens = word_tokenize(cleaned_text)

    # Remove stop-words, short tokens, and purely numeric tokens
    tokens = [
        t for t in tokens
        if t not in _STOP_WORDS
        and len(t) >= 3
        and not t.isnumeric()
    ]

    # Lemmatise using noun POS (fast and sufficient for this domain)
    tokens = [_lemmatizer(t) for t in tokens]

    return tokens


def _lemmatizer(token: str) -> str:
    """Lemmatise a single token as a noun then as a verb."""
    lemma = _LEMMATIZER.lemmatize(token, pos="n")
    if lemma == token:
        lemma = _LEMMATIZER.lemmatize(token, pos="v")
    return lemma


def preprocess_series(texts) -> tuple[list[str], list[list[str]]]:
    """
    Batch-preprocess a pandas Series or iterable of raw review strings.

    Returns
    -------
    cleaned_texts : list[str]
        Moderately-cleaned strings (for VADER / TextBlob / TF-IDF).
    token_lists : list[list[str]]
        Lemmatised token lists (for LDA).
    """
    cleaned_texts = []
    token_lists   = []

    for text in texts:
        cleaned = clean_text(str(text))
        tokens  = tokenise_and_lemmatise(cleaned)
        cleaned_texts.append(cleaned)
        token_lists.append(tokens)

    logger.info(
        "Preprocessed %d texts  (avg tokens: %.1f)",
        len(cleaned_texts),
        sum(len(t) for t in token_lists) / max(len(token_lists), 1),
    )
    return cleaned_texts, token_lists
