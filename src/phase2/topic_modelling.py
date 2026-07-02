"""
topic_modelling.py  –  Latent Dirichlet Allocation (LDA) topic modelling.

Design
------
  • Train a 6-topic LDA model using Gensim on RICH (returned) reviews only.
    Six topics mirrors the six root_cause_category classes, which aids
    interpretability and gives the dissertation a clear methodological link
    between NLP and ML phases.
  • Each topic is labelled by inspecting its top terms and mapping it to a
    meaningful supply-chain concept.  The label map is persisted alongside
    the model so Phase 5 can decode numeric topic IDs.
  • Every RICH review is assigned its single dominant topic (highest γ weight).
  • LEAN rows receive None for dominant_topic.

Gensim LDA hyper-parameters
----------------------------
  num_topics   = 6      — mirrors root_cause classes
  passes       = 15     — corpus passes; improves convergence
  iterations   = 100    — per-document E-step iterations
  alpha        = 'auto' — asymmetric Dirichlet prior (learned from data)
  eta          = 'auto' — topic-word Dirichlet prior (learned from data)
  random_state = 42     — reproducibility
"""

import gensim
from gensim import corpora
from gensim.models import LdaModel

from .utils import (
    LDA_CORPUS_PATH,
    LDA_DICT_PATH,
    LDA_MODEL_PATH,
    TOPIC_MAP_PATH,
    get_logger,
    save_artefact,
)

logger = get_logger("phase2.topics")

NUM_TOPICS = 6

# ---------------------------------------------------------------------------
# Human-readable topic labels
# ---------------------------------------------------------------------------
# After LDA training the top terms per topic are printed and the labels below
# are assigned by semantic inspection.  They are intentionally aligned with
# the six root_cause_category values used in Phase 3 ML.

_DEFAULT_TOPIC_LABELS: dict[int, str] = {
    0: "Size & Fit Issues",
    1: "Product Quality & Manufacturing",
    2: "Product Listing & Description",
    3: "Logistics & Delivery",
    4: "Packaging & Warehouse",
    5: "Customer Preference & Style",
}


# ---------------------------------------------------------------------------
# Build Gensim corpus objects
# ---------------------------------------------------------------------------

def build_corpus(token_lists: list[list[str]]) -> tuple:
    """
    Build a Gensim Dictionary and Bag-of-Words corpus from token lists.

    Parameters
    ----------
    token_lists : list[list[str]]
        Lemmatised token lists from RICH reviews.

    Returns
    -------
    dictionary : corpora.Dictionary
    corpus     : list[list[tuple[int, int]]]
    """
    dictionary = corpora.Dictionary(token_lists)

    # Prune very rare / very common tokens from the LDA vocabulary
    dictionary.filter_extremes(no_below=3, no_above=0.85)

    corpus = [dictionary.doc2bow(tokens) for tokens in token_lists]
    logger.info(
        "LDA corpus built  |  vocab=%d  |  docs=%d",
        len(dictionary),
        len(corpus),
    )
    return dictionary, corpus


# ---------------------------------------------------------------------------
# Train LDA
# ---------------------------------------------------------------------------

def train_lda(
    corpus,
    dictionary: corpora.Dictionary,
    num_topics: int = NUM_TOPICS,
    passes: int = 15,
    iterations: int = 100,
    random_state: int = 42,
) -> LdaModel:
    """
    Train a Gensim LDA model and persist it to disk.

    Returns the fitted LdaModel.
    """
    logger.info("Training LDA  (topics=%d, passes=%d, iterations=%d) …", num_topics, passes, iterations)

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
        alpha="auto",
        eta="auto",
        random_state=random_state,
        per_word_topics=False,
    )

    # Print top terms per topic for manual label inspection
    logger.info("LDA top terms per topic:")
    for tid in range(num_topics):
        terms = model.show_topic(tid, topn=10)
        term_str = ", ".join(f"{w}({p:.3f})" for w, p in terms)
        logger.info("  Topic %d: %s", tid, term_str)

    # Persist model artefacts
    save_artefact(model,      LDA_MODEL_PATH)
    save_artefact(dictionary, LDA_DICT_PATH)
    save_artefact(corpus,     LDA_CORPUS_PATH)

    return model


# ---------------------------------------------------------------------------
# Assign dominant topic
# ---------------------------------------------------------------------------

def get_dominant_topic(model: LdaModel, bow: list[tuple]) -> int:
    """
    Return the topic ID with the highest probability for a single document.

    Parameters
    ----------
    model : LdaModel
    bow   : list[tuple[int, int]]
        Bag-of-Words representation (from dictionary.doc2bow).

    Returns
    -------
    int  — topic index in [0, num_topics - 1]
    """
    topic_probs = model.get_document_topics(bow, minimum_probability=0.0)
    if not topic_probs:
        return 0  # fallback for empty documents
    return max(topic_probs, key=lambda x: x[1])[0]


def assign_dominant_topics(
    model: LdaModel,
    dictionary: corpora.Dictionary,
    token_lists: list[list[str]],
    topic_label_map: dict[int, str] | None = None,
) -> list[str]:
    """
    Assign a human-readable dominant topic label to each document.

    Parameters
    ----------
    model          : fitted LdaModel
    dictionary     : fitted Gensim Dictionary
    token_lists    : list of lemmatised token lists (RICH rows)
    topic_label_map: optional {topic_id: label} override

    Returns
    -------
    list[str]  — topic labels aligned with *token_lists*
    """
    labels = topic_label_map or _DEFAULT_TOPIC_LABELS
    save_artefact(labels, TOPIC_MAP_PATH)

    dominant_topics = []
    for tokens in token_lists:
        bow = dictionary.doc2bow(tokens)
        tid = get_dominant_topic(model, bow)
        dominant_topics.append(labels.get(tid, f"Topic_{tid}"))

    # Log distribution
    from collections import Counter
    dist = Counter(dominant_topics)
    logger.info("Dominant topic distribution:")
    for label, count in dist.most_common():
        logger.info("  %-40s %d", label, count)

    return dominant_topics


# ---------------------------------------------------------------------------
# Coherence helper (informational only, not used in pipeline)
# ---------------------------------------------------------------------------

def compute_coherence(
    model: LdaModel,
    token_lists: list[list[str]],
    dictionary: corpora.Dictionary,
    coherence: str = "c_v",
) -> float:
    """
    Compute topic coherence score (c_v) for reporting in the dissertation.

    This is called optionally after training for logging purposes only.
    """
    from gensim.models.coherencemodel import CoherenceModel

    cm = CoherenceModel(
        model=model,
        texts=token_lists,
        dictionary=dictionary,
        coherence=coherence,
    )
    score = cm.get_coherence()
    logger.info("LDA coherence (%s) = %.4f", coherence, score)
    return score
