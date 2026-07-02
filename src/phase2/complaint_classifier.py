"""
complaint_classifier.py  –  NLP-driven complaint category classification.

Design rationale
----------------
A pure rule-based classifier (hard-coded per return_reason) would not
demonstrate NLP capability — it would simply copy a field already in the
dataset.  Instead this module implements a **hybrid signal classifier** that:

  1. Inspects extracted_keywords for domain-specific complaint signals
     (highest-priority signal — keyword evidence from the review itself).
  2. Uses the LDA dominant_topic as a secondary signal when keywords are
     ambiguous.
  3. Falls back to the review text via lightweight keyword scanning when
     neither of the above is conclusive.

This approach is fully NLP-derived: the complaint_category is inferred
from what the customer actually *wrote*, not from any pre-existing label.

Categories (11)
---------------
  Size & Fit
  Product Quality
  Material Issues
  Stitching / Finish Defect
  Colour Difference
  Delivery Delay
  Packaging Damage
  Wrong Item Received
  Missing Component
  Customer Preference
  Product Description Mismatch

Rules
-----
Each rule is a dict with:
  keywords : list[str]  — terms matched against extracted_keywords
  text_patterns : list[str]  — regex patterns matched against review_text
  category : str  — assigned label if triggered
  priority : int  — lower = higher priority (first match wins per signal type)
"""

import re
from typing import Optional

from .utils import get_logger

logger = get_logger("phase2.complaint_classifier")

# ---------------------------------------------------------------------------
# Complaint rules — ordered by priority (ascending = higher priority first)
# ---------------------------------------------------------------------------

_COMPLAINT_RULES: list[dict] = [
    # --- Wrong Item ---
    {
        "priority": 1,
        "category": "Wrong Item Received",
        "keywords": ["wrong item", "wrong colour", "wrong size", "wrong product",
                     "incorrect item", "not what ordered", "different item"],
        "text_patterns": [
            r"wrong (item|colour|color|size|product|order)",
            r"(sent|received|got|delivered).{0,20}wrong",
            r"not (what|the).{0,15}(order|bought|expect)",
            r"incorrect (item|product|size|colour)",
        ],
    },
    # --- Missing Component ---
    {
        "priority": 2,
        "category": "Missing Component",
        "keywords": ["missing component", "missing piece", "incomplete", "not included",
                     "missing part", "belt loop", "accessory missing", "no belt"],
        "text_patterns": [
            r"(missing|absent|not included|no).{0,20}(piece|part|component|accessory|belt|button|loop|tag)",
            r"incomplete (order|item|set|product)",
            r"(component|part|piece) (was|were|is) (missing|absent|not there)",
        ],
    },
    # --- Packaging Damage ---
    {
        "priority": 3,
        "category": "Packaging Damage",
        "keywords": ["damaged packaging", "damaged box", "crushed", "torn packaging",
                     "arrived damaged", "damage in transit", "packaging"],
        "text_patterns": [
            r"(packaging|package|box|parcel|bag).{0,20}(damage|torn|crush|broken|ripped)",
            r"(damage|torn|crushed|broken|ripped).{0,20}(packaging|package|box|parcel)",
            r"arrived.{0,20}(damage|broken|crushed|torn)",
        ],
    },
    # --- Delivery Delay ---
    {
        "priority": 4,
        "category": "Delivery Delay",
        "keywords": ["delivery delay", "late delivery", "slow delivery", "delayed",
                     "late arrival", "weeks wait", "overdue", "never arrived"],
        "text_patterns": [
            r"(delivery|deliver|shipping|shipment|dispatch).{0,20}(delay|late|slow|overdue)",
            r"(waited|waiting).{0,20}(week|day|month)",
            r"(never|still not|yet to).{0,20}(arrive|received|delivered)",
            r"delay.{0,20}notification",
        ],
    },
    # --- Size & Fit ---
    {
        "priority": 5,
        "category": "Size & Fit",
        "keywords": ["wrong size", "too small", "too large", "too big", "too tight",
                     "too loose", "size guide", "sizing", "fit issue", "runs small",
                     "runs large", "size up", "size down", "fit poorly"],
        "text_patterns": [
            r"(too|very).{0,10}(small|large|big|tight|loose|narrow|wide|short|long)",
            r"(size|sizing|fit).{0,20}(wrong|off|issue|problem|mislead|incorrect)",
            r"(run|runs).{0,10}(small|large|big)",
            r"(should have|suggest).{0,15}(size up|size down|larger|smaller)",
            r"does not fit|didn.t fit|won.t fit",
        ],
    },
    # --- Colour Difference ---
    {
        "priority": 6,
        "category": "Colour Difference",
        "keywords": ["colour difference", "colour mismatch", "wrong colour", "different colour",
                     "not same colour", "colour inaccurate", "looks different"],
        "text_patterns": [
            r"(colour|color).{0,20}(different|wrong|off|mismatch|inaccurate|not as|mislead)",
            r"(looks|appeared|seemed).{0,20}(different colour|different color|another shade)",
            r"(darker|lighter|brighter|duller).{0,15}(than|expected|pictured|shown)",
        ],
    },
    # --- Material Issues ---
    {
        "priority": 7,
        "category": "Material Issues",
        "keywords": ["material quality", "fabric quality", "poor material", "cheap material",
                     "faded", "bobbled", "pilling", "thin fabric", "see through",
                     "see-through", "material different", "wrong material"],
        "text_patterns": [
            r"(material|fabric|cloth|textile).{0,20}(poor|cheap|thin|bad|wrong|different|flimsy|rough)",
            r"(faded|pilling|bobbled|bobbling|shrunk|shrank).{0,20}(wash|wear|use)",
            r"(see.through|transparent|sheer)",
            r"not.{0,10}(satin|cotton|wool|silk|leather|linen|polyester)",
        ],
    },
    # --- Stitching / Finish Defect ---
    {
        "priority": 8,
        "category": "Stitching / Finish Defect",
        "keywords": ["stitching", "stitches", "seam", "loose thread", "unravelling",
                     "zip broken", "zip faulty", "button missing", "button popped",
                     "hem coming", "unfinished seam"],
        "text_patterns": [
            r"(stitch|stitching|seam|hem|thread).{0,20}(loose|broken|undone|unravel|coming apart|popped|missing)",
            r"(zip|zipper|button|hook|clasp).{0,20}(broken|faulty|missing|fell off|popped off|doesn.t work)",
            r"(came apart|falling apart|falling off|unravel)",
        ],
    },
    # --- Product Quality (general manufacturing) ---
    {
        "priority": 9,
        "category": "Product Quality",
        "keywords": ["poor quality", "bad quality", "low quality", "quality control",
                     "defective", "manufacturing defect", "fell apart", "broke",
                     "not durable", "did not last"],
        "text_patterns": [
            r"(quality).{0,20}(poor|bad|low|terrible|awful|disappointing|not what expected)",
            r"(fell|falling).{0,10}apart",
            r"(defective|faulty|broken|damaged).{0,15}(product|item|piece)",
            r"(lost|lost its).{0,10}shape",
        ],
    },
    # --- Product Description Mismatch ---
    {
        "priority": 10,
        "category": "Product Description Mismatch",
        "keywords": ["not as described", "description misleading", "misleading description",
                     "not matching listing", "not as advertised", "description wrong",
                     "photo different", "image different"],
        "text_patterns": [
            r"(not|doesn.t|did not).{0,10}(match|look like|as described|as shown|as advertised|as pictured)",
            r"(description|listing|photo|image|picture).{0,20}(wrong|mislead|inaccurate|different|false)",
            r"(nothing like|looks nothing like)",
        ],
    },
    # --- Customer Preference (catch-all for change-of-mind returns) ---
    {
        "priority": 11,
        "category": "Customer Preference",
        "keywords": ["changed mind", "no longer needed", "not my style",
                     "not quite right", "prefer", "wardrobe", "not suitable",
                     "impulse buy", "reconsidered"],
        "text_patterns": [
            r"(changed|change).{0,10}(mind|decision)",
            r"(no longer|don.t).{0,10}(need|want|require)",
            r"(not|wasn.t).{0,20}(right for|suitable for|what i|what I).{0,20}(wardrobe|look|style|need)",
            r"impulse (buy|purchase)",
        ],
    },
]

# Pre-compile all regex patterns once at import time
for _rule in _COMPLAINT_RULES:
    _rule["_compiled"] = [re.compile(p, re.IGNORECASE) for p in _rule["text_patterns"]]


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------

def _match_keywords(extracted_keywords: list[str], rule: dict) -> bool:
    """Return True if any rule keyword appears in extracted_keywords."""
    kw_set = {k.lower() for k in extracted_keywords}
    return any(rk.lower() in kw_set for rk in rule["keywords"])


def _match_text(review_text: str, rule: dict) -> bool:
    """Return True if any compiled pattern matches review_text."""
    return any(pat.search(review_text) for pat in rule["_compiled"])


def classify_complaint(
    review_text: str,
    extracted_keywords: list[str],
    dominant_topic: str,
) -> str:
    """
    Classify a single returned review into a complaint category.

    Signal priority
    ---------------
    1. Keyword match (highest confidence — term came from the review itself)
    2. Text pattern match (regex on raw text — catches phrasing variants)
    3. Topic-based fallback (broad LDA signal)

    Parameters
    ----------
    review_text        : str   — raw (or lightly cleaned) review text
    extracted_keywords : list  — TF-IDF keywords for this review
    dominant_topic     : str   — LDA topic label assigned to this review

    Returns
    -------
    str — one of the 11 complaint category labels
    """
    review_text_lower = review_text.lower() if review_text else ""

    # --- Pass 1: keyword signal ---
    for rule in sorted(_COMPLAINT_RULES, key=lambda r: r["priority"]):
        if _match_keywords(extracted_keywords, rule):
            return rule["category"]

    # --- Pass 2: text pattern signal ---
    for rule in sorted(_COMPLAINT_RULES, key=lambda r: r["priority"]):
        if _match_text(review_text_lower, rule):
            return rule["category"]

    # --- Pass 3: topic-based fallback ---
    return _topic_to_category(dominant_topic)


def _topic_to_category(dominant_topic: str) -> str:
    """
    Map an LDA topic label to a broad complaint category as a last resort.

    This is only reached when neither keywords nor text patterns fire.
    """
    _topic_map = {
        "Size & Fit Issues":                "Size & Fit",
        "Product Quality & Manufacturing":  "Product Quality",
        "Product Listing & Description":    "Product Description Mismatch",
        "Logistics & Delivery":             "Delivery Delay",
        "Packaging & Warehouse":            "Packaging Damage",
        "Customer Preference & Style":      "Customer Preference",
    }
    return _topic_map.get(dominant_topic, "Customer Preference")


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------

def classify_complaints_batch(
    review_texts: list[str],
    extracted_keywords_list: list[list[str]],
    dominant_topics: list[str],
) -> list[str]:
    """
    Classify complaint categories for a batch of RICH reviews.

    Parameters
    ----------
    review_texts           : list[str]
    extracted_keywords_list: list[list[str]] — aligned with review_texts
    dominant_topics        : list[str]       — aligned with review_texts

    Returns
    -------
    list[str]  — complaint category labels aligned with inputs
    """
    if not (len(review_texts) == len(extracted_keywords_list) == len(dominant_topics)):
        raise ValueError("All input lists must have the same length.")

    categories = []
    for text, keywords, topic in zip(review_texts, extracted_keywords_list, dominant_topics):
        cat = classify_complaint(
            review_text=str(text),
            extracted_keywords=keywords if isinstance(keywords, list) else [],
            dominant_topic=str(topic),
        )
        categories.append(cat)

    from collections import Counter
    dist = Counter(categories)
    logger.info("Complaint category distribution:")
    for label, count in dist.most_common():
        logger.info("  %-40s %d", label, count)

    return categories
