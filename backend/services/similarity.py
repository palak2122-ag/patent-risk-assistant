# backend/services/similarity.py
#
# Compares the user's input against a provided patent list and returns
# ranked matches with similarity scores and risk labels.
#
# Approach: TF-IDF vectorization + cosine similarity.
# - No GPU, no paid API, no model download required.
# - scikit-learn ships prebuilt Windows wheels for Python 3.13.
#
# Key design decision — query representation
# -------------------------------------------
# The query fed to TF-IDF is the RAW CLEANED TEXT from the user, NOT the
# joined list of extracted concepts.
#
# Why: extract_concepts() filters stopwords and truncates to top_n terms.
# Joining those back into a string then re-tokenizing in sklearn creates
# two problems:
#   1. Discriminative words beyond the top_n cutoff are silently dropped.
#   2. Adjacent pre-extracted bigrams get re-glued, producing phantom
#      cross-bigrams (e.g. "learning system" from "machine learning" +
#      "system detecting") that pollute the query vector.
#
# Using the full cleaned text lets sklearn's TF-IDF tokenizer see every
# real adjacent word pair (e.g. "edge devices", "real time", "neural
# networks") without any information loss.  sklearn's built-in stop_words
# handles common English words on the patent side as well, keeping the
# vocabulary comparable.
#
# The extracted_concepts list is still returned by the API and displayed in
# the UI — it is used only for human-readable display, not for scoring.
#
# Patent list input
# -----------------
# find_similar_patents() now accepts an explicit `patents` parameter.
# This decouples the scorer from any fixed data source: the caller decides
# whether to pass live API results or the local demo dataset.
# The local PATENTS constant is kept for use as a labelled offline fallback.

import json
import os
import re
from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Local fallback dataset
# (used ONLY when explicitly passed by the caller as an offline demo)
# ---------------------------------------------------------------------------

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "patents.json")


def _load_local_patents() -> List[dict]:
    """Read patents.json and return the list of patent dicts."""
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Loaded once at import time; available to callers that want the demo dataset.
# Field names here follow the local schema: patent_id, title, abstract, url.
LOCAL_DEMO_PATENTS: List[dict] = _load_local_patents()

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """
    Lowercase and strip non-alphabetic characters.
    Applied to both the query and patent documents so they share the same
    character space before TF-IDF vectorization.
    Hyphens are converted to spaces so "real-time" → "real time".
    """
    text = text.lower()
    text = re.sub(r"[-]", " ", text)          # hyphen → space
    text = re.sub(r"[^a-z\s]", " ", text)     # strip everything else
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------------------------
# Risk labeling
# ---------------------------------------------------------------------------

def _risk_label(score: float) -> str:
    """
    Convert a cosine similarity score (0.0–1.0) into a plain risk label.

    These labels indicate VOCABULARY OVERLAP with a known patent, not
    legal infringement conclusions.

    Thresholds (tunable):
        >= 0.20  →  High    (meaningful lexical overlap)
        >= 0.08  →  Medium  (some shared terminology)
        <  0.08  →  Low     (little to no overlap)
    """
    if score >= 0.20:
        return "High"
    elif score >= 0.08:
        return "Medium"
    else:
        return "Low"

# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def find_similar_patents(
    concepts: List[str],                # display-only; NOT used for scoring
    top_n: int = 5,
    raw_text: str = "",                 # full user input — the TF-IDF query
    patents: Optional[List[dict]] = None,  # list of patent dicts to score against
) -> List[dict]:
    """
    Score every patent in `patents` against the user's raw input text and
    return the top `top_n` matches.

    Parameters
    ----------
    concepts : list of keyword strings from extractor.extract_concepts()
               (returned in result dicts for display; NOT used for scoring)
    top_n    : how many results to return (default 5)
    raw_text : the original user input string.  Cleaned and used as the
               TF-IDF query document.  Falls back to joining `concepts` if
               empty (backward-compatible behaviour).
    patents  : list of patent dicts to score against.  Each dict must have
               keys: patent_id, title, abstract, url.
               If None or empty, returns an empty list — the caller is
               responsible for providing a non-empty patent pool.

    Scoring approach
    ----------------
    1. Both the query and each patent's title+abstract are cleaned with
       _clean() (lowercase, hyphen→space, strip non-alpha).
    2. All documents (patents + query) are jointly vectorized with
       TfidfVectorizer(ngram_range=(1,2), stop_words="english").
    3. Cosine similarity between the query vector and each patent vector
       gives a score in [0, 1].
    4. Scores are mapped to High/Medium/Low via fixed thresholds.

    Returns
    -------
    List of dicts, each with keys:
        patent_id, title, abstract, url, similarity_score, risk_level
    Sorted by similarity_score descending.
    """
    if not patents:
        return []

    if not concepts and not raw_text:
        return []

    # Build the query document: prefer raw_text; fall back to concept join
    if raw_text and raw_text.strip():
        query_doc = _clean(raw_text)
    else:
        query_doc = _clean(" ".join(concepts))

    # Build one document per patent: title + abstract together.
    patent_docs = [
        _clean(f"{p['title']} {p['abstract']}")
        for p in patents
    ]

    # Fit TF-IDF on all patent docs + the query so they share one vocabulary.
    all_docs = patent_docs + [query_doc]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    # Query vector is the last row; patent vectors are all preceding rows
    query_vec = tfidf_matrix[-1]
    patent_vecs = tfidf_matrix[:-1]

    # Cosine similarity: shape (1, num_patents) → flatten to 1-D
    scores = cosine_similarity(query_vec, patent_vecs).flatten()

    # Pair each patent with its score and sort descending
    scored: List[Tuple[float, dict]] = sorted(
        zip(scores, patents),
        key=lambda x: x[0],
        reverse=True,
    )

    # Build result dicts for the top N matches
    results = []
    for score, patent in scored[:top_n]:
        results.append({
            "patent_id": patent["patent_id"],
            "title": patent["title"],
            "abstract": patent["abstract"],
            "url": patent["url"],
            "similarity_score": round(float(score), 4),
            "risk_level": _risk_label(float(score)),
        })

    return results
