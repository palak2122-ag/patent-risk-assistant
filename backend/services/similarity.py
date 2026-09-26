# backend/services/similarity.py
#
# Compares extracted user concepts against the local patent dataset and
# returns ranked matches with similarity scores and risk labels.
#
# Approach: TF-IDF vectorization + cosine similarity.
# - No GPU, no paid API, no model download required.
# - scikit-learn ships prebuilt Windows wheels for Python 3.13.

import json
import os
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Load the patent dataset once at module import time
# (avoids re-reading the file on every request)
# ---------------------------------------------------------------------------

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "patents.json")

def _load_patents() -> List[dict]:
    """Read patents.json and return the list of patent dicts."""
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

PATENTS = _load_patents()

# ---------------------------------------------------------------------------
# Risk labeling
# ---------------------------------------------------------------------------

def _risk_label(score: float) -> str:
    """
    Convert a cosine similarity score (0.0–1.0) into a plain risk label.

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
    concepts: List[str],
    top_n: int = 5,
) -> List[dict]:
    """
    Score every patent in the dataset against the user's extracted concepts
    and return the top `top_n` matches.

    Parameters
    ----------
    concepts : list of keyword strings from extractor.extract_concepts()
    top_n    : how many results to return (default 5)

    Returns
    -------
    List of dicts, each with keys:
        patent_id, title, abstract, url, similarity_score, risk_level
    Sorted by similarity_score descending.
    """
    if not concepts:
        return []

    # Build the query string from extracted concepts
    query = " ".join(concepts)

    # Build one document per patent: title + abstract together
    # (gives the vectorizer more signal than abstract alone)
    patent_docs = [
        f"{p['title']} {p['abstract']}"
        for p in PATENTS
    ]

    # Fit TF-IDF on all patent docs + the query together so they share
    # the same vocabulary, then split the vectors back apart
    all_docs = patent_docs + [query]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # unigrams + bigrams, same as extractor
        # No stop_words here — the extractor already filtered stopwords.
        # Letting sklearn re-filter silently drops tokens like "system" or
        # "real" that the extractor kept, which zeros out the query vector.
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    # Query vector is the last row; patent vectors are all preceding rows
    query_vec = tfidf_matrix[-1]
    patent_vecs = tfidf_matrix[:-1]

    # Compute cosine similarity: shape (1, num_patents) → flatten to 1-D
    scores = cosine_similarity(query_vec, patent_vecs).flatten()

    # Pair each patent with its score and sort descending
    scored: List[Tuple[float, dict]] = sorted(
        zip(scores, PATENTS),
        key=lambda x: x[0],
        reverse=True,
    )

    # Build and return result dicts for the top N matches
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
