# backend/services/extractor.py
#
# Extracts the most meaningful technical keywords from a user's input
# (code snippet, README, or plain description).
#
# Approach: no paid APIs needed.
# 1. Lowercase and tokenize the text.
# 2. Remove common English stopwords and short tokens.
# 3. Return the top N most-frequent meaningful words/bigrams as concepts.

import re
from collections import Counter
from typing import List

# ---------------------------------------------------------------------------
# Stopwords — common English words that carry no technical meaning
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "just", "also", "as",
    "if", "then", "that", "this", "these", "those", "it", "its", "we", "our",
    "you", "your", "they", "their", "he", "she", "his", "her", "i", "my",
    "which", "who", "what", "when", "where", "how", "all", "any", "both",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "once",
    # Code-specific noise
    "def", "class", "return", "import", "from", "self", "true", "false",
    "none", "null", "var", "let", "const", "function", "new", "print",
    "type", "int", "str", "bool", "list", "dict", "set", "get", "set",
}

# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def extract_concepts(text: str, top_n: int = 8) -> List[str]:
    """
    Extract the top `top_n` technical keywords from `text`.

    Returns a list of keyword strings, e.g.:
        ["machine learning", "object detection", "neural network", ...]

    Steps:
        1. Clean: strip code punctuation, lowercase everything.
        2. Build unigrams (single words) and bigrams (two-word phrases).
        3. Filter out stopwords and tokens shorter than 3 characters.
        4. Count frequency and return the most common terms.
    """
    # Step 1 — normalize: lowercase, replace non-alpha with spaces
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    tokens = cleaned.split()

    # Step 2 — filter tokens: remove stopwords and very short words
    meaningful = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]

    # Step 3 — build unigrams
    unigrams = list(meaningful)

    # Step 4 — build bigrams (pairs of consecutive meaningful words)
    bigrams = [
        f"{meaningful[i]} {meaningful[i + 1]}"
        for i in range(len(meaningful) - 1)
    ]

    # Step 5 — count all terms together and pick the top N
    counter = Counter(unigrams + bigrams)
    top_terms = [term for term, _ in counter.most_common(top_n * 2)]

    # Step 6 — prefer bigrams over bare unigrams.
    # Rule: skip a unigram if a bigram that contains it is already chosen.
    # Use an exact-membership set — never substring matching, which wrongly
    # blocks e.g. "real time" because "real" was already added.
    chosen: List[str] = []
    chosen_words: set = set()   # individual words covered by chosen bigrams
    for term in top_terms:
        parts = term.split()
        if len(parts) == 2:
            # Always accept bigrams; mark their words as covered
            chosen.append(term)
            chosen_words.update(parts)
        else:
            # Accept a unigram only if it isn't already covered by a bigram
            if term not in chosen_words:
                chosen.append(term)
        if len(chosen) >= top_n:
            break

    return chosen
