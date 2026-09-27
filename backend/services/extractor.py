# backend/services/extractor.py
#
# Extracts the most meaningful technical keywords from a user's input
# (code snippet, README, or plain description).
#
# Approach: no paid APIs needed.
# 1. Lowercase and tokenize the text.
# 2. Remove common English stopwords and short tokens.
# 3. Return the top N most meaningful words/bigrams as concepts.
#
# IMPORTANT — what this function is used for:
#   The returned list is shown in the UI as "Extracted Concepts" pills.
#   It is NOT used as the TF-IDF query string; similarity.py uses the
#   caller's raw text directly so that real adjacent bigrams are preserved.

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

def extract_concepts(text: str, top_n: int = 10) -> List[str]:
    """
    Extract the top `top_n` technical keywords/phrases from `text` for display.

    Returns a list of keyword strings, e.g.:
        ["object detection", "neural networks", "edge devices", "real time", ...]

    Algorithm
    ---------
    1. Clean: strip punctuation, lowercase.
    2. Build a filtered token list (stopwords removed, length >= 3).
    3. Form bigrams from *adjacent filtered tokens* — this preserves real
       technical phrases even when stopwords sit between constituent words
       in the original (e.g. "networks on edge" → filtered tokens are adjacent
       so "networks edge" would not form; see note below).
    4. Score candidates:
         bigrams  get  base_freq * 3   (strong preference for phrases)
         unigrams get  base_freq * 1
       This ensures bigrams win over bare unigrams even when every term
       appears only once (the common case for short single-sentence inputs).
    5. Greedily select top_n terms, skipping unigrams already covered by a
       chosen bigram.

    Note on bigram adjacency: bigrams are built from *originally adjacent*
    token pairs where BOTH tokens pass the stopword/length filter.
    "neural networks on edge devices" → tokens [..., "networks", "on",
    "edge", ...] → "on" fails the filter, so "networks" and "edge" are not
    treated as adjacent and "networks edge" is NOT produced.  Only genuine
    neighbouring content words form bigrams (e.g. "neural networks",
    "edge devices", "real time").
    """
    # Step 1 — normalize: lowercase, replace non-alpha with spaces
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    tokens = cleaned.split()

    # Step 2 — decide which tokens are "meaningful" (non-stopword, length >= 3)
    is_meaningful = [t not in STOPWORDS and len(t) >= 3 for t in tokens]
    unigrams = [t for t, ok in zip(tokens, is_meaningful) if ok]

    # Step 3 — build bigrams only from ORIGINALLY ADJACENT token pairs where
    # BOTH tokens pass the filter.  This avoids phantom phrases like
    # "system detecting" that arise when a stopword sat between them.
    # Example: "networks on edge" → tokens [networks, on, edge]
    #   is_meaningful: [True, False, True] → the pair is NOT adjacent after
    #   filtering, so "networks edge" is correctly excluded.
    bigrams = [
        f"{tokens[i]} {tokens[i + 1]}"
        for i in range(len(tokens) - 1)
        if is_meaningful[i] and is_meaningful[i + 1]
    ]

    # Step 4 — score: bigrams get 3x weight so they beat unigrams when all
    # frequencies are 1 (the typical case for short single-sentence inputs).
    freq_uni = Counter(unigrams)
    freq_bi  = Counter(bigrams)

    # Build a unified scored list: (weighted_score, term)
    # Bigrams come first within the same effective score bracket.
    candidates = (
        [(freq_bi[b] * 3, b) for b in freq_bi]
        + [(freq_uni[u] * 1, u) for u in freq_uni]
    )
    # Sort by score descending; for equal scores the insertion order (bigrams
    # first) is preserved because Python's sort is stable.
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Step 5 — greedily select up to top_n with two overlap rules:
    #
    # Rule A (unigram dedup): skip a unigram if its word is already covered
    #   by a previously chosen bigram — avoids redundant pills like
    #   "neural networks" + "neural".
    #
    # Rule B (bigram chain dedup): skip a bigram if its FIRST word is the
    #   same as the LAST word of the previously added bigram — avoids the
    #   sliding-window chain  "machine learning" / "learning system" /
    #   "system detecting" that makes the pill row read as
    #   "machine learninglearning systemsystem detecting…" when the user
    #   copies or scans the text.
    chosen: List[str] = []
    chosen_words: set = set()   # words already covered by a chosen bigram
    last_bigram_end: str = ""   # last word of the most recently chosen bigram

    for _score, term in candidates:
        parts = term.split()
        if len(parts) == 2:
            first_word, last_word = parts[0], parts[1]
            # Rule B: skip if this bigram's first word == last word of prev bigram
            if first_word == last_bigram_end:
                continue
            chosen.append(term)
            chosen_words.update(parts)
            last_bigram_end = last_word
        else:
            # Rule A: skip unigrams already covered by a chosen bigram
            if term not in chosen_words:
                chosen.append(term)
        if len(chosen) >= top_n:
            break

    return chosen
