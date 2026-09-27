# backend/services/patent_retriever.py
#
# Retrieves relevant patents from the PatentsView PatentSearch API based on
# the user's raw input description.
#
# Authentication
# --------------
# Requires the environment variable PATENTSVIEW_API_KEY.
# The key is sent as the HTTP request header:  X-Api-Key: <key>
# It is NEVER included in the response or logged.
#
# API reference
# -------------
# Base URL  : https://search.patentsview.org
# Endpoint  : POST /api/v1/patent
# Docs      : https://search.patentsview.org/docs/docs/Search%20API/SearchAPIReference/
#
# Fallback behaviour
# ------------------
# If the API key is missing, the API is unreachable, times out, or returns an
# error, this module returns a RetrievalResult with:
#   patents  = []          (empty — caller decides what to do)
#   source   = "offline"
#   error    = <human-readable message>
#
# The caller (main.py) is responsible for deciding whether to fall back to the
# local demo dataset and for surfacing the correct label to the frontend.

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://search.patentsview.org"
_ENDPOINT = "/api/v1/patent"
_TIMEOUT_SECONDS = 15
_MAX_RESULTS = 25   # PatentsView returns up to 1 000; 25 is plenty for TF-IDF

# Fields we request from PatentsView — the minimum we need.
# (patent_id, patent_title, patent_abstract)
_REQUESTED_FIELDS = ["patent_id", "patent_title", "patent_abstract"]

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    The result of a patent retrieval attempt.

    Attributes
    ----------
    patents : list of dicts with keys patent_id, title, abstract, url.
              Empty when source == "offline" or an error occurred.
    source  : "live"    — results came from PatentsView API
              "offline" — API was not contacted (key missing) or failed
    error   : None on success; human-readable string on any failure.
    """
    patents: List[dict] = field(default_factory=list)
    source: str = "offline"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_search_query(raw_text: str) -> str:
    """
    Derive a compact full-text search string from the user's raw input.

    Strategy:
    1. Lowercase and strip punctuation (keep spaces).
    2. Remove very common English stopwords.
    3. Take the first 12 distinct meaningful tokens (≥4 chars).
    4. Join with spaces — PatentsView _text_any treats this as OR across terms.

    Keeping the query short prevents URL/body bloat and focuses the search on
    the most discriminative terms from the description.  A wider OR search
    recovers more potentially relevant patents; TF-IDF cosine scoring then
    re-ranks them against the full description.
    """
    # Basic stopwords for query pruning (not the same as extractor.py's list)
    _QUERY_STOPS = {
        "a","an","the","and","or","but","in","on","at","to","for","of","with",
        "by","from","is","are","was","were","be","been","have","has","will",
        "that","this","these","those","it","its","as","if","not","no","into",
        "using","used","use","which","each","can","may","would","also","such",
        "their","they","we","our","you","when","where","how","all","any","both",
    }

    cleaned = re.sub(r"[^a-zA-Z\s]", " ", raw_text.lower())
    tokens = cleaned.split()
    seen: set = set()
    selected: List[str] = []
    for tok in tokens:
        if len(tok) >= 4 and tok not in _QUERY_STOPS and tok not in seen:
            seen.add(tok)
            selected.append(tok)
        if len(selected) >= 12:
            break

    return " ".join(selected)


def _normalize_patent(record: dict) -> dict:
    """
    Map a single PatentsView API record to the internal patent dict shape.

    PatentsView field → internal field:
        patent_id       → patent_id
        patent_title    → title
        patent_abstract → abstract
        (constructed)   → url
    """
    pid = (record.get("patent_id") or "").strip()
    title = (record.get("patent_title") or "").strip()
    abstract = (record.get("patent_abstract") or "").strip()

    # Build a Google Patents public URL from the patent_id.
    # PatentsView returns numeric-only IDs (e.g. "10382765") for utility patents
    # and prefixed IDs (e.g. "D345393", "RE47123") for design/reissue patents.
    # We always prepend "US" for Google Patents; it handles all US grant types.
    if pid:
        url = f"https://patents.google.com/patent/US{pid}"
    else:
        url = ""

    return {
        "patent_id": pid,
        "title": title,
        "abstract": abstract,
        "url": url,
    }


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def retrieve_patents(raw_text: str) -> RetrievalResult:
    """
    Query the PatentsView PatentSearch API for patents relevant to raw_text.

    Parameters
    ----------
    raw_text : the user's full project/invention description (arbitrary text).
               Used to derive a dynamic search query — never hardcoded.

    Returns
    -------
    RetrievalResult
        .patents — list of normalized patent dicts (may be empty on error)
        .source  — "live" or "offline"
        .error   — None on success; message string on any failure
    """
    # ── 1. Check for API key ─────────────────────────────────────────────────
    api_key = os.environ.get("PATENTSVIEW_API_KEY", "").strip()
    if not api_key:
        return RetrievalResult(
            source="offline",
            error=(
                "Live patent search unavailable. "
                "Configure PATENTSVIEW_API_KEY to enable live patent results."
            ),
        )

    # ── 2. Build search query from the user's description ───────────────────
    search_terms = _build_search_query(raw_text)
    if not search_terms:
        return RetrievalResult(
            source="offline",
            error="Could not extract search terms from the provided description.",
        )

    # ── 3. Construct request payload ─────────────────────────────────────────
    # _text_any on patent_abstract: OR-matches any of the space-separated terms.
    # We also request sort by patent_date desc to bias toward recent patents.
    payload = {
        "q": {
            "_text_any": {
                "patent_abstract": search_terms,
            }
        },
        "f": _REQUESTED_FIELDS,
        "o": {"size": _MAX_RESULTS},
        "s": [{"patent_date": "desc"}],
    }

    headers = {
        "X-Api-Key": api_key,   # key stays server-side; never returned to frontend
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    # ── 4. Make the HTTP request ─────────────────────────────────────────────
    url = f"{_BASE_URL}{_ENDPOINT}"
    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
            response = client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException:
        return RetrievalResult(
            source="offline",
            error=(
                f"PatentsView API request timed out after {_TIMEOUT_SECONDS}s. "
                "Check your network connection or try again."
            ),
        )
    except httpx.RequestError as exc:
        return RetrievalResult(
            source="offline",
            error=f"PatentsView API could not be reached: {type(exc).__name__}.",
        )

    # ── 5. Handle HTTP errors ────────────────────────────────────────────────
    if response.status_code == 401:
        return RetrievalResult(
            source="offline",
            error="PatentsView API key is invalid or unauthorised (HTTP 401).",
        )
    if response.status_code == 403:
        return RetrievalResult(
            source="offline",
            error="Access denied by PatentsView API (HTTP 403). Check your API key.",
        )
    if response.status_code != 200:
        return RetrievalResult(
            source="offline",
            error=(
                f"PatentsView API returned an unexpected status: "
                f"HTTP {response.status_code}."
            ),
        )

    # ── 6. Parse and validate the response body ──────────────────────────────
    try:
        body = response.json()
    except Exception:
        return RetrievalResult(
            source="offline",
            error="PatentsView API returned a malformed (non-JSON) response.",
        )

    # The API always includes an "error" key; non-null means server-side error
    if body.get("error"):
        return RetrievalResult(
            source="offline",
            error=f"PatentsView API error: {body['error']}",
        )

    raw_patents = body.get("patents") or []
    if not isinstance(raw_patents, list):
        return RetrievalResult(
            source="offline",
            error="PatentsView API response had an unexpected 'patents' format.",
        )

    # ── 7. Normalize and filter out records missing required fields ──────────
    normalized: List[dict] = []
    for record in raw_patents:
        p = _normalize_patent(record)
        # Skip records with no ID or no abstract (can't be scored meaningfully)
        if p["patent_id"] and p["abstract"]:
            normalized.append(p)

    if not normalized:
        return RetrievalResult(
            source="offline",
            error=(
                "PatentsView returned no patents matching the search terms derived "
                "from your description. Try a more detailed description."
            ),
        )

    return RetrievalResult(patents=normalized, source="live", error=None)
