# backend/main.py
# Entry point for the Patent Risk & Similarity Assistant API.
# Serves the frontend UI and exposes a POST /analyze endpoint.

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

from backend.services.extractor import extract_concepts
from backend.services.similarity import find_similar_patents, LOCAL_DEMO_PATENTS
from backend.services.patent_retriever import retrieve_patents

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Patent Risk & Similarity Assistant",
    description="Analyzes code or project descriptions and identifies potentially similar existing patents.",
    version="0.3.0",
)

# Allow requests from any origin (required for local dev / hackathon)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static frontend ──────────────────────────────────────────────────────────
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """The JSON body the client sends to /analyze."""
    input: str  # Raw code, README text, or plain description


class PatentMatch(BaseModel):
    """A single patent result returned to the client."""
    patent_id: str
    title: str
    abstract: str
    similarity_score: float          # 0.0 – 1.0
    risk_level: str                  # "High", "Medium", or "Low"
    url: str


class AnalyzeResponse(BaseModel):
    """The full response returned by /analyze."""
    extracted_concepts: List[str]    # Key technical concepts found in the input
    matches: List[PatentMatch]       # Ranked list of similar patents
    summary: str                     # Plain-English risk summary
    data_source: str                 # "live" or "offline"
    data_source_message: Optional[str] = None  # set when offline or on error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_summary(matches: List[PatentMatch], data_source: str) -> str:
    """
    Generate a plain-English one-paragraph risk summary.
    Includes a disclaimer that labels are vocabulary-overlap indicators,
    not legal infringement conclusions.
    """
    disclaimer = (
        "DISCLAIMER: These similarity labels reflect vocabulary overlap only "
        "and are NOT legal infringement conclusions. Always consult a qualified "
        "patent attorney before making legal or commercial decisions."
    )

    if not matches:
        return f"No similar patents were found. {disclaimer}"

    top = matches[0]
    high_count   = sum(1 for m in matches if m.risk_level == "High")
    medium_count = sum(1 for m in matches if m.risk_level == "Medium")
    low_count    = sum(1 for m in matches if m.risk_level == "Low")

    source_label = "live patent search" if data_source == "live" else "offline demo dataset"

    summary = (
        f"The closest match in the {source_label} is \"{top.title}\" ({top.patent_id}) "
        f"with a similarity score of {top.similarity_score:.2f} — "
        f"risk level: {top.risk_level}. "
    )

    parts = []
    if high_count:
        parts.append(f"{high_count} High-risk")
    if medium_count:
        parts.append(f"{medium_count} Medium-risk")
    if low_count:
        parts.append(f"{low_count} Low-risk")

    summary += f"Across all results: {', '.join(parts)} overlap(s) found. "

    if top.risk_level == "High":
        summary += "Consider reviewing the claims of the high-risk patent(s) with a legal advisor before proceeding. "
    elif top.risk_level == "Medium":
        summary += "Some terminology overlaps exist; review the abstracts and consider differentiating your approach. "
    else:
        summary += "Low overlap detected — your project appears to use sufficiently distinct technology. "

    summary += disclaimer
    return summary


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Serve the frontend UI from the root URL."""
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    return FileResponse(index_path, media_type="text/html")


@app.get("/health")
def health():
    """JSON health-check."""
    return {"status": "ok", "message": "Patent Risk & Similarity Assistant is running."}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Accepts a code snippet, README, or description and returns:
    - extracted_concepts     : key technical terms found in the input
    - matches                : top similar patents ranked by TF-IDF cosine similarity
    - summary                : plain-English risk assessment with disclaimer
    - data_source            : "live" (PatentsView API) or "offline" (demo dataset)
    - data_source_message    : set when offline or when an error occurred
    """
    if not request.input or not request.input.strip():
        raise HTTPException(
            status_code=400,
            detail="Input must not be empty. Please provide code or a project description.",
        )

    # Step 1: extract meaningful technical concepts for the UI pills
    concepts = extract_concepts(request.input)

    # Step 2: attempt live patent retrieval from PatentsView API.
    # The retriever reads PATENTSVIEW_API_KEY from the environment — the key
    # is NEVER returned to the frontend.
    retrieval = retrieve_patents(request.input)

    # Step 3: choose the patent pool and record the data source
    if retrieval.source == "live" and retrieval.patents:
        patent_pool = retrieval.patents
        data_source = "live"
        data_source_message = None
    else:
        # Do NOT silently serve local patents as if they were live results.
        # Return the error and empty matches so the frontend shows the
        # "unavailable" state clearly.
        return AnalyzeResponse(
            extracted_concepts=concepts,
            matches=[],
            summary=(
                retrieval.error or
                "Live patent search unavailable. "
                "Configure PATENTSVIEW_API_KEY to enable live patent results."
            ),
            data_source="offline",
            data_source_message=(
                retrieval.error or
                "Live patent search unavailable. "
                "Configure PATENTSVIEW_API_KEY to enable live patent results."
            ),
        )

    # Step 4: score the live patents against the user's description
    raw_matches = find_similar_patents(
        concepts,
        top_n=5,
        raw_text=request.input,
        patents=patent_pool,
    )

    # Step 5: convert to Pydantic models
    matches = [PatentMatch(**m) for m in raw_matches]

    # Step 6: build a human-readable summary (includes disclaimer)
    summary = _build_summary(matches, data_source)

    return AnalyzeResponse(
        extracted_concepts=concepts,
        matches=matches,
        summary=summary,
        data_source=data_source,
        data_source_message=data_source_message,
    )
