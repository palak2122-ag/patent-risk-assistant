# backend/main.py
# Entry point for the Patent Risk & Similarity Assistant API.
# Uses FastAPI to expose a single POST /analyze endpoint.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from backend.services.extractor import extract_concepts
from backend.services.similarity import find_similar_patents

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Patent Risk & Similarity Assistant",
    description="Analyzes code or project descriptions and identifies potentially similar existing patents.",
    version="0.2.0",
)

# Allow requests from the frontend (any origin is fine for local dev / hackathon)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_summary(matches: List[PatentMatch]) -> str:
    """
    Generate a plain-English one-paragraph risk summary based on the
    top match and the overall spread of risk levels found.
    """
    if not matches:
        return "No similar patents were found in the dataset."

    top = matches[0]
    high_count   = sum(1 for m in matches if m.risk_level == "High")
    medium_count = sum(1 for m in matches if m.risk_level == "Medium")
    low_count    = sum(1 for m in matches if m.risk_level == "Low")

    # Describe the top match
    summary = (
        f"The closest match is \"{top.title}\" ({top.patent_id}) "
        f"with a similarity score of {top.similarity_score:.2f} — "
        f"risk level: {top.risk_level}. "
    )

    # Describe the overall risk spread
    parts = []
    if high_count:
        parts.append(f"{high_count} High-risk")
    if medium_count:
        parts.append(f"{medium_count} Medium-risk")
    if low_count:
        parts.append(f"{low_count} Low-risk")

    summary += f"Across all results: {', '.join(parts)} overlap(s) found. "

    # Add actionable advice based on the top risk level
    if top.risk_level == "High":
        summary += "Consider reviewing the claims of the high-risk patent(s) with a legal advisor before proceeding."
    elif top.risk_level == "Medium":
        summary += "Some terminology overlaps exist; review the abstracts and consider differentiating your approach."
    else:
        summary += "Low overlap detected — your project appears to use sufficiently distinct technology."

    return summary


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Health-check endpoint — confirms the API is running."""
    return {"status": "ok", "message": "Patent Risk & Similarity Assistant is running."}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Accepts a code snippet, README, or description and returns:
    - extracted_concepts : key technical terms found in the input
    - matches            : top similar patents ranked by TF-IDF cosine similarity
    - summary            : plain-English risk assessment
    """
    # Basic validation — reject empty or whitespace-only input
    if not request.input or not request.input.strip():
        raise HTTPException(
            status_code=400,
            detail="Input must not be empty. Please provide code or a project description.",
        )

    # Step 1: extract meaningful technical concepts from the user's text
    concepts = extract_concepts(request.input)

    # Step 2: find the most similar patents using TF-IDF cosine similarity
    raw_matches = find_similar_patents(concepts, top_n=5)

    # Step 3: convert raw dicts into validated Pydantic models
    matches = [PatentMatch(**m) for m in raw_matches]

    # Step 4: build a human-readable summary
    summary = _build_summary(matches)

    return AnalyzeResponse(
        extracted_concepts=concepts,
        matches=matches,
        summary=summary,
    )
