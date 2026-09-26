# backend/main.py
# Entry point for the Patent Risk & Similarity Assistant API.
# Uses FastAPI to expose a single POST /analyze endpoint.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Patent Risk & Similarity Assistant",
    description="Analyzes code or project descriptions and identifies potentially similar existing patents.",
    version="0.1.0",
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

def mock_analyze(user_input: str) -> AnalyzeResponse:
    """
    Returns hard-coded mock data so the endpoint is immediately testable.
    Steps 2–5 will replace this with real LLM + patent search + scoring logic.
    """
    # Pretend the LLM extracted these concepts from the input
    concepts = [
        "real-time object detection",
        "edge inference pipeline",
        "convolutional neural network optimization",
    ]

    # Pretend these patents were found and scored by the search + similarity service
    matches = [
        PatentMatch(
            patent_id="US10,123,456",
            title="System and Method for Real-Time Edge Object Detection",
            abstract="A system that performs convolutional neural network inference on edge devices with reduced latency using quantization techniques.",
            similarity_score=0.87,
            risk_level="High",
            url="https://patents.google.com/patent/US10123456",
        ),
        PatentMatch(
            patent_id="US9,876,543",
            title="Optimized CNN Pipeline for Embedded Vision Systems",
            abstract="Methods for compressing and deploying deep learning models on resource-constrained hardware for real-time visual processing.",
            similarity_score=0.61,
            risk_level="Medium",
            url="https://patents.google.com/patent/US9876543",
        ),
        PatentMatch(
            patent_id="US8,765,432",
            title="Low-Power Neural Network Accelerator Architecture",
            abstract="Hardware architecture for accelerating neural network computations with emphasis on power efficiency.",
            similarity_score=0.42,
            risk_level="Low",
            url="https://patents.google.com/patent/US8765432",
        ),
    ]

    summary = (
        "Your input closely matches 1 high-risk patent (US10,123,456) related to "
        "real-time edge inference. Consider reviewing its claims before proceeding. "
        "1 medium-risk and 1 low-risk overlap were also found."
    )

    return AnalyzeResponse(
        extracted_concepts=concepts,
        matches=matches,
        summary=summary,
    )


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
    Accepts a code snippet, README, or description and returns a list of
    potentially similar patents with similarity scores and a risk summary.
    """
    # Basic validation — reject empty or whitespace-only input
    if not request.input or not request.input.strip():
        raise HTTPException(
            status_code=400,
            detail="Input must not be empty. Please provide code or a project description.",
        )

    # TODO (Step 2): replace mock_analyze with real concept extractor
    # TODO (Step 3): replace mock patents with real USPTO API results
    # TODO (Step 4): replace hardcoded scores with sentence-transformer similarity
    return mock_analyze(request.input)
