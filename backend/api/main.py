"""
CAPA Recommendation System — FastAPI Backend
Endpoints:
  POST /recommend       — AI-driven CAPA recommendations
  POST /capa/add        — Add new CAPA record to historical dataset
  GET  /history         — Recent recommendation history
  GET  /stats           — Dashboard statistics
  GET  /health          — Health check
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.recommendation import (
    load_assets, recommend, add_capa_record, get_history, get_stats
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_assets()
    yield


app = FastAPI(
    title="CAPA Recommendation System API",
    description="AI-powered CAPA recommendations using FAISS + SentenceTransformers",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    incident_description: str
    severity: Optional[str] = None
    department: Optional[str] = None

    @field_validator("incident_description")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("incident_description must not be empty")
        if len(v) < 10:
            raise ValueError("Please provide a more detailed description (min 10 characters)")
        return v


class AddCapaRequest(BaseModel):
    product: str
    department: str
    severity: str
    root_cause: str
    incident_summary: str
    root_cause_detail: Optional[str] = ""
    corrective_action: str
    preventive_action: str
    investigator_name: Optional[str] = ""
    investigator_role: Optional[str] = ""
    regulatory_reference: Optional[str] = ""
    open_date: Optional[str] = ""
    ca_completion_date: Optional[str] = ""
    pa_completion_date: Optional[str] = ""


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "CAPA Recommendation System v2"}


@app.post("/recommend")
def recommend_endpoint(body: RecommendRequest):
    try:
        result = recommend(body.incident_description)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/capa/add")
def add_capa_endpoint(body: AddCapaRequest):
    try:
        result = add_capa_record(body.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add CAPA: {str(e)}")


@app.get("/history")
def history_endpoint():
    try:
        return {"history": get_history()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats_endpoint():
    try:
        return get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
