# main.py  (Step 12 - part 1)
# Job: the web API. Other programs (and our Streamlit page) talk to the RAG system through it.
# Run (from the project folder):  uvicorn src.api.main:app
# Interactive docs open at:       http://localhost:8000/docs

import time
from contextlib import asynccontextmanager
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.config import load_config
from src.logger import get_logger
from src.rag.generator import GenerationError

log = get_logger("api")


# ---------- the shapes of the data going in and out ----------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500,
                          description="The employee's question, up to 500 characters")


class Citation(BaseModel):
    number: int
    file_name: str
    section: Optional[str] = None
    page: Optional[int] = None
    snippet: Optional[str] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    refusal_reason: Optional[str] = None
    citations: List[Citation]
    citation_warnings: List[str] = []
    cache: str                                   # "exact", "semantic" or "none"
    cache_similarity: Optional[float] = None
    response_time_seconds: float


# ---------- start-up: load the heavy parts ONCE ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # imported here so that importing this file stays quick and easy to test
    from src.cache.cache_manager import CachedRAG
    from src.rag.pipeline import build_pipeline

    log.info("Loading the RAG pipeline (embedding model + classifier) ...")
    app.state.rag = CachedRAG(build_pipeline())
    log.info("API ready.")
    yield


app = FastAPI(title="NovaTech Knowledge Q&A", version="1.0", lifespan=lifespan)


# ---------- endpoints ----------

def check_ollama():
    """Is Ollama reachable, and is our model downloaded?"""
    settings = load_config()["llm"]
    try:
        reply = requests.get(settings["base_url"] + "/api/tags", timeout=3)
        reply.raise_for_status()
        names = [m["name"] for m in reply.json().get("models", [])]
        wanted = settings["model"]
        # Ollama lists names like "llama3.2:1b"; "llama3.2" alone means ":latest"
        has_model = wanted in names or (":" not in wanted and wanted + ":latest" in names)
        return {"reachable": True, "model": wanted, "model_available": has_model}
    except Exception as error:
        return {"reachable": False, "model": settings["model"], "model_available": False,
                "error": str(error)[:120]}


@app.get("/health")
def health(request: Request):
    ollama = check_ollama()
    pipeline_loaded = getattr(request.app.state, "rag", None) is not None
    ok = pipeline_loaded and ollama["reachable"] and ollama["model_available"]
    return {
        "status": "ok" if ok else "degraded",
        "pipeline_loaded": pipeline_loaded,
        "kb_version": load_config()["project"]["kb_version"],
        "ollama": ollama,
    }


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request):
    start = time.perf_counter()
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="The system is still starting up.")
    try:
        result = rag.ask(body.question)
    except ValueError as error:                       # empty / too long question
        raise HTTPException(status_code=400, detail=str(error))
    except GenerationError as error:                  # Ollama is down
        raise HTTPException(status_code=503, detail=str(error))
    except Exception:
        log.exception("Unexpected error while answering")
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        refused=result["refused"],
        refusal_reason=result["refusal_reason"],
        citations=result["citations"],
        citation_warnings=result["citation_warnings"],
        cache=result.get("cache", "none"),
        cache_similarity=result.get("cache_similarity"),
        response_time_seconds=round(time.perf_counter() - start, 3),
    )