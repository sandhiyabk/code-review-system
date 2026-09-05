# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from core.pipeline import review_code
from core.llm_client import get_llm_client
import uvicorn

app = FastAPI(
    title="AI Code Review Assistant",
    description="RAG + LLM + AST powered code reviewer",
    version="1.0.0"
)

# ─── Models ───────────────────────────────────────────
class CodeInput(BaseModel):
    code: str
    language: str = "python"

class ComplexityInfo(BaseModel):
    time: str
    space: str
    explanation: str

class ReviewOutput(BaseModel):
    bugs: List[str]
    complexity: ComplexityInfo
    suggestions: List[str]
    quality_score: int
    improved_code: str

# ─── Routes ───────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "AI Code Review Assistant is running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    # Report which LLM backend is actually active so the health endpoint
    # reflects real configuration (Groq cloud vs local Ollama, etc.).
    try:
        info = get_llm_client().get_backend_info()
        model = info["model"]
        backend = info["backend"]
    except Exception:
        # Never fail the health check because of backend probing
        model = "unknown"
        backend = "unknown"
    return {
        "status": "healthy",
        "model": model,
        "backend": backend,
        "vector_db": "chromadb",
        "rag": "active"
    }

@app.post("/review", response_model=ReviewOutput)
async def review(input: CodeInput):
    if not input.code.strip():
        raise HTTPException(
            status_code=400,
            detail="Code cannot be empty"
        )
    if len(input.code) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Code too long — maximum 5000 characters"
        )
    if input.language.lower() != "python":
        raise HTTPException(
            status_code=400,
            detail="Currently only Python is supported"
        )
    try:
        result = review_code(input.code)
        return result
    except Exception as e:
        # Convert backend errors into a helpful message instead of a raw
        # traceback string. The pipeline already returns user-friendly
        # dicts for LLM failures, so this only catches truly unexpected
        # errors — and we still keep the detail readable.
        from core.llm_reviewer import get_error_suggestions
        raise HTTPException(
            status_code=500,
            detail=(
                f"Code review failed: {str(e)[:300]}. "
                "If you are running a local LLM (Ollama, LM Studio, "
                "LocalAI), set LLM_BACKEND=ollama and ensure it is "
                "running. Suggestions: "
                + "; ".join(get_error_suggestions(str(e))[:2])
            )
        )

# ─── Run ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )