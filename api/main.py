# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from core.pipeline import review_code
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
    return {
        "status": "healthy",
        "model": "llama-3.3-70b-versatile",
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
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ─── Run ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )