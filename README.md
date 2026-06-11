# CodeReview AI

**Multi-layer AI code review assistant powered by AST parsing, RAG, and LLM inference.**

CodeReview AI analyzes Python source code through a five-stage pipeline — static analysis, semantic rule retrieval, prompt enrichment, LLM review, and hallucination validation — to surface bugs, complexity concerns, style violations, and actionable refactoring suggestions with a quality score and an improved code diff.

---

## Pipeline Architecture

```
  Source Code
       │
       ▼
┌────────────────┐
│  1. AST Parse   │  Detect functions, loops, nesting, error handling, imports
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ 2. RAG Retrieve│  Semantic search of 26 Python style rules via ChromaDB
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ 3. Prompt      │  Enrich prompt with AST facts + retrieved rules
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ 4. LLM Review  │  Groq Llama 3.3-70B → structured JSON output
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ 5. Validate    │  Hallucination filter + schema enforcement
└───────┬────────┘
        │
        ▼
  Bugs · Complexity · Suggestions · Score · Improved Code
```

---

## Features

- **AST-level static analysis** — extracts functions, nested loops, error handling, and imports without running the code
- **RAG-enhanced reviews** — semantically retrieves relevant style rules from a vector store to ground the LLM response
- **LLM-powered review** — uses Groq's Llama 3.3-70B for deep code understanding
- **Hallucination validation** — cross-checks LLM suggestions against actual AST functions to eliminate fabricated advice
- **Quality scoring** — 1–10 score with actionable improvement suggestions
- **Improved code output** — receives a rewritten version of the code incorporating the review feedback
- **Dual interface** — FastAPI REST API + Streamlit web UI

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Static Analysis | Python `ast` module |
| Vector Store | ChromaDB (cosine similarity, semantic retrieval) |
| LLM Inference | Groq API — `llama-3.3-70b-versatile` |
| API Framework | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Validation | Custom hallucination filter + Pydantic schemas |
| Environment | Python 3.13, `python-dotenv` |

---

## Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/code-review-ai.git
cd code-review-ai

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Configure your API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

### Configuration

Set the following environment variable in `.env` (already git-ignored):

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key for Llama 3.3-70B access |

---

## Usage

### Start the API server

```bash
python -m api.main
```

The server starts at `http://localhost:8000` with interactive docs at `/docs`.

### Start the Streamlit UI

```bash
streamlit run ui/app.py
```

Opens in your browser at `http://localhost:5173`.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info and version |
| `GET` | `/health` | Health check with model and vector store status |
| `POST` | `/review` | Submit code for review |

### Review a code snippet

```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def calc(a,b):\n  for i in range(len(a)):\n    for j in range(len(b)):\n      print(a[i],b[j])",
    "language": "python"
  }'
```

#### Response

```json
{
  "bugs": [
    "Nested loop over both arrays — O(n*m) complexity risk with large inputs"
  ],
  "complexity": {
    "time": "O(n * m)",
    "space": "O(1)",
    "explanation": "Two nested loops iterate over the full length of both input lists"
  },
  "suggestions": [
    "Use enumerate() instead of range(len()) when you need both index and value",
    "Extract inner loop into a helper function for readability"
  ],
  "quality_score": 4,
  "improved_code": "def calc(a, b):\n    for i, val_a in enumerate(a):\n        for j, val_b in enumerate(b):\n            print(val_a, val_b)"
}
```

---

## Project Structure

```
code-review-ai/
├── api/
│   └── main.py              FastAPI server — routes, models, error handling
├── core/
│   ├── ast_analyzer.py      Python AST structural analysis
│   ├── chroma_store.py      ChromaDB vector store — rule embedding + retrieval
│   ├── llm_reviewer.py      Groq LLM client — prompt submission + JSON parsing
│   ├── pipeline.py          Pipeline orchestrator — chains all 5 layers
│   ├── prompt_builder.py    Enriched prompt assembly from AST + RAG context
│   └── validator.py         Hallucination filter + output schema enforcement
├── data/
│   └── style_rules.py       26 static Python style rules for RAG retrieval
├── tests/
│   └── test_samples.py      6 sample code tests covering all pipeline paths
├── ui/
│   └── app.py               Streamlit frontend — code input + results display
├── .env                     API key (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How It Works

### Layer 1 — AST Analysis
Parses the submitted code into a syntax tree. Extracts function definitions, import statements, loop counts, nested loop depth, and try/except presence. Returns structured metadata for the prompt and a validity flag.

### Layer 2 — RAG Retrieval
Sentence-level style rules (26 total, covering PEP8, performance, error handling, naming, and structure) are embedded into a ChromaDB collection. On each review, the raw code is used as a query to retrieve the top-4 semantically similar rules (cosine distance < 0.7).

### Layer 3 — Prompt Assembly
Concatenates the AST metadata, retrieved style rules, and original code into a structured prompt. The prompt explicitly constrains the LLM to reference only functions that actually exist in the code.

### Layer 4 — LLM Review
Sends the assembled prompt to `llama-3.3-70b-versatile` via Groq with `temperature=0.1` for deterministic JSON output. Extracts and parses the JSON response.

### Layer 5 — Validation
Scans each suggestion for hallucinated function references (e.g., suggesting a call to a function that doesn't exist in the AST). Strips invalid suggestions. Falls back to safe defaults if the entire response is malformed or the API call fails.

---

## Testing

Run the test suite to validate the full pipeline against six representative code samples:

```bash
python tests/test_samples.py
```

Run only samples matching a pattern:

```bash
python tests/test_samples.py --sample nested
```

| Sample | Code Quality | Expected Score |
|--------|-------------|----------------|
| `nested_loops` | Deeply nested loops | 1–5 |
| `no_error_handling` | Missing try/except | 3–7 |
| `bad_naming` | Single-letter variables | 2–6 |
| `good_code` | Well-structured code | 7–10 |
| `multiple_issues` | Many violations | 1–4 |
| `syntax_error` | Invalid Python | 0–2 |

---

## Limitations

- **Language support** — currently Python-only (validated at the API layer)
- **Code length** — capped at 5,000 characters per request
- **Static rules** — the 26 style rules are hand-written; no auto-discovery of project-specific conventions
- **No persistent storage** — ChromaDB is in-memory; rules are reloaded on every server start
- **No authentication** — the API has no auth layer; intended for local/trusted-network use

---

## License

MIT
