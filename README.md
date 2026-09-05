---
title: AI Code Review Assistant
emoji: 🏃
colorFrom: pink
colorTo: indigo
sdk: streamlit
sdk_version: 1.58.0
python_version: '3.13'
app_file: ui/app.py
pinned: false
short_description: Building trustworthy AI code reviews by combining static ana
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---

## 🔬 New Features (Optional / Additive)

This project now includes three new **optional, additive** features. They
do **not** modify or break any existing code — the original review flow
works exactly as before. New features are imported only if you choose to
use them, and they degrade gracefully if a dependency is missing.

### 1. 📊 RAGAS Evaluation Pipeline (`core/evaluator.py`)

After the review is generated, this optional module measures the **quality**
of the review (not just whether it ran) using 4 RAGAS metrics:

| Metric | Weight | What it measures |
|--------|--------|------------------|
| **Faithfulness** | 35% | Are suggestions grounded in retrieved style rules? (anti-hallucination) |
| **Answer Relevancy** | 30% | Does the review address your actual code? |
| **Context Precision** | 20% | Were the retrieved rules actually relevant? |
| **Context Recall** | 15% | Were all needed rules retrieved? |

- Produces an **overall quality score** (0–1), a **quality label**, and a
  **plain-English interpretation** that flags the weakest metric.
- Uses the same **Groq API** (LLaMA 3.3-70B) already configured — no new keys.
- **Async to the pipeline**: the review shows first, evaluation runs after.
- **Graceful fallback**: if RAGAS fails, returns `is_evaluated: False` and
  the review still works normally.
- **Caching**: identical code+review pairs are not re-evaluated.

**Usage:**
```python
from core.evaluator import CodeReviewEvaluator

evaluator = CodeReviewEvaluator()
result = evaluator.evaluate_review(
    code_input=code,
    generated_review=review,      # dict from the pipeline
    retrieved_rules=relevant_rules,  # list from ChromaDB
)
# result["overall_quality"], result["quality_label"], result["interpretation"]
```

### 2. 🔗 GitHub Integration (`core/github_integration.py`)

Review a file directly from a **public GitHub repository** instead of
pasting code. Uses the **GitHub REST API** (Contents API), NOT raw URLs.

Can submit either:
- A full file URL: `https://github.com/user/repo/blob/main/file.py`
- Or a repo + file path + branch

**Security guarantees:**
- Only accepts `github.com` URLs (rejects raw/gitlab/bitbucket).
- Sanitizes paths against traversal (`../`, `~/`, `//`, null bytes).
- Never logs file content (may contain secrets).
- Optional `GITHUB_TOKEN` env var for 5000 req/hr (vs 60 unauthenticated).
- Enforces a 100KB file-size limit and an extension whitelist
  (`.py .js .ts .java .cpp .c .go .rs`).

**Usage:**
```python
from core.github_integration import GitHubIntegration

gh = GitHubIntegration()
file_data = gh.fetch_from_url("https://github.com/user/repo/blob/main/file.py")
# file_data["content"], file_data["file_name"], file_data["language"], ...
status = gh.get_rate_limit_status()  # {"remaining": 45, "limit": 60, ...}
```

### 3. 🖥 Streamlit UI Components (`ui/components/`)

Two importable components to drop into `ui/app.py` with minimal changes:

- **`github_input.py`** → `render_github_input()`: adds a
  "Paste code | GitHub URL" radio, fetch button, rate-limit status,
  and fetched-code verification.
- **`evaluation_display.py`** → `render_evaluation()`: gauge-style overall
  score, 4 colored metric bars, interpretation, and an educational
  "What do these metrics mean?" expander.

Both are **optional** — `ui/app.py` works unmodified without them.

### 🧪 Running the Tests

Install pytest first:
```bash
pip install pytest
```

Then run:
```bash
# All new tests
python -m pytest tests/test_github_integration.py tests/test_evaluator.py -v

# Only GitHub tests (fully mocked — no network)
python -m pytest tests/test_github_integration.py -v
```

**Note:** The RAGAS evaluation tests are split into:
- Tests that never require RAGAS (labels, interpretation, fallback, weights)
  — always run.
- Tests that require the `ragas` + `datasets` libraries — these are
  **automatically skipped** if RAGAS isn't installed, so the suite never
  fails due to a missing optional dependency.

## 🌐 LLM Configuration

The app supports **multiple LLM backends**. It auto-detects the backend to
use, and you can force a specific one with the `LLM_BACKEND` environment
variable. **The Groq cloud backend remains the default and requires no
changes** for existing deployments.

### Auto-detection order

| Priority | Backend | Detection trigger |
|----------|---------|-------------------|
| 1 | **Groq** (cloud, default) | `GROQ_API_KEY` is set |
| 2 | **Ollama** (local) | `OLLAMA_HOST` reachable *and* no Groq key |
| 3 | **OpenAI** (cloud) | `OPENAI_API_KEY` is set |
| 4 | **Custom** (OpenAI-compatible, e.g. LM Studio / LocalAI) | `OPENAI_BASE_URL` is set |

> Explicit override always wins: `LLM_BACKEND=groq|ollama|openai|custom`
> Force a specific model with `LLM_MODEL`.

### Option 1: Groq API (Recommended — Free)

1. Get a free API key at https://console.groq.com
2. Set it in your environment or `.env` file:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. (Optional) Override the model:
   ```
   LLM_MODEL=openai/gpt-oss-20b
   ```

### Option 2: Ollama (Local — no API key needed)

1. Install Ollama from https://ollama.ai and start it (`ollama serve`)
2. Pull the default model:
   ```
   ollama pull llama3.2
   ```
3. Configure the app:
   ```
   LLM_BACKEND=ollama
   # optional: OLLAMA_HOST=http://localhost:11434
   # optional: LLM_MODEL=llama3.2
   ```
The review pipeline uses Ollama's OpenAI-compatible endpoint automatically.

### Option 3: OpenAI (Cloud)

```
OPENAI_API_KEY=your_key_here
# optional: LLM_MODEL=gpt-4o-mini
```

### Option 4: Custom OpenAI-compatible endpoint (LM Studio / LocalAI, etc.)

```
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=anything   # must be non-empty for the client
LLM_MODEL=your-local-model
```

### Error handling

All LLM errors are translated into **friendly, actionable messages** with
suggestions — raw API errors are never shown directly to the user. If you
see an error, check the sidebar status indicator to confirm which backend
is active.

### ⚙ Requirements & Environment

Added to `requirements.txt` (existing dependencies untouched):
```
ragas>=0.1.0
datasets>=2.0.0
openai   # required for Ollama / OpenAI-compatible local backends
```

Added to `.env.example`:
```
GITHUB_TOKEN=optional_for_higher_rate_limits
LLM_BACKEND=ollama               # optional: force a backend
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=llama3.2               # optional: override default model
OPENAI_API_KEY=...
OPENAI_BASE_URL=http://localhost:1234/v1
```

### File Tree (New Files Only)

```
core/evaluator.py                 # RAGAS evaluation pipeline
core/github_integration.py        # GitHub REST API integration
core/llm_client.py                # Unified LLM client (Groq/Ollama/OpenAI/custom)
ui/components/__init__.py
ui/components/github_input.py     # Streamlit GitHub input component
ui/components/evaluation_display.py  # Streamlit evaluation component
tests/test_evaluator.py           # Evaluator tests
tests/test_github_integration.py  # GitHub tests (mocked, no network)
.env.example                      # Added GITHUB_TOKEN placeholder
```

**Modified additively** (existing flows preserved):
- `core/llm_reviewer.py` — graceful, categorized error messages + shared client
- `core/pipeline.py` — skips AST validation on LLM error results
- `core/evaluator.py` — uses the shared LLM client (Groq default preserved)
- `ui/app.py` — sidebar LLM-backend status indicator
- `api/main.py` — helpful error messages instead of raw tracebacks
- `requirements.txt` — added `openai` (optional, for local backends)
