# core/llm_reviewer.py

import json
import re
from dotenv import load_dotenv

# New unified LLM client — supports Groq (default), Ollama (local),
# OpenAI, and any OpenAI-compatible endpoint. The rest of this module
# only knows about get_llm_client(); it never cares which backend runs.
from core.llm_client import get_llm_client

load_dotenv()


def classify_error(error_str: str) -> str:
    """
    Classify an error message into a user-friendly category.

    The category drives which help content the UI shows, so users get
    targeted advice instead of a raw API error string.
    """
    error_lower = error_str.lower()
    if "api_key" in error_lower or "auth" in error_lower:
        return "authentication"
    elif "rate" in error_lower or "limit" in error_lower:
        return "rate_limit"
    elif "timeout" in error_lower or "connect" in error_lower \
            or "refused" in error_lower:
        return "connection"
    elif "local" in error_lower or "ollama" in error_lower:
        return "local_llm"
    else:
        return "unknown"


def get_error_suggestions(error_str: str) -> list:
    """
    Return actionable suggestions based on the error category.

    These are shown to the user below the error message so the failure
    always comes with a concrete next step.
    """
    error_type = classify_error(error_str)

    suggestions = {
        "authentication": [
            "Set GROQ_API_KEY in your environment variables",
            "Get a free API key at console.groq.com",
            "If using Streamlit Cloud, add the key in Secrets settings",
        ],
        "rate_limit": [
            "Wait 30–60 seconds before trying again",
            "Reduce the code snippet size to lower token usage",
            "Groq free tier allows a limited number of requests per minute",
        ],
        "connection": [
            "Check your internet connection",
            "The LLM API may be temporarily down",
            "Try again in a few minutes",
        ],
        "local_llm": [
            "Set LLM_BACKEND=ollama if you run Ollama locally",
            "Make sure Ollama is running (ollama serve) on localhost",
            "Pull the default model: ollama pull llama3.2",
        ],
        "unknown": [
            "Try refreshing the page and submitting again",
            "If the problem persists, try a smaller code snippet",
            "Run the review again after a short wait",
        ],
    }
    return suggestions.get(error_type, suggestions["unknown"])


def get_llm_review(prompt: str) -> dict:
    """
    Sends enriched prompt to the active LLM backend.
    Returns structured review as dict.

    On any LLM failure this returns the review-shaped dict with a
    USER-FRIENDLY message (plus error_type + suggestions) instead of
    surfacing a raw API error, so the UI can render it cleanly.
    """
    try:
        llm = get_llm_client()
        raw = llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert Python code reviewer.
You MUST respond with ONLY valid JSON.
No explanation before or after JSON.
No markdown code blocks.
Just raw JSON.

Required format:
{
    "bugs": ["specific bug description"],
    "complexity": {
        "time": "O(n)",
        "space": "O(1)",
        "explanation": "brief explanation"
    },
    "suggestions": ["specific actionable suggestion"],
    "quality_score": 7,
    "improved_code": "the improved version of the code"
}""",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
            max_tokens=3000,
        )

        # Clean any markdown if present
        clean = re.sub(r"```json|```", "", raw).strip()

        return json.loads(clean)

    except json.JSONDecodeError:
        # Structured fallback if the LLM didn't return parseable JSON.
        # `raw` may be undefined if parsing failed before assignment, so
        # guard with locals() to keep the message helpful either way.
        raw_preview = locals().get("raw", "empty response")[:500]
        return {
            "bugs": ["Could not parse the LLM review response"],
            "complexity": {
                "time": "Unknown",
                "space": "Unknown",
                "explanation": "Analysis failed",
            },
            "suggestions": [
                "Try submitting the code again — the model returned "
                "malformed JSON.",
                raw_preview,
            ],
            "quality_score": 0,
            "improved_code": "",
            # Mark the result as LLM-failed so the pipeline skips the
            # AST cross-validation (which is meaningless for error output).
            "llm_error": True,
            "error_type": classify_error("json"),
        }

    except RuntimeError as e:
        # Our LLMClient raises RuntimeError with a user-friendly message —
        # surface it directly, no raw traceback.
        return _build_error_result(str(e))

    except Exception as e:
        # Unexpected error — still catch it and convert to a helpful message
        # (never leak a raw traceback to the user).
        return _build_error_result(str(e))


def _build_error_result(error_str: str) -> dict:
    """
    Convert an LLM failure into a review-shaped dict with a helpful
    message, an error category, and actionable suggestions.
    """
    error_type = classify_error(error_str)
    suggestions = get_error_suggestions(error_str)

    return {
        "bugs": [
            f"LLM API error: {error_str[:200]}. "
            "If you are running a local LLM (Ollama, LM Studio, LocalAI), "
            "set LLM_BACKEND=ollama and make sure it is running."
        ],
        "complexity": {
            "time": "N/A",
            "space": "N/A",
            "explanation": "",
        },
        "suggestions": suggestions,
        "quality_score": 0,
        "improved_code": "",
        # Mark the result as LLM-failed so the pipeline skips validation.
        "llm_error": True,
        "error_type": error_type,
    }


# Test it directly
if __name__ == "__main__":
    test_prompt = """
Review this Python code:

def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i != j and arr[i] == arr[j]:
                duplicates.append(arr[i])
    return duplicates

Return ONLY valid JSON with keys:
bugs, complexity, suggestions, quality_score, improved_code
"""
    result = get_llm_review(test_prompt)
    print(json.dumps(result, indent=2))