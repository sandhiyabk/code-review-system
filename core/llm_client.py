# core/llm_client.py
"""
Unified LLM client that abstracts multiple backends.

Purpose:
    The existing codebase calls the Groq API directly via `from groq import Groq`
    and `client = Groq(api_key=os.getenv("GROQ_API_KEY"))`. That meant:
      - The ONLY supported backend was Groq.
      - If a user ran an LLM locally (Ollama, LM Studio, LocalAI) the app
        would still try Groq and show a confusing raw API error.

    This module introduces a single abstraction — `get_llm_client()` — that
    supports, in order of priority:
        1. Groq API                  (default cloud backend)
        2. Ollama                    (local, OpenAI-compatible)
        3. OpenAI                    (cloud)
        4. OpenAI-compatible endpoint(base_url override, e.g. LM Studio / LocalAI)

    All other modules call `get_llm_client()` and never need to know which
    backend is actually being used. This preserves the existing Groq flow
    while making local LLMs a first-class (and optional) addition.
"""

import os
from dotenv import load_dotenv

# Load env vars so GROQ_API_KEY / OLLAMA_HOST / OPENAI_* are read the same
# way the rest of the project already does (see core/llm_reviewer.py).
load_dotenv()


def detect_ollama_running(host: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama is running locally.

    Ollama exposes `/api/tags` when the server is up. We do a quick
    2-second-check so the app never hangs waiting on an offline host.

    Returns True if running, False otherwise. Never raises — a failed
    network call just means "Ollama is not running" (safe default).
    """
    import requests
    try:
        response = requests.get(f"{host}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        # Offline host / refused connection / timeout — treat as not running
        return False


def get_available_ollama_models(host: str = "http://localhost:11434") -> list:
    """
    List the models installed in Ollama.

    Returns a list of model-name strings, or an empty list if Ollama is
    not running / returns an error. This is used to show a helpful message
    when the user requests a model that isn't pulled yet.
    """
    import requests
    try:
        response = requests.get(f"{host}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        return []
    except Exception:
        return []


class LLMClient:
    """
    Unified LLM client that supports multiple backends.

    Backend auto-detection priority (override with LLM_BACKEND env var):
        1. GROQ_API_KEY set        → groq     (cloud, default)
        2. Ollama detected locally → ollama   (local)
        3. OPENAI_API_KEY set      → openai   (cloud)
        4. OPENAI_BASE_URL set     → custom   (any OpenAI-compatible server)

    Explicit override env vars:
        LLM_BACKEND  = groq | ollama | openai | custom
        LLM_MODEL    = any-model-name   (overrides the per-backend default)
        OLLAMA_HOST  = http://host:port (default http://localhost:11434)
    """

    def __init__(self):
        """Detect the backend, build the matching client, pick a model."""
        self.backend = self._detect_backend()
        self.client = self._initialize_client()
        self.default_model = self._get_default_model()

    def _detect_backend(self) -> str:
        """Auto-detect which LLM backend to use.

        Priority:
        - An explicit LLM_BACKEND env var always wins (user override).
        - Otherwise we prefer the cloud Groq backend when a key exists,
          then fall back to a locally-running Ollama, then OpenAI, then
          a custom OpenAI-compatible endpoint, and finally "none".
        """
        # Explicit override takes priority
        explicit = os.getenv("LLM_BACKEND", "").lower()
        if explicit in ["groq", "ollama", "openai", "custom"]:
            return explicit

        # Auto-detect based on available credentials / running services
        if os.getenv("GROQ_API_KEY"):
            # Preserve the existing Groq-first behavior — the deployed app
            # has GROQ_API_KEY set, so nothing changes for existing users.
            return "groq"

        # Only probe Ollama when we don't have a Groq key, so a slow/offline
        # Ollama host never delays the normal Groq path.
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if detect_ollama_running(ollama_host):
            return "ollama"

        if os.getenv("OPENAI_API_KEY"):
            return "openai"

        if os.getenv("OPENAI_BASE_URL"):
            return "custom"

        # Nothing available — "none" backend surfaces a helpful message later
        return "none"

    def _initialize_client(self):
        """Create the client object matching the detected backend."""
        if self.backend == "groq":
            from groq import Groq
            return Groq(api_key=os.getenv("GROQ_API_KEY"))

        elif self.backend == "ollama":
            # Ollama exposes an OpenAI-compatible API at <host>/v1, so we
            # reuse the official OpenAI client. "ollama" is a dummy key —
            # Ollama ignores auth, so any non-empty value works.
            from openai import OpenAI
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            return OpenAI(
                base_url=f"{host}/v1",
                api_key="ollama"
            )

        elif self.backend == "openai":
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        elif self.backend == "custom":
            # LM Studio / LocalAI / any OpenAI-compatible local server.
            from openai import OpenAI
            return OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY", "custom")
            )

        else:
            # "none" backend — no client. complete() will raise a helpful
            # error telling the user how to configure an LLM.
            return None

    def _get_default_model(self) -> str:
        """Pick the default model for the active backend.

        LLM_MODEL env var always overrides. Otherwise we use the same model
        the app currently sends to Groq (openai/gpt-oss-20b) so the existing
        deployment behaves IDENTICALLY after this change.
        """
        # Allow explicit override
        if os.getenv("LLM_MODEL"):
            return os.getenv("LLM_MODEL")

        defaults = {
            # Keep the deployed Groq model unchanged (STRICT RULE: don't
            # break the existing Groq flow). Change via LLM_MODEL if needed.
            "groq": "openai/gpt-oss-20b",
            # Most common local model; user can set LLM_MODEL instead.
            "ollama": "llama3.2",
            "openai": "gpt-4o-mini",
            "custom": "llama3.2",
            "none": None,
        }
        return defaults.get(self.backend, "llama3.2")

    def complete(
        self,
        messages: list,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> str:
        """Send a chat completion and return the text content.

        Raises RuntimeError with a user-friendly message on failure, so the
        caller can display it directly without showing a raw traceback.

        Ollama-specific: if the requested model isn't pulled, we list the
        installed models and the exact `ollama pull` command to run.
        """
        # Guard against having no configured backend at all
        if self.client is None or self.backend == "none":
            raise RuntimeError(
                "No LLM backend is configured. Please set one of:\n"
                "1. GROQ_API_KEY   (free cloud key at console.groq.com)\n"
                "2. Ollama         (local — install from ollama.ai)\n"
                "3. OPENAI_API_KEY (OpenAI cloud)\n"
                "4. OPENAI_BASE_URL (custom OpenAI-compatible endpoint)"
            )

        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e)

            # Ollama-specific: the named model isn't installed locally.
            # Show installed models + the fix command instead of raw text.
            if self.backend == "ollama" and "not found" in error_str:
                available = get_available_ollama_models()
                models_str = ", ".join(available) if available else "none installed"
                raise RuntimeError(
                    f"Ollama model '{self.default_model}' is not installed.\n"
                    f"Installed models: {models_str}\n"
                    f"Fix: run 'ollama pull {self.default_model}'"
                ) from e

            # Generic error — keep the raw detail up to 300 chars so the
            # user can diagnose, but never expose it as a raw traceback.
            raise RuntimeError(
                f"LLM error ({self.backend}): {error_str[:300]}"
            ) from e

    def get_backend_info(self) -> dict:
        """Return info about the active backend for display in the UI."""
        return {
            "backend": self.backend,
            "model": self.default_model,
            "is_local": self.backend == "ollama",
            "is_cloud": self.backend in ["groq", "openai"],
        }


# ──────────────────────────────────────────────────────────
# Singleton accessor
# ──────────────────────────────────────────────────────────
_client_instance = None


def get_llm_client() -> LLMClient:
    """Return the singleton LLMClient, creating it on first use.

    A singleton is used so the client (and any backend-detection network
    probes) is created exactly once per process. Streamlit re-runs the
    script per interaction, so this also keeps repeated calls cheap.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance


# ──────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    llm = get_llm_client()
    print("Active backend:", llm.backend)
    print("Default model :", llm.default_model)
    try:
        text = llm.complete(
            messages=[{"role": "user", "content": "Say hello in 3 words"}],
            max_tokens=50,
        )
        print("Response     :", text)
    except RuntimeError as e:
        print("No LLM available:\n", e)