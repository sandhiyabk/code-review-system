# ui/app.py
"""
Self-contained Streamlit UI for the AI Code Review Assistant.

This version runs the review pipeline DIRECTLY (imports core.pipeline)
and does NOT depend on a separately-running FastAPI backend. This makes
it deployable as a standalone Streamlit app (Streamlit Cloud / HuggingFace
Space) without needing a second service.

It also optionally integrates the two new additive components:
  - GitHub file input (ui.components.github_input)
  - RAGAS review-quality evaluation (ui.components.evaluation_display)

All optional integrations degrade gracefully if their dependencies
(RAGAS, etc.) are unavailable. The core review always works.
"""

import sys
import os

# Fix module path — allows importing core from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ─── Core pipeline (direct import — no HTTP backend needed) ───
from core.pipeline import review_code
from core.chroma_store import StyleRuleStore
from core.llm_client import get_llm_client

# ─── Optional additive components (never required) ────────────
HAS_GITHUB_UI = False
HAS_EVAL_UI = False
try:
    from ui.components.github_input import render_github_input
    HAS_GITHUB_UI = True
except ImportError:
    pass
try:
    from ui.components.evaluation_display import render_evaluation
    HAS_EVAL_UI = True
except ImportError:
    pass

st.set_page_config(
    page_title="AI Code Review Assistant",
    page_icon="🔍",
    layout="wide"
)


def render_llm_status():
    """
    Show which LLM backend is active in the sidebar.

    Helps users understand whether the cloud Groq backend or a local
    LLM (Ollama) is being used, and surfaces a conflict-free setup
    message if no backend is configured.
    """
    try:
        client = get_llm_client()
        info = client.get_backend_info()

        if info["backend"] == "none":
            st.sidebar.error(
                "No LLM configured\n\n"
                "Set GROQ_API_KEY for cloud (free)\n"
                "or install Ollama for local use"
            )
        elif info["is_local"]:
            st.sidebar.success(
                f"Local LLM: {info['model']} via Ollama"
            )
        else:
            st.sidebar.success(
                f"Cloud LLM: {info['model']}\n"
                f"via {info['backend'].title()}"
            )
    except Exception:
        # Never break the UI if the status check fails
        pass


# Render backend status indicator in sidebar
render_llm_status()

# Cache the ChromaDB store so it's only initialized once per session
@st.cache_resource
def get_store():
    """Return a reusable ChromaDB style-rule store."""
    return StyleRuleStore()


# ─── Header ───────────────────────────────────────────
st.title("🔍 AI Code Review Assistant")
st.caption("Powered by RAG Pipeline + LLM + AST Validation | Built by a 2027 Batch Student")

st.divider()

# ─── Layout ───────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Your Code")

    # ── Optional GitHub input (additive) ──
    # Provides a "Paste code | GitHub URL" toggle. If the fetched
    # code is ready for review, it takes priority over the text box.
    if HAS_GITHUB_UI:
        gh_source = render_github_input()

    code = st.text_area(
        "Paste Python code here",
        height=300,
        placeholder="""def find_user(users, id):
    for i in range(len(users)):
        for j in range(len(users)):
            if users[i].id == id:
                return users[i]""",
        label_visibility="collapsed"
    )

    # If a GitHub file was successfully fetched and the user clicked
    # "Review This Code", use that content instead of the paste box.
    if HAS_GITHUB_UI and gh_source.get("fetch_success"):
        if gh_source.get("review_clicked"):
            code = gh_source["code"]

    # ── Compact status note (self-contained mode) ──
    st.caption("🟢 Running as standalone app (no backend service required)")

col_btn1, col_btn2 = st.columns([2, 1])

with col_btn1:
    review_btn = st.button(
        "🔍 Review Code",
        type="primary",
        use_container_width=True
    )

with col_btn2:
    clear_btn = st.button(
        "Clear",
        use_container_width=True
    )

if clear_btn:
    st.rerun()

# ─── Review Results ──────────────────────────────────
with col2:
    st.subheader("📊 Review Results")

    if review_btn:
        if not code.strip():
            st.warning("Please paste some Python code first")
        else:
            with st.spinner("🔄 Analyzing code..."):
                try:
                    # Run the pipeline DIRECTLY — no HTTP backend dependency
                    result = review_code(code)
                    st.session_state.result = result

                    # Store the retrieved rules so the evaluation
                    # component can score them (additive).
                    try:
                        store = get_store()
                        st.session_state.retrieved_rules = \
                            store.get_relevant_rules(code)
                    except Exception:
                        st.session_state.retrieved_rules = []

                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

    # Display results
    if "result" in st.session_state:
        result = st.session_state.result

        # Quality Score
        score = result.get("quality_score", 0)

        if score >= 8:
            score_color = "green"
            score_label = "Excellent"
        elif score >= 6:
            score_color = "orange"
            score_label = "Good"
        elif score >= 4:
            score_color = "orange"
            score_label = "Needs Work"
        else:
            score_color = "red"
            score_label = "Poor"

        st.markdown(f"### Quality Score: :{score_color}[{score}/10 — {score_label}]")
        st.progress(score / 10)

        st.divider()

        # Bugs
        bugs = result.get("bugs", [])
        if bugs and bugs[0] != "No bugs found":
            st.subheader("🐛 Bugs Found")
            for bug in bugs:
                if bug:
                    st.error(f"→ {bug}")
        else:
            st.success("✅ No bugs found")

        # Complexity
        complexity = result.get("complexity", {})
        if complexity:
            st.subheader("⏱ Complexity Analysis")
            c1, c2 = st.columns(2)
            c1.metric("Time Complexity", complexity.get("time", "N/A"))
            c2.metric("Space Complexity", complexity.get("space", "N/A"))
            explanation = complexity.get("explanation", "")
            if explanation:
                st.caption(f"💡 {explanation}")

        st.divider()

        # Suggestions
        suggestions = result.get("suggestions", [])
        if suggestions:
            st.subheader("💡 Suggestions")
            for i, suggestion in enumerate(suggestions, 1):
                if suggestion:
                    st.info(f"{i}. {suggestion}")

        # Improved Code
        improved = result.get("improved_code", "")
        if improved and improved != code:
            st.subheader("✨ Improved Version")
            st.code(improved, language="python")

            # Copy button hint
            st.caption("👆 Click top-right corner of code block to copy")

        # ── Optional RAGAS evaluation (additive, runs async) ──
        if HAS_EVAL_UI:
            render_evaluation(
                code_input=code,
                generated_review=result,
                retrieved_rules=st.session_state.get("retrieved_rules", []),
            )

    else:
        # Placeholder when no result yet
        st.info("👈 Paste your code and click Review to get started")

        st.markdown("""
        **What this tool analyzes:**
        - 🐛 Bugs and logical errors
        - ⏱ Time and space complexity
        - 💡 Style and best practice violations
        - ✨ Provides improved code version

        **Powered by:**
        - RAG Pipeline (ChromaDB)
        - LLM API (Groq cloud or local Ollama)
        - Python AST validation
        """)

# ─── Footer ───────────────────────────────────────────
st.divider()
st.caption(
    "Built by a 2027 batch CS student from Kamaraj College of Engineering | "
    "Part of AI Engineer journey | "
    "GitHub: [your-github-link]"
)
