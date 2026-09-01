# ui/app.py

import sys
import os

# Fix module path — allows importing core from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import json

st.set_page_config(
    page_title="AI Code Review Assistant",
    page_icon="🔍",
    layout="wide"
)

# ─── Header ───────────────────────────────────────────
st.title("🔍 AI Code Review Assistant")
st.caption("Powered by RAG Pipeline + LLM + AST Validation | Built by a 2027 Batch Student")

st.divider()

# ─── Layout ───────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Your Code")

    code = st.text_area(
        "Paste Python code here",
        height=320,
        placeholder="""def find_user(users, id):
    for i in range(len(users)):
        for j in range(len(users)):
            if users[i].id == id:
                return users[i]""",
        label_visibility="collapsed"
    )

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

    # API status check
    try:
        health = requests.get("http://localhost:8000/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.warning("⚠️ FastAPI not running — start it first")
        st.code("uvicorn api.main:app --reload", language="bash")

with col2:
    st.subheader("📊 Review Results")

    if review_btn:
        if not code.strip():
            st.warning("Please paste some Python code first")
        else:
            with st.spinner("🔄 Analyzing code..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/review",
                        json={"code": code, "language": "python"},
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.result = result
                    else:
                        st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API")
                    st.info("Run this in a separate terminal first:")
                    st.code("uvicorn api.main:app --reload", language="bash")

                except requests.exceptions.Timeout:
                    st.error("Request timed out — LLM taking too long")

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
        - Llama3.3 via Groq API
        - Python AST validation
        """)

# ─── Footer ───────────────────────────────────────────
st.divider()
st.caption(
    "Built by a 2027 batch CS student from Kamaraj College of Engineering | "
    "Part of AI Engineer journey | "
    "GitHub: [your-github-link]"
)