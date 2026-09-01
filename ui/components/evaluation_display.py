# ui/components/evaluation_display.py
"""
Evaluation Display streamlit component.

After a code review is generated, this component automatically
evaluates its quality using the RAGAS-based CodeReviewEvaluator
and renders the results in a visually friendly format.

What it renders:
- Loading indicator while evaluating
- Gauge-style display for the overall quality score
- Four metric bars (faithfulness, relevancy, precision, recall)
- Color coding: green > 0.8, orange 0.6-0.8, red < 0.6
- Human-readable interpretation paragraph
- Expandable "What do these metrics mean?" section

This is an OPTIONAL component. The existing app.py continues to
work without importing it.
"""

import sys
import os
import time

# Ensure project root is on path so core/ modules can be imported
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from typing import Optional

import streamlit as st

# ──────────────────────────────────────────────────────────
# Lazy import of the evaluator — wrapped in try/except so the
# UI still loads if RAGAS isn't installed
# ──────────────────────────────────────────────────────────
try:
    from core.evaluator import CodeReviewEvaluator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False


def render_evaluation(
    code_input: str,
    generated_review: dict,
    retrieved_rules: list,
    auto_run: bool = True,
) -> Optional[dict]:
    """
    Render the evaluation display in Streamlit.

    This runs the RAGAS evaluation (async to the review display)
    and shows the results.

    Args:
        code_input: The original Python code that was reviewed
        generated_review: The review dict from the pipeline
        retrieved_rules: Style rules retrieved from ChromaDB
        auto_run: If True, automatically evaluate when rendered.
                  If False, only show a button to trigger evaluation.

    Returns:
        The evaluation result dict (or the fallback dict if
        evaluation failed). Returns None if evaluation is skipped.
    """
    # Guard: if evaluator not available, show a friendly note
    if not EVALUATOR_AVAILABLE:
        st.caption(
            "📊 Review quality evaluation unavailable — "
            "the RAGAS library is not installed."
        )
        return None

    st.divider()
    st.subheader("📊 Review Quality Evaluation")

    if not code_input or not generated_review:
        st.info("Evaluation will run once a review is generated.")
        return None

    # ── Optional manual trigger (if auto_run is False) ──
    if not auto_run:
        if st.button("📊 Evaluate Review Quality", type="secondary"):
            return _run_and_display(code_input, generated_review, retrieved_rules)
        return None

    # ── Auto-run evaluation with loading indicator ──
    with st.spinner("🧠 Evaluating review quality..."):
        result = _run_and_display(code_input, generated_review, retrieved_rules)
    return result


def _run_and_display(
    code_input: str,
    generated_review: dict,
    retrieved_rules: list,
) -> dict:
    """
    Run the evaluation and render the display.
    """
    # Instantiate (or reuse) the evaluator
    if "evaluator" not in st.session_state:
        with st.spinner("Initializing evaluation model..."):
            st.session_state["evaluator"] = CodeReviewEvaluator()

    evaluator = st.session_state["evaluator"]

    # Run evaluation — this is the core call
    # Wrapped so any unexpected error still shows a message
    try:
        result = evaluator.evaluate_review(
            code_input=code_input,
            generated_review=generated_review,
            retrieved_rules=retrieved_rules,
        )
    except Exception as e:
        # Fallback display for unexpected errors
        result = {
            "is_evaluated": False,
            "error": f"Evaluation error: {str(e)}",
            "overall_quality": None,
            "quality_label": "Not evaluated",
        }

    # ── Render the results ──
    if not result.get("is_evaluated", False):
        # Graceful fallback display
        st.warning(
            f"⚠️ {result.get('error', 'Evaluation temporarily unavailable')}"
        )
        st.caption(
            "The code review above was generated successfully — "
            "only the quality evaluation is unavailable."
        )
        return result

    # ── Render successful evaluation ──
    _render_scores(result)
    return result


def _render_scores(result: dict) -> None:
    """
    Render the evaluation metric scores with visual styling.
    """
    overall = result.get("overall_quality", 0)
    label = result.get("quality_label", "")

    # ── Gauge-style overall score ──
    overall_pct = int(overall * 100)
    overall_color = _score_color(overall)

    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 1rem;
            border-radius: 10px;
            background: {overall_color};
            color: white;
            font-weight: bold;
        ">
            <div style="font-size: 3rem;">{overall_pct}</div>
            <div style="font-size: 1.2rem;">Overall Quality</div>
            <div style="font-size: 1rem;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Overview of all metrics ──
    st.markdown("### Metric Scores")

    # Define the metrics to display
    metrics = [
        ("Faithfulness", result.get("faithfulness", 0)),
        ("Answer Relevancy", result.get("answer_relevancy", 0)),
        ("Context Precision", result.get("context_precision", 0)),
        ("Context Recall", result.get("context_recall", 0)),
    ]

    for name, score in metrics:
        _render_metric_bar(name, score)

    # ── Timing info ──
    eval_ms = result.get("evaluation_time_ms", 0)
    if eval_ms:
        st.caption(f"⏱ Evaluation completed in {eval_ms/1000:.1f}s")

    # ── Interpretation paragraph ──
    interpretation = result.get("interpretation", "")
    if interpretation:
        st.markdown("### 💬 What This Means")
        st.write(interpretation)

    # ── Expandable educational section ──
    _render_metric_explanations()


def _render_metric_bar(name: str, score: float) -> None:
    """
    Render a single metric as a colored bar with a label.
    """
    pct = max(0.0, min(1.0, score))  # Clamp to 0-1
    color = _score_color(pct)

    # Create a colored HTML progress bar for better visuals
    bar_html = f"""
    <div style="display: flex; align-items: center; margin-bottom: 8px;">
        <div style="flex: 0 0 150px; font-weight: 500;">{name}</div>
        <div style="flex: 1; background: #eee; border-radius: 5px; height: 20px; margin: 0 10px;">
            <div style="
                width: {int(pct*100)}%;
                height: 100%;
                background: {color};
                border-radius: 5px;
            "></div>
        </div>
        <div style="flex: 0 0 80px; text-align: right; font-family: monospace;">
            {score:.2f}
        </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


def _score_color(score: float) -> str:
    """
    Map a score to a color:
    - Green for good (>0.8)
    - Orange for okay (0.6-0.8)
    - Red for needs work (<0.6)
    """
    if score >= 0.8:
        return "#28a745"  # green
    elif score >= 0.6:
        return "#fd7e14"  # orange
    else:
        return "#dc3545"  # red


def _render_metric_explanations() -> None:
    """
    Render an expandable educational section explaining each metric
    in simple terms for non-experts.
    """
    with st.expander("❓ What do these metrics mean?"):
        st.markdown(
            """
            Here's what each score measures in plain English:

            | Metric | What it measures | Good score looks like |
            |--------|-----------------|----------------------|
            | **Faithfulness** | Are the review suggestions actually grounded in your coding standards? Did the AI make up advice that isn't supported? | High = the AI only suggested things backed by your rules. Low = the AI invented rules or advice. |
            | **Answer Relevancy** | Does the review actually address YOUR specific code, or is it generic advice? | High = the review mentions your functions, patterns, and specific issues. Low = the review could apply to any code. |
            | **Context Precision** | Were the style rules retrieved from the database actually relevant to this code? | High = the retrieved rules are specific to what you wrote. Low = irrelevant or generic rules were pulled in. |
            | **Context Recall** | Did we retrieve ALL the relevant rules needed for this review? | High = no important rules were missed. Low = some rules that should have been used weren't found. |

            ### How to read the overall score:
            - **80-100**: The review is reliable and well-grounded.
            - **60-79**: The review is generally good with minor gaps.
            - **40-59**: The review is okay but has meaningful issues.
            - **0-39**: The review needs significant improvement.

            ### Why this matters:
            - A high **faithfulness** means the AI is not hallucinating — a critical safety feature.
            - High **relevancy** means the advice is actionable and specific to your code.
            - High **precision/recall** means the retrieval part of the pipeline is working well.
            """
        )
