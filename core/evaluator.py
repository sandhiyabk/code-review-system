# core/evaluator.py
"""
LLM-based Quality Evaluation for Code Reviews.

This module measures the quality of an AI-generated code review (not just
whether it ran) using the same four metrics and weights as the original
RAGAS-style design:

| Metric            | Weight | What it measures |
|-------------------|--------|------------------|
| Faithfulness      | 35%    | Is the review's advice grounded in the retrieved style rules? |
| Answer Relevancy  | 30%    | Does the review address the submitted code specifically? |
| Context Precision | 20%    | Were the retrieved rules actually relevant to the code? |
| Context Recall    | 15%    | Were all needed rules retrieved? |

RAGAS metrics are LLM-judge metrics: an LLM compares the review against
the retrieved context and returns a score. This module performs exactly
that comparison using the app's *own* unified LLM backend (`get_llm_client`)
— the same one that produces the review. This means:

- No extra dependencies (no `ragas`/`datasets`/embeddings required).
- Works identically on every backend (Groq cloud, Ollama, OpenAI,
  or any OpenAI-compatible endpoint).
- Degrades gracefully: if no backend is configured or the LLM call fails,
  a structured fallback is returned and the main review still works.

The evaluation is designed to run ASYNC to the main review pipeline —
the review shows first, evaluation runs after to avoid blocking the user.
"""

import os
import re
import time
import hashlib
import json
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# ──────────────────────────────────────────────────────────
# Weighted scoring configuration
# Weights reflect importance of each metric for code review quality:
# - Faithfulness (35%): Most critical — prevents hallucinated advice
# - Answer Relevancy (30%): Review must address the actual code
# - Context Precision (20%): Retrieved rules must be relevant
# - Context Recall (15%): Must retrieve all needed rules
# ──────────────────────────────────────────────────────────
METRIC_WEIGHTS = {
    "faithfulness": 0.35,
    "answer_relevancy": 0.30,
    "context_precision": 0.20,
    "context_recall": 0.15,
}

# Simple in-memory cache to avoid re-evaluating identical inputs.
# Key: hash of (code + review_json), Value: evaluation result dict.
_EVAL_CACHE: Dict[str, dict] = {}


class CodeReviewEvaluator:
    """
    Evaluates code review quality using LLM-judged metrics.

    Uses the same unified LLM backend as the review pipeline (Groq by
    default, or a local/OpenAI-compatible endpoint if configured via
    LLM_BACKEND). This keeps evaluation consistent with the backend that
    produced the review.

    Usage:
        evaluator = CodeReviewEvaluator()
        result = evaluator.evaluate_review(
            code_input="def foo(): ...",
            generated_review={"bugs": [...], "suggestions": [...]},
            retrieved_rules=["rule 1", "rule 2"]
        )
    """

    def __init__(self):
        """
        Initialize the evaluator against the shared LLM client.

        If no LLM backend is configured, or initialization fails, all
        evaluations gracefully return the fallback result.
        """
        self._llm = None
        self._backend = None
        self._init_time_ms = 0

        try:
            start = time.time()

            # Reuse the unified LLM client so evaluation follows whichever
            # backend the user configured (Groq by default, or a local LLM).
            from core.llm_client import get_llm_client
            llm = get_llm_client()

            if llm.backend == "none" or llm.client is None:
                # No backend configured — evaluation will use the fallback
                print("[evaluator] No LLM backend configured — "
                      "evaluation disabled")
                return

            self._llm = llm
            self._backend = llm.backend
            self._init_time_ms = round((time.time() - start) * 1000)
            print(f"[evaluator] Initialized in {self._init_time_ms}ms "
                  f"(backend={llm.backend})")

        except Exception as e:
            # If initialization fails, evaluation will use fallback
            # (never blocks the main review and never shows a traceback)
            print(f"[evaluator] Failed to initialize: {e}")

    def _generate_cache_key(
        self,
        code_input: str,
        generated_review: dict
    ) -> str:
        """
        Generate a deterministic cache key from code + review.

        Uses SHA-256 hash to create a fixed-length key that
        uniquely identifies this specific code-review pair.
        This prevents re-evaluation of identical inputs.
        """
        # Serialize review to stable JSON (sorted keys, no whitespace)
        review_str = json.dumps(generated_review, sort_keys=True)
        combined = code_input.strip() + "|||" + review_str
        return hashlib.sha256(combined.encode()).hexdigest()

    def _review_to_text(self, review: dict) -> str:
        """
        Convert structured review dict to plain text for the LLM judge.

        The metrics compare the review-as-text against the retrieved
        rules and the submitted code, so we flatten the structured
        review into readable paragraphs first.
        """
        parts = []

        # Add bugs section
        bugs = review.get("bugs", [])
        if bugs:
            parts.append("Bugs found: " + "; ".join(str(b) for b in bugs))

        # Add suggestions section
        suggestions = review.get("suggestions", [])
        if suggestions:
            parts.append(
                "Suggestions: " + "; ".join(str(s) for s in suggestions)
            )

        # Add complexity info
        complexity = review.get("complexity", {})
        if complexity:
            parts.append(
                f"Complexity: time={complexity.get('time', 'N/A')}, "
                f"space={complexity.get('space', 'N/A')}. "
                f"{complexity.get('explanation', '')}"
            )

        # Add quality score
        score = review.get("quality_score", "N/A")
        parts.append(f"Quality score: {score}/10")

        # Add improved code if present
        improved = review.get("improved_code", "")
        if improved:
            parts.append(f"Improved code provided: {improved[:200]}...")

        return " | ".join(parts)

    def _parse_scores(self, raw: str) -> Optional[Dict[str, float]]:
        """
        Parse the LLM's JSON scoring response into clamped 0-1 scores.

        Returns a dict with one key per metric in METRIC_WEIGHTS, or
        None if the response cannot be parsed (the caller then falls
        back to the graceful fallback result).
        """
        try:
            # Clean any markdown code fences if present
            clean = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(clean)

            scores = {}
            for key in METRIC_WEIGHTS:
                value = float(data.get(key))
                # Clamp to a valid 0.0-1.0 range (defensive)
                scores[key] = max(0.0, min(1.0, value))
            return scores

        except Exception as e:
            print(f"[evaluator] Could not parse LLM scores: {e}")
            return None

    def _ask_for_scores(
        self,
        code_input: str,
        review_text: str,
        retrieved_rules: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Ask the configured LLM to rate the review on the four metrics.

        This is a single LLM call that returns a JSON object with four
        scores in the 0.0-1.0 range — mirroring what the RAGAS
        LLM-judge metrics compute, without any external dependencies.
        """
        rules_text = "; ".join(retrieved_rules) if retrieved_rules else (
            "None - no coding standards were retrieved for this code."
        )

        system_prompt = (
            "You are an expert evaluator of code-review quality. "
            "You rate reviews on four scales, each 0.0 to 1.0. "
            "You must respond with ONLY valid JSON and no other text."
        )

        user_prompt = (
            "Rate the quality of the following AI-generated code review.\n\n"
            "=== SUBMITTED CODE ===\n"
            f"{code_input[:2000]}\n\n"
            "=== RETRIEVED CODING STANDARDS (CONTEXT) ===\n"
            f"{rules_text}\n\n"
            "=== GENERATED REVIEW ===\n"
            f"{review_text[:600]}\n\n"
            "Rate the review on these four scales (0.0 to 1.0):\n"
            "- faithfulness: how much of the review's advice is directly "
            "supported by the retrieved coding standards? (treat the "
            "standards as the only source of truth)\n"
            "- answer_relevancy: how relevant is the review to the "
            "submitted code specifically, rather than generic advice?\n"
            "- context_precision: what fraction of the retrieved coding "
            "standards are actually relevant to this code?\n"
            "- context_recall: how well do the retrieved standards cover "
            "the important issues in this code? Use 0.5 if no standards "
            "were retrieved.\n\n"
            "Respond with ONLY valid JSON in exactly this shape:\n"
            '{"faithfulness":0.0,"answer_relevancy":0.0,'
            '"context_precision":0.0,"context_recall":0.0}'
        )

        try:
            raw = self._llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            return self._parse_scores(raw)

        except Exception as e:
            print(f"[evaluator] Score request failed: {e}")
            return None

    def evaluate_review(
        self,
        code_input: str,
        generated_review: dict,
        retrieved_rules: List[str]
    ) -> dict:
        """
        Run all 4 LLM-judged metrics and return comprehensive evaluation.

        This is the main public method. It:
        1. Checks the cache for previous evaluations
        2. Asks the LLM to rate the review on the 4 metrics
        3. Computes weighted overall quality score
        4. Generates human-readable interpretation
        5. Caches the result

        Args:
            code_input: The original Python code
            generated_review: The review output from the pipeline
            retrieved_rules: Style rules retrieved from ChromaDB

        Returns:
            dict with all metric scores, overall quality, label,
            interpretation, and metadata. Returns graceful fallback
            if evaluation fails for any reason.
        """
        # ── Cache check ──
        cache_key = self._generate_cache_key(code_input, generated_review)
        if cache_key in _EVAL_CACHE:
            print("[evaluator] Returning cached evaluation result")
            return _EVAL_CACHE[cache_key]

        # ── Pre-flight checks ──
        if self._llm is None:
            return self._fallback_result(
                "Evaluation LLM not initialized. Configure an LLM backend "
                "(GROQ_API_KEY for cloud, or LLM_BACKEND=ollama for local)."
            )

        start_time = time.time()

        try:
            # ── Flatten the review into text for the LLM judge ──
            review_text = self._review_to_text(generated_review)

            # ── Ask the LLM to rate the review ──
            scores = self._ask_for_scores(
                code_input=code_input,
                review_text=review_text,
                retrieved_rules=retrieved_rules,
            )
            if scores is None:
                return self._fallback_result(
                    "The evaluation LLM did not return parseable scores. "
                    "Please try again."
                )

            # ── Calculate weighted overall quality ──
            overall = sum(
                scores[metric] * weight
                for metric, weight in METRIC_WEIGHTS.items()
            )
            overall = round(overall, 3)

            # ── Compute timing ──
            elapsed_ms = round((time.time() - start_time) * 1000)

            # ── Build final result ──
            evaluation_result = {
                "is_evaluated": True,
                "faithfulness": round(scores["faithfulness"], 3),
                "answer_relevancy": round(scores["answer_relevancy"], 3),
                "context_precision": round(scores["context_precision"], 3),
                "context_recall": round(scores["context_recall"], 3),
                "overall_quality": overall,
                "quality_label": self.get_quality_label(overall),
                "interpretation": self.get_interpretation(scores),
                "evaluation_time_ms": elapsed_ms,
            }

            # ── Cache the result ──
            _EVAL_CACHE[cache_key] = evaluation_result

            print(
                f"[evaluator] Evaluation complete in {elapsed_ms}ms — "
                f"overall: {overall} ({evaluation_result['quality_label']})"
            )

            return evaluation_result

        except Exception as e:
            # ── Graceful fallback on any error ──
            elapsed_ms = round((time.time() - start_time) * 1000)
            print(f"[evaluator] Evaluation failed after {elapsed_ms}ms: {e}")
            return self._fallback_result(str(e))

    def get_quality_label(self, score: float) -> str:
        """
        Map overall quality score to human-readable label.

        Scale:
        - 0.8 to 1.0 → "High Quality"
        - 0.6 to 0.8 → "Good Quality"
        - 0.4 to 0.6 → "Acceptable"
        - 0.0 to 0.4 → "Needs Review"
        """
        if score >= 0.8:
            return "High Quality"
        elif score >= 0.6:
            return "Good Quality"
        elif score >= 0.4:
            return "Acceptable"
        else:
            return "Needs Review"

    def get_interpretation(self, metrics: dict) -> str:
        """
        Generate a human-readable interpretation of evaluation scores.

        Identifies the lowest-scoring metric and explains what that
        means in plain English. Also highlights strong metrics.
        This helps non-technical users understand the evaluation.
        """
        if not metrics:
            return "No metrics available for interpretation."

        # Find the weakest metric (lowest score)
        weakest_metric = min(metrics, key=metrics.get)
        weakest_score = metrics[weakest_metric]

        # Find the strongest metric (highest score)
        strongest_metric = max(metrics, key=metrics.get)
        strongest_score = metrics[strongest_metric]

        # Map metric names to plain English descriptions
        metric_explanations = {
            "faithfulness": (
                "the review suggestions may not be fully grounded "
                "in the retrieved coding standards"
            ),
            "answer_relevancy": (
                "the review may not be specifically addressing "
                "the code you submitted"
            ),
            "context_precision": (
                "some retrieved coding standards may not be "
                "directly relevant to your code"
            ),
            "context_recall": (
                "some relevant coding standards may not have been "
                "retrieved for this review"
            ),
        }

        strong_explanations = {
            "faithfulness": (
                "the review suggestions are well-grounded in your "
                "coding standards"
            ),
            "answer_relevancy": (
                "the review is highly relevant to your specific code"
            ),
            "context_precision": (
                "the retrieved coding standards are precisely "
                "relevant to your code"
            ),
            "context_recall": (
                "all relevant coding standards were retrieved "
                "for this review"
            ),
        }

        # Build interpretation text
        parts = []

        # Start with overall assessment
        if weakest_score < 0.4:
            parts.append(
                "This review needs attention. "
            )
        elif weakest_score < 0.6:
            parts.append(
                "This review is acceptable but has room for improvement. "
            )
        else:
            parts.append(
                "This is a solid review. "
            )

        # Highlight the weakest area
        parts.append(
            f"The main concern is {metric_explanations.get(weakest_metric, weakest_metric)} "
            f"(score: {weakest_score:.2f}). "
        )

        # Highlight the strongest area
        if strongest_score > 0.7:
            parts.append(
                f"On the positive side, {strong_explanations.get(strongest_metric, strongest_metric)} "
                f"(score: {strongest_score:.2f})."
            )

        # Add actionable advice
        advice = {
            "faithfulness": (
                "Consider reviewing the suggestions against your "
                "coding standards to filter out any unsupported advice."
            ),
            "answer_relevancy": (
                "The review may contain generic advice. Focus on "
                "suggestions that specifically mention your code patterns."
            ),
            "context_precision": (
                "Some retrieved rules may not apply. You can add "
                "more specific rules to improve retrieval precision."
            ),
            "context_recall": (
                "Consider adding more specific coding rules to your "
                "standards database for better coverage."
            ),
        }

        parts.append(advice.get(weakest_metric, ""))

        return "".join(parts)

    def _fallback_result(self, error_message: str) -> dict:
        """
        Return a graceful fallback when evaluation cannot run.

        This ensures the main review pipeline is NEVER blocked
        by evaluation failures. The user still gets their review,
        just without quality metrics.
        """
        return {
            "is_evaluated": False,
            "error": f"Evaluation temporarily unavailable: {error_message}",
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
            "overall_quality": None,
            "quality_label": "Not evaluated",
            "interpretation": (
                "Review quality evaluation is currently unavailable. "
                "The code review above was generated successfully."
            ),
            "evaluation_time_ms": 0,
        }


# ──────────────────────────────────────────────────────────
# Module-level convenience function
# ──────────────────────────────────────────────────────────
_evaluator_instance = None


def evaluate_code_review(
    code_input: str,
    generated_review: dict,
    retrieved_rules: list
) -> dict:
    """
    Convenience function to evaluate a code review.

    Creates a singleton evaluator instance and runs evaluation.
    This is the recommended way to call the evaluator from
    other modules (pipeline, API, UI).

    Example:
        from core.evaluator import evaluate_code_review
        eval_result = evaluate_code_review(code, review, rules)
        if eval_result["is_evaluated"]:
            print(f"Quality: {eval_result['overall_quality']}")
    """
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = CodeReviewEvaluator()
    return _evaluator_instance.evaluate_review(
        code_input, generated_review, retrieved_rules
    )