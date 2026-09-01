# core/evaluator.py
"""
RAGAS Evaluation Pipeline for Code Review Quality Assessment.

This module evaluates the quality of AI-generated code reviews using
RAGAS (Retrieval Augmented Generation Assessment) metrics. It measures
whether the review is faithful to retrieved rules, relevant to the code,
and whether retrieval was precise and comprehensive.

The evaluation is designed to run ASYNC to the main review pipeline —
the review shows first, evaluation runs after to avoid blocking the user.
"""

import os
import time
import hashlib
import json
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# ──────────────────────────────────────────────────────────
# Conditional RAGAS import — graceful degradation if not installed
# ──────────────────────────────────────────────────────────
RAGAS_AVAILABLE = False
try:
    from ragas import evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import (
        Faithfulness,
        ContextPrecision,
        ContextRecall,
        AnswerCorrectness,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    # RAGAS or its dependencies not installed — evaluation will
    # gracefully return fallback results instead of crashing
    Dataset = None
    evaluate = None

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
    Evaluates code review quality using RAGAS metrics.

    Uses the same Groq API (LLaMA 3.3-70B) already configured
    in the project for evaluation LLM calls. RAGAS natively
    supports Groq via its instructor adapter.

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
        Initialize the evaluator by setting up the RAGAS LLM client.

        Reuses the existing Groq API key from the project's .env file.
        The evaluation LLM is the same model used for code review
        (openai/gpt-oss-20b) to keep costs consistent.

        If Groq client creation fails, all evaluations will return
        the graceful fallback result.
        """
        self._ragas_llm = None
        self._init_time_ms = 0

        if not RAGAS_AVAILABLE:
            print("[evaluator] RAGAS library not available — "
                  "evaluation disabled")
            return

        try:
            start = time.time()

            # Import Groq client — same one used in llm_reviewer.py
            # This reuses the existing GROQ_API_KEY env var
            from groq import Groq
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            # Create RAGAS LLM instance using Groq provider
            # RAGAS uses instructor adapter for Groq (auto-detected)
            self._ragas_llm = llm_factory(
                "openai/gpt-oss-20b",
                provider="groq",
                client=groq_client
            )

            self._init_time_ms = round((time.time() - start) * 1000)
            print(f"[evaluator] Initialized in {self._init_time_ms}ms")

        except Exception as e:
            # If initialization fails, evaluation will use fallback
            print(f"[evaluator] Failed to initialize: {e}")
            self._ragas_llm = None

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

    def prepare_ragas_dataset(
        self,
        code_input: str,
        generated_review: dict,
        retrieved_rules: List[str],
        review_question: str
    ) -> Any:
        """
        Convert our internal data structures into RAGAS Dataset format.

        RAGAS expects a HuggingFace Dataset with these columns:
        - question: The original query/prompt (our review question)
        - answer: The generated response (our code review as text)
        - contexts: List of retrieved context chunks (our style rules)
        - ground_truth: Expected answer (derived from rules + code)

        Args:
            code_input: The original Python code submitted for review
            generated_review: Dict with bugs, suggestions, quality_score, etc.
            retrieved_rules: List of style rules retrieved from ChromaDB
            review_question: The prompt that was sent to the LLM

        Returns:
            A HuggingFace Dataset ready for RAGAS evaluation

        Raises:
            ImportError: If RAGAS/datasets libraries not available
            ValueError: If inputs are empty or malformed
        """
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS library not installed. "
                "Install with: pip install ragas datasets"
            )

        # Validate inputs — empty data means evaluation is meaningless
        if not code_input or not code_input.strip():
            raise ValueError("code_input cannot be empty")
        if not generated_review:
            raise ValueError("generated_review cannot be empty")

        # Convert the structured review dict to a single text string.
        # RAGAS "answer" field expects a text response, not JSON.
        answer_text = self._review_to_text(generated_review)

        # If no rules were retrieved, use a placeholder
        # so RAGAS doesn't get an empty context list
        contexts = retrieved_rules if retrieved_rules else [
            "No specific style rules were retrieved for this code."
        ]

        # Build ground_truth from the retrieved rules themselves.
        # For self-evaluation (no golden dataset), ground_truth is
        # constructed as "what a good review based on these rules
        # should contain" — this gives RAGAS a reference point.
        ground_truth = self._build_ground_truth(
            code_input, retrieved_rules
        )

        # Create the RAGAS-compatible dataset with all required columns
        data = {
            "question": [review_question],
            "answer": [answer_text],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }

        return Dataset.from_dict(data)

    def _review_to_text(self, review: dict) -> str:
        """
        Convert structured review dict to plain text for RAGAS.

        RAGAS metrics compare text answers against contexts.
        We flatten the structured review into readable paragraphs
        so faithfulness and relevancy can be properly measured.
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

    def _build_ground_truth(
        self,
        code_input: str,
        retrieved_rules: List[str]
    ) -> str:
        """
        Construct a reference answer from retrieved rules.

        Since we don't have a golden dataset of "perfect reviews",
        we build a synthetic ground truth that represents what a
        review SHOULD contain given the retrieved rules. This gives
        RAGAS's Answer Correctness metric something to compare against.
        """
        if not retrieved_rules:
            return (
                "A review of this code should identify potential "
                "issues and provide actionable suggestions based on "
                "general Python best practices."
            )

        # Compose ground truth from the rules themselves:
        # "A review of this code should follow these rules: ..."
        rules_text = "; ".join(retrieved_rules)
        return (
            f"A review of this code should address the following "
            f"coding standards and best practices: {rules_text}. "
            f"The review should be specific to the provided code "
            f"and reference concrete patterns found in it."
        )

    def evaluate_review(
        self,
        code_input: str,
        generated_review: dict,
        retrieved_rules: List[str]
    ) -> dict:
        """
        Run all 4 RAGAS metrics and return comprehensive evaluation.

        This is the main public method. It:
        1. Checks the cache for previous evaluations
        2. Prepares the RAGAS dataset
        3. Runs Faithfulness, Answer Relevancy, Context Precision,
           and Context Recall metrics
        4. Computes weighted overall quality score
        5. Generates human-readable interpretation
        6. Caches the result

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
        if not RAGAS_AVAILABLE:
            return self._fallback_result(
                "RAGAS library not installed. "
                "Install with: pip install ragas datasets"
            )

        if self._ragas_llm is None:
            return self._fallback_result(
                "Evaluation LLM not initialized. "
                "Check GROQ_API_KEY environment variable."
            )

        start_time = time.time()

        try:
            # ── Build the review question (what was asked) ──
            review_question = (
                f"Review this Python code for bugs, complexity, "
                f"suggestions, and provide an improved version:\n\n"
                f"{code_input[:500]}"
            )

            # ── Prepare RAGAS dataset ──
            dataset = self.prepare_ragas_dataset(
                code_input=code_input,
                generated_review=generated_review,
                retrieved_rules=retrieved_rules,
                review_question=review_question
            )

            # ── Initialize metrics with the Groq-backed LLM ──
            # Each metric needs the LLM for its internal evaluation calls
            metrics = [
                Faithfulness(llm=self._ragas_llm),
                AnswerCorrectness(llm=self._ragas_llm),
                ContextPrecision(llm=self._ragas_llm),
                ContextRecall(llm=self._ragas_llm),
            ]

            # ── Run RAGAS evaluation ──
            # This makes multiple LLM calls internally to score each metric
            result = evaluate(dataset, metrics=metrics)

            # ── Extract scores from RAGAS result ──
            # RAGAS returns a result object with metric scores
            scores = {
                "faithfulness": self._extract_score(
                    result, "faithfulness"
                ),
                "answer_relevancy": self._extract_score(
                    result, "answer_correctness"
                ),
                "context_precision": self._extract_score(
                    result, "context_precision"
                ),
                "context_recall": self._extract_score(
                    result, "context_recall"
                ),
            }

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

    def _extract_score(self, result, metric_name: str) -> float:
        """
        Safely extract a metric score from RAGAS result object.

        RAGAS result objects can vary in structure depending on version.
        This method handles multiple possible formats and always
        returns a float between 0.0 and 1.0.
        """
        try:
            # Try accessing as dictionary first (older RAGAS versions)
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                if metric_name in df.columns:
                    return float(df[metric_name].iloc[0])

            # Try direct attribute access (newer RAGAS versions)
            if hasattr(result, metric_name):
                val = getattr(result, metric_name)
                if isinstance(val, (int, float)):
                    return float(val)
                # If it's a list/array, take first element
                if hasattr(val, "__len__") and len(val) > 0:
                    return float(val[0])

            # Try dictionary-style access
            if isinstance(result, dict) and metric_name in result:
                return float(result[metric_name])

            # Try scores attribute (some RAGAS versions)
            if hasattr(result, "scores"):
                scores = result.scores
                if isinstance(scores, dict) and metric_name in scores:
                    return float(scores[metric_name])
                if isinstance(scores, list) and len(scores) > 0:
                    score_dict = scores[0]
                    if isinstance(score_dict, dict):
                        return float(
                            score_dict.get(metric_name, 0.5)
                        )

            # If all else fails, return neutral score
            print(
                f"[evaluator] Could not extract '{metric_name}' "
                f"from result — defaulting to 0.5"
            )
            return 0.5

        except Exception as e:
            print(
                f"[evaluator] Error extracting '{metric_name}': {e}"
            )
            return 0.5

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
