# tests/test_evaluator.py
"""
Tests for the LLM-based Code Review Evaluator.

These tests validate:
- Quality label mapping (High/Good/Acceptable/Needs Review)
- Human-readable interpretation generation
- Graceful fallback when no LLM backend is available
- Parsing/clamping of LLM score responses
- The full evaluated result path (with a mocked LLM client)

All tests are hermetic — they never make real network/LLM calls.
Evaluation is tested deterministically either by forcing the fallback
path or by injecting a fake in-memory LLM.

Run with:
    pytest tests/test_evaluator.py -v
    (or from project root: python -m pytest tests/test_evaluator.py -v)
"""

import sys
import os

# Add project root to path so core/ can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from core.evaluator import CodeReviewEvaluator


class FakeLLM:
    """
    Minimal stand-in for the LLM client used by the evaluator.

    Provides the same `complete(...)` interface so the evaluated path
    can be tested without any network access.
    """

    backend = "groq"
    default_model = "fake-model"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    def complete(self, messages, temperature=0.2, max_tokens=2000):
        self.calls.append(messages)
        return self.reply


@pytest.fixture
def evaluator():
    """
    Create a fresh evaluator instance for each test.
    Initialization is lazy (no network), so this is always cheap.
    """
    return CodeReviewEvaluator()


# ══════════════════════════════════════════════════════════
# Quality Label Tests
# These test pure logic — no LLM or network required.
# ══════════════════════════════════════════════════════════

class TestQualityLabels:
    """Test the get_quality_label boundary conditions."""

    def test_high_quality_boundary(self, evaluator):
        """Score 0.8 exactly should be 'High Quality'."""
        assert evaluator.get_quality_label(0.8) == "High Quality"

    def test_high_quality_above(self, evaluator):
        """Score above 0.8 should be 'High Quality'."""
        assert evaluator.get_quality_label(0.95) == "High Quality"
        assert evaluator.get_quality_label(1.0) == "High Quality"

    def test_good_quality_boundary(self, evaluator):
        """Score 0.6 should be 'Good Quality' (lower bound inclusive)."""
        assert evaluator.get_quality_label(0.6) == "Good Quality"

    def test_good_quality_mid(self, evaluator):
        """Score 0.7 should be 'Good Quality'."""
        assert evaluator.get_quality_label(0.7) == "Good Quality"

    def test_acceptable_boundary(self, evaluator):
        """Score 0.4 should be 'Acceptable' (lower bound inclusive)."""
        assert evaluator.get_quality_label(0.4) == "Acceptable"

    def test_acceptable_mid(self, evaluator):
        """Score 0.5 should be 'Acceptable'."""
        assert evaluator.get_quality_label(0.5) == "Acceptable"

    def test_needs_review_low(self, evaluator):
        """Score below 0.4 should be 'Needs Review'."""
        assert evaluator.get_quality_label(0.0) == "Needs Review"
        assert evaluator.get_quality_label(0.3) == "Needs Review"
        assert evaluator.get_quality_label(0.39) == "Needs Review"

    def test_all_boundaries(self, evaluator):
        """Verify the full boundary mapping in one test."""
        cases = {
            1.0: "High Quality",
            0.99: "High Quality",
            0.8: "High Quality",
            0.79: "Good Quality",
            0.7: "Good Quality",
            0.6: "Good Quality",
            0.59: "Acceptable",
            0.5: "Acceptable",
            0.4: "Acceptable",
            0.39: "Needs Review",
            0.0: "Needs Review",
        }
        for score, expected in cases.items():
            assert evaluator.get_quality_label(score) == expected, \
                f"Expected {expected} for score {score}"


# ══════════════════════════════════════════════════════════
# Interpretation Tests
# ──────────────────────────────────────────────────────────

class TestInterpretation:
    """Test the get_interpretation human-readable text generation."""

    def test_empty_metrics(self, evaluator):
        """Empty metrics should return a default message, not crash."""
        result = evaluator.get_interpretation({})
        assert isinstance(result, str)
        assert "No metrics" in result

    def test_identifies_weakest_metric(self, evaluator):
        """
        With one low metric, interpretation should reference it.
        context_recall (weakest) should be flagged as a retrieval gap.
        """
        metrics = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_precision": 0.8,
            "context_recall": 0.2,  # weakest
        }
        result = evaluator.get_interpretation(metrics)
        # The context-recall explanation mentions "retrieved"/"standards"
        assert "retrieved" in result.lower()
        assert "standards" in result.lower()

    def test_strong_high_scores(self, evaluator):
        """High scores should produce a positive interpretation."""
        metrics = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_precision": 0.8,
            "context_recall": 0.85,
        }
        result = evaluator.get_interpretation(metrics)
        assert "solid" in result.lower()

    def test_low_scores_warn(self, evaluator):
        """Very low scores should produce a warning interpretation."""
        metrics = {
            "faithfulness": 0.2,
            "answer_relevancy": 0.5,
            "context_precision": 0.6,
            "context_recall": 0.5,
        }
        result = evaluator.get_interpretation(metrics)
        assert "attention" in result.lower()
        # Faithfulness (weakest) explanation mentions "grounded"/"standards"
        assert "grounded" in result.lower()

    def test_result_is_string(self, evaluator):
        """Interpretation must always be a string."""
        result = evaluator.get_interpretation({
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
        })
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════
# Graceful Fallback Tests
# These verify that evaluation degrades gracefully when no
# LLM backend is available — without ever making a network call.
# ──────────────────────────────────────────────────────────

class TestGracefulFallback:
    """Test that evaluation gracefully fails without breaking."""

    def test_fallback_returns_expected_structure(self, evaluator):
        """
        Without an LLM backend, evaluation must return the
        is_evaluated=False fallback structure, not crash.
        """
        # Force the no-backend path deterministically
        evaluator._llm = None

        result = evaluator.evaluate_review(
            code_input="def foo():\n    return 1",
            generated_review={
                "bugs": ["none"],
                "suggestions": ["use hints"],
                "quality_score": 7,
                "complexity": {"time": "O(1)", "space": "O(1)",
                               "explanation": ""},
            },
            retrieved_rules=["Use type hints"],
        )

        assert isinstance(result, dict)
        assert result["is_evaluated"] is False
        assert result["quality_label"] == "Not evaluated"
        assert result["overall_quality"] is None
        assert result["faithfulness"] is None
        assert "is_evaluated" in result

    def test_fallback_does_not_crash_with_empty_rules(self, evaluator):
        """
        Empty retrieved rules should not cause a crash.
        """
        evaluator._llm = None

        result = evaluator.evaluate_review(
            code_input="x = 5",
            generated_review={"bugs": [], "suggestions": []},
            retrieved_rules=[],
        )
        assert isinstance(result, dict)
        assert result["is_evaluated"] is False

    def test_fallback_caches_no_llm_error_message(self, evaluator):
        """Fallback error should mention configuring the LLM backend."""
        evaluator._llm = None
        result = evaluator.evaluate_review(
            code_input="x = 5",
            generated_review={"bugs": []},
            retrieved_rules=["r"],
        )
        assert "LLM" in result["error"]


# ══════════════════════════════════════════════════════════
# LLM Score Parsing Tests
# These exercise the parsing/clamping logic in isolation.
# ──────────────────────────────────────────────────────────

class TestScoreParsing:
    """Test _parse_scores against raw LLM responses."""

    def test_parses_valid_json(self, evaluator):
        raw = ('{"faithfulness": 0.9, "answer_relevancy": 0.8, '
              '"context_precision": 0.7, "context_recall": 0.6}')
        scores = evaluator._parse_scores(raw)
        assert scores == {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }

    def test_parses_json_with_markdown_fence(self, evaluator):
        raw = ('```json\n{"faithfulness": 0.5, "answer_relevancy": 0.5, '
               '"context_precision": 0.5, "context_recall": 0.5}\n```')
        scores = evaluator._parse_scores(raw)
        assert all(scores[m] == 0.5 for m in scores)

    def test_clamps_out_of_range_scores(self, evaluator):
        raw = ('{"faithfulness": 5.0, "answer_relevancy": -1.0, '
               '"context_precision": 1.2, "context_recall": 0.4}')
        scores = evaluator._parse_scores(raw)
        assert scores["faithfulness"] == 1.0
        assert scores["answer_relevancy"] == 0.0
        assert scores["context_precision"] == 1.0
        assert scores["context_recall"] == 0.4

    def test_bad_json_returns_none(self, evaluator):
        assert evaluator._parse_scores("not json at all") is None

    def test_missing_metric_returns_none(self, evaluator):
        raw = '{"faithfulness": 0.9}'
        assert evaluator._parse_scores(raw) is None


# ══════════════════════════════════════════════════════════
# Full Evaluated-Path Tests (mocked LLM, no network)
# ──────────────────────────────────────────────────────────

class TestEvaluatedPath:
    """Test the full evaluation flow with a fake in-memory LLM."""

    def test_success_returns_evaluated_result(self, evaluator):
        llm = FakeLLM(
            '{"faithfulness": 0.9, "answer_relevancy": 0.8, '
            '"context_precision": 0.7, "context_recall": 0.6}'
        )
        evaluator._llm = llm

        result = evaluator.evaluate_review(
            code_input="def foo():\n    return 1",
            generated_review={
                "bugs": ["unused variable"],
                "suggestions": ["remove it"],
                "quality_score": 6,
                "complexity": {"time": "O(1)", "space": "O(1)",
                               "explanation": "simple"},
            },
            retrieved_rules=["Use clear variable names"],
        )

        assert result["is_evaluated"] is True
        assert result["faithfulness"] == 0.9
        assert result["answer_relevancy"] == 0.8
        assert result["context_precision"] == 0.7
        assert result["context_recall"] == 0.6
        # 0.9*0.35 + 0.8*0.30 + 0.7*0.20 + 0.6*0.15 = 0.785
        assert result["overall_quality"] == pytest.approx(0.785)
        assert result["quality_label"] == "Good Quality"
        assert isinstance(result["interpretation"], str)

    def test_success_passes_review_text_and_rules(self, evaluator):
        llm = FakeLLM(
            '{"faithfulness": 0.5, "answer_relevancy": 0.5, '
            '"context_precision": 0.5, "context_recall": 0.5}'
        )
        evaluator._llm = llm

        evaluator.evaluate_review(
            code_input="z = 1",
            generated_review={"bugs": ["a bug"], "suggestions": ["a fix"]},
            retrieved_rules=["Rule one", "Rule two"],
        )

        # The single user prompt should contain the code, rules, and review
        user_content = llm.calls[0][1]["content"]
        assert "z = 1" in user_content
        assert "Rule one" in user_content
        assert "a bug" in user_content

    def test_unparseable_llm_response_falls_back(self, evaluator):
        evaluator._llm = FakeLLM("I'm sorry, I can't do that.")

        result = evaluator.evaluate_review(
            code_input="x = 1",
            generated_review={"bugs": []},
            retrieved_rules=["some rule"],
        )

        assert result["is_evaluated"] is False
        assert result["quality_label"] == "Not evaluated"

    def test_llm_exception_falls_back(self, evaluator):
        class ExplodingLLM:
            def complete(self, messages, temperature=0.2, max_tokens=2000):
                raise RuntimeError("backend down")

        evaluator._llm = ExplodingLLM()

        result = evaluator.evaluate_review(
            code_input="x = 1",
            generated_review={"bugs": []},
            retrieved_rules=["some rule"],
        )

        assert result["is_evaluated"] is False
        assert "unavailable" in result["error"]


# ══════════════════════════════════════════════════════════
# Weighted Score Logic Test
# ──────────────────────────────────────────────────────────

class TestWeightedScore:
    """
    Tests the weighted-average composition logic.
    """

    def test_weights_sum_to_one(self):
        """The metric weights must sum to 1.0 for a valid weighted average."""
        from core.evaluator import METRIC_WEIGHTS
        total = sum(METRIC_WEIGHTS.values())
        # Allow for float precision
        assert abs(total - 1.0) < 1e-9

    def test_required_weights_present(self):
        """All 4 required metrics must have defined weights."""
        from core.evaluator import METRIC_WEIGHTS
        required = {
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        }
        assert required <= set(METRIC_WEIGHTS.keys())

    def test_faithfulness_is_heaviest(self):
        """Faithfulness should be the most weighted metric (35%)."""
        from core.evaluator import METRIC_WEIGHTS
        assert METRIC_WEIGHTS["faithfulness"] == 0.35

    def test_weight_ordering(self):
        """
        Verify the expected weight order:
        faithfulness > answer_relevancy > context_precision > context_recall
        """
        from core.evaluator import METRIC_WEIGHTS
        assert METRIC_WEIGHTS["faithfulness"] > METRIC_WEIGHTS["answer_relevancy"]
        assert METRIC_WEIGHTS["answer_relevancy"] > METRIC_WEIGHTS["context_precision"]
        assert METRIC_WEIGHTS["context_precision"] > METRIC_WEIGHTS["context_recall"]