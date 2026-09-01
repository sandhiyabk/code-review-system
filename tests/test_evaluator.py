# tests/test_evaluator.py
"""
Tests for the RAGAS-based Code Review Evaluator.

These tests validate:
- Quality label mapping (High/Good/Acceptable/Needs Review)
- Human-readable interpretation generation
- Graceful fallback when RAGAS is unavailable
- Dataset preparation logic (when RAGAS is available)

The RAGAS-dependent tests are skipped automatically if the
RAGAS library is not installed, so the test suite never
fails due to a missing optional dependency.

Run with:
    pytest tests/test_evaluator.py -v
    (or from project root: python -m pytest tests/test_evaluator.py -v)
"""

import sys
import os

# Add project root to path so core/ can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# Import evaluator — this won't fail even if RAGAS is missing
# because the module handles missing RAGAS gracefully
from core.evaluator import CodeReviewEvaluator


@pytest.fixture
def evaluator():
    """
    Create a fresh evaluator instance for each test.
    Note: initialization may be expensive if RAGAS is available,
    so tests that need it use a session-scoped fixture instead.
    """
    return CodeReviewEvaluator()


# ══════════════════════════════════════════════════════════
# Quality Label Tests
# These test pure logic — no RAGAS or network required.
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
# Dataset Preparation Tests (requires RAGAS + datasets)
# ──────────────────────────────────────────────────────────

try:
    from core.evaluator import RAGAS_AVAILABLE as _RAGAS_INSTALLED
except Exception:
    _RAGAS_INSTALLED = False


@pytest.mark.skipif(not _RAGAS_INSTALLED, reason="RAGAS not installed")
class TestDatasetPreparation:
    """
    Tests for prepare_ragas_dataset.
    Requires ragas and datasets libraries to be installed.
    """

    def test_creates_valid_dataset(self, evaluator):
        """Should create a Dataset with the correct columns."""
        dataset = evaluator.prepare_ragas_dataset(
            code_input="def foo():\n    return 1",
            generated_review={
                "bugs": ["no bugs"],
                "suggestions": ["add type hints"],
                "quality_score": 8,
                "complexity": {"time": "O(1)", "space": "O(1)",
                               "explanation": "simple"},
            },
            retrieved_rules=["Use type hints for all functions"],
            review_question="Review this Python code",
        )

        # Verify correct columns exist in RAGAS format
        required_columns = ["question", "answer", "contexts", "ground_truth"]
        for col in required_columns:
            assert col in dataset.column_names, f"Missing column: {col}"

    def test_empty_code_raises(self, evaluator):
        """Empty code input should raise ValueError."""
        with pytest.raises(ValueError):
            evaluator.prepare_ragas_dataset(
                code_input="",
                generated_review={"bugs": []},
                retrieved_rules=["rule"],
                review_question="Review",
            )

    def test_empty_review_raises(self, evaluator):
        """Empty review should raise ValueError."""
        with pytest.raises(ValueError):
            evaluator.prepare_ragas_dataset(
                code_input="def foo(): pass",
                generated_review=None,
                retrieved_rules=["rule"],
                review_question="Review",
            )


# ══════════════════════════════════════════════════════════
# Graceful Fallback Tests
# These never require RAGAS — they verify behavior when
# evaluation cannot run.
# ──────────────────────────────────────────────────────────

class TestGracefulFallback:
    """Test that evaluation gracefully fails without breaking."""

    def test_fallback_returns_expected_structure(self, evaluator):
        """
        When RAGAS is not available, evaluation must return
        the is_evaluated=False fallback structure, not crash.
        """
        # This test must pass regardless of RAGAS availability
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

        # The result must always be a dict
        assert isinstance(result, dict)

        # If RAGAS is not installed, it must be the fallback
        if not _RAGAS_INSTALLED:
            assert result["is_evaluated"] is False
            assert result["quality_label"] == "Not evaluated"
            assert result["overall_quality"] is None
            assert "is_evaluated" in result

    def test_fallback_does_not_crash_with_empty_rules(self, evaluator):
        """
        Empty retrieved rules should not cause a crash.
        """
        result = evaluator.evaluate_review(
            code_input="x = 5",
            generated_review={"bugs": [], "suggestions": []},
            retrieved_rules=[],
        )
        assert isinstance(result, dict)


# ══════════════════════════════════════════════════════════
# Weighted Score Logic Test
# ──────────────────────────────────────────────────────────

class TestWeightedScore:
    """
    Tests the weighted-average composition logic.

    Even though the overall score is computed inside evaluate_review
    (which requires RAGAS), we can verify the weight definitions
    and the formula logic used in the module.
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
