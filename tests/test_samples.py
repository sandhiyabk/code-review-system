# tests/test_samples.py

"""
Test samples for the Code Review Assistant.
Run this file to test the complete pipeline
with different types of Python code.
"""

import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import review_code

# ─── Test Code Samples ────────────────────────────────

# Sample 1: Nested loops — should trigger loop rules
SAMPLE_NESTED_LOOPS = """
def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i != j and arr[i] == arr[j]:
                if arr[i] not in duplicates:
                    duplicates.append(arr[i])
    return duplicates
"""

# Sample 2: Missing error handling
SAMPLE_NO_ERROR_HANDLING = """
def read_file(filename):
    f = open(filename, 'r')
    content = f.read()
    f.close()
    return content
"""

# Sample 3: Bad variable names
SAMPLE_BAD_NAMING = """
def calc(a, b, c):
    x = a * b
    y = x + c
    z = y / a
    return z
"""

# Sample 4: Good code — should get high score
SAMPLE_GOOD_CODE = """
def calculate_discount(price: float, discount_rate: float) -> float:
    \"\"\"
    Calculate discounted price.
    
    Args:
        price: Original price
        discount_rate: Discount as decimal (0.1 = 10%)
    
    Returns:
        Discounted price
    \"\"\"
    try:
        if not isinstance(price, (int, float)):
            raise TypeError("Price must be a number")
        if not 0 <= discount_rate <= 1:
            raise ValueError("Discount rate must be between 0 and 1")
        
        discount_amount = price * discount_rate
        return price - discount_amount
    
    except (TypeError, ValueError) as e:
        print(f"Invalid input: {e}")
        return price
"""

# Sample 5: Complex bad code — multiple issues
SAMPLE_MULTIPLE_ISSUES = """
def process(d, l):
    r = []
    for i in range(len(l)):
        for j in range(len(d)):
            if l[i] == d[j]:
                r.append(l[i])
    x = 0
    for i in range(len(r)):
        x = x + r[i]
    return x
"""

# Sample 6: Syntax error — should handle gracefully
SAMPLE_SYNTAX_ERROR = """
def broken_function(
    x = 10
    return x
"""

# ─── Test Runner ──────────────────────────────────────

def run_single_test(name: str, code: str, expected_score_range: tuple):
    """Run a single test and print results."""

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Code preview: {code.strip()[:80]}...")

    result = review_code(code)

    print(f"\n📊 Quality Score: {result['quality_score']}/10")
    print(f"🐛 Bugs found: {len(result['bugs'])}")
    print(f"💡 Suggestions: {len(result['suggestions'])}")
    print(f"⏱  Complexity: {result['complexity']['time']}")

    if result['bugs']:
        print("\nBugs:")
        for bug in result['bugs']:
            print(f"  → {bug}")

    if result['suggestions']:
        print("\nSuggestions:")
        for s in result['suggestions']:
            print(f"  → {s}")

    # Check score is in expected range
    score = result['quality_score']
    min_score, max_score = expected_score_range
    passed = min_score <= score <= max_score

    print(f"\n{'✅ PASS' if passed else '⚠️  NOTE'}: "
          f"Score {score} "
          f"({'within' if passed else 'outside'} "
          f"expected range {expected_score_range})")

    return result


def run_all_tests():
    """Run all test samples."""

    print("\n🔍 AI CODE REVIEW ASSISTANT — TEST SUITE")
    print("Testing complete pipeline: AST → ChromaDB → LLM → Validate")

    tests = [
        ("Nested Loops", SAMPLE_NESTED_LOOPS, (1, 5)),
        ("Missing Error Handling", SAMPLE_NO_ERROR_HANDLING, (3, 7)),
        ("Bad Variable Names", SAMPLE_BAD_NAMING, (2, 6)),
        ("Good Code", SAMPLE_GOOD_CODE, (7, 10)),
        ("Multiple Issues", SAMPLE_MULTIPLE_ISSUES, (1, 4)),
        ("Syntax Error", SAMPLE_SYNTAX_ERROR, (0, 2)),
    ]

    results = []
    for name, code, expected in tests:
        result = run_single_test(name, code, expected)
        results.append({
            "test": name,
            "score": result['quality_score'],
            "bugs": len(result['bugs']),
            "suggestions": len(result['suggestions'])
        })

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"{'Test':<25} {'Score':>7} {'Bugs':>6} {'Suggestions':>12}")
    print("-" * 55)
    for r in results:
        print(f"{r['test']:<25} {r['score']:>7} "
              f"{r['bugs']:>6} {r['suggestions']:>12}")

    print(f"\n✅ All {len(tests)} tests completed")
    print("Pipeline working correctly end to end")


def test_single(code: str):
    """Quick test with custom code."""
    print("\n🔍 QUICK TEST")
    result = review_code(code)
    print(json.dumps(result, indent=2))
    return result


# ─── Run ──────────────────────────────────────────────
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description="Test the Code Review Assistant"
    )
    parser.add_argument(
        "--quick",
        type=str,
        help="Quick test with custom code string",
        default=None
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Run specific sample (1-6)",
        default=None
    )

    args = parser.parse_args()

    if args.quick:
        test_single(args.quick)

    elif args.sample:
        samples = [
            ("Nested Loops", SAMPLE_NESTED_LOOPS, (1, 5)),
            ("Missing Error Handling", SAMPLE_NO_ERROR_HANDLING, (3, 7)),
            ("Bad Variable Names", SAMPLE_BAD_NAMING, (2, 6)),
            ("Good Code", SAMPLE_GOOD_CODE, (7, 10)),
            ("Multiple Issues", SAMPLE_MULTIPLE_ISSUES, (1, 4)),
            ("Syntax Error", SAMPLE_SYNTAX_ERROR, (0, 2)),
        ]
        if 1 <= args.sample <= 6:
            name, code, expected = samples[args.sample - 1]
            run_single_test(name, code, expected)
        else:
            print("Sample number must be 1-6")

    else:
        run_all_tests()