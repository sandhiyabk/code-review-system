# core/pipeline.py

from core.ast_analyzer import analyze_with_ast
from core.chroma_store import StyleRuleStore
from core.prompt_builder import build_prompt
from core.llm_reviewer import get_llm_review
from core.validator import validate_review
import json

# Initialize ChromaDB store once
store = StyleRuleStore()


def review_code(code: str) -> dict:
    """
    Full pipeline: code → AST → RAG → LLM → validate → result
    """

    print("\n=== Starting Code Review Pipeline ===")

    # Layer 1: AST Analysis
    print("Step 1: Analyzing code structure with AST...")
    ast_data = analyze_with_ast(code)

    if not ast_data["valid"]:
        return {
            "bugs": ["Syntax error in code: " + ast_data.get("error", "")],
            "complexity": {"time": "N/A", "space": "N/A", "explanation": ""},
            "suggestions": ["Fix syntax errors first"],
            "quality_score": 0,
            "improved_code": code
        }

    print(f"   Found: {ast_data['functions']} functions, "
          f"{ast_data['nested_loops']} nested loops")

    # Layer 2: RAG — Retrieve relevant rules
    print("Step 2: Retrieving relevant style rules from ChromaDB...")
    relevant_rules = store.get_relevant_rules(code)
    print(f"   Retrieved: {len(relevant_rules)} relevant rules")

    # Layer 3: Build enriched prompt
    print("Step 3: Building enriched prompt...")
    prompt = build_prompt(code, ast_data, relevant_rules)

    # Layer 4: LLM Review (backend-agnostic — Groq or local Ollama)
    print("Step 4: Sending to LLM for review...")
    raw_review = get_llm_review(prompt)

    # If the LLM call itself failed, get_llm_review returns a review-shaped
    # dict with a user-friendly error + suggestions. Cross-validating that
    # against AST is meaningless and would overwrite the score, so return
    # the error result as-is (additive guard, doesn't affect success path).
    if raw_review.get("llm_error"):
        return raw_review

    # Layer 5: Validate
    print("Step 5: Validating against AST data...")
    final_review = validate_review(raw_review, ast_data)

    print("=== Review Complete ===\n")
    return final_review


# Test the complete pipeline
if __name__ == "__main__":
    test_code = """
def find_user(users, target_id):
    for i in range(len(users)):
        for j in range(len(users)):
            if users[i].id == target_id:
                return users[i]
    return None
"""

    result = review_code(test_code)
    print(json.dumps(result, indent=2))