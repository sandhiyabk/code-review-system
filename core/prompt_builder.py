# core/prompt_builder.py

from typing import List, Dict, Any


def build_prompt(
    code: str,
    ast_data: Dict[str, Any],
    relevant_rules: List[str]
) -> str:
    """
    Combines code + AST findings + relevant style rules
    into a single enriched prompt for the LLM.
    """

    # Format rules as numbered list
    if relevant_rules:
        rules_text = "\n".join(
            [f"{i+1}. {rule}"
             for i, rule in enumerate(relevant_rules)]
        )
    else:
        rules_text = "No specific rules retrieved — apply general Python best practices."

    # Format AST findings
    ast_summary = f"""
- Functions defined: {ast_data.get('functions', [])}
- Nested loops: {ast_data.get('nested_loops', 0)}
- Has error handling: {ast_data.get('has_error_handling', False)}
- Known issues: {ast_data.get('issues', [])}
    """.strip()

    prompt = f"""Review the following Python code.

=== STRUCTURAL ANALYSIS (from AST parser) ===
{ast_summary}

=== RELEVANT STYLE RULES (retrieved for this code) ===
{rules_text}

=== CODE TO REVIEW ===
{code}

=== INSTRUCTIONS ===
1. Use the structural analysis to confirm real issues
2. Apply the retrieved style rules specifically
3. Only reference functions that exist: {ast_data.get('functions', [])}
4. Be specific — mention line patterns not generic advice
5. Return ONLY valid JSON

JSON format:
{{
    "bugs": ["specific bug 1", "specific bug 2"],
    "complexity": {{
        "time": "O(n²)",
        "space": "O(n)",
        "explanation": "explanation here"
    }},
    "suggestions": ["specific suggestion 1"],
    "quality_score": 5,
    "improved_code": "def improved_version..."
}}"""

    return prompt


# Test it directly
if __name__ == "__main__":
    test_code = "def f(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)):\n            print(arr[i])"

    test_ast = {
        "functions": ["f"],
        "nested_loops": 1,
        "has_error_handling": False,
        "issues": ["Nested loops detected"]
    }

    test_rules = [
        "Avoid nested loops deeper than 2 levels",
        "Use enumerate() instead of range(len())"
    ]

    prompt = build_prompt(test_code, test_ast, test_rules)
    print(prompt)
    print("\n--- Prompt length:", len(prompt), "chars ---")