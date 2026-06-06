# core/ast_analyzer.py

import ast
from typing import Dict, List, Any

def analyze_with_ast(code: str) -> Dict[str, Any]:
    """
    Analyzes Python code structure using AST.
    Returns structural findings for LLM context
    and hallucination validation.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "valid": False,
            "error": str(e),
            "functions": [],
            "nested_loops": 0,
            "has_error_handling": False,
            "imports": [],
            "issues": ["SyntaxError: " + str(e)]
        }

    analysis = {
        "valid": True,
        "functions": [],
        "nested_loops": 0,
        "has_error_handling": False,
        "imports": [],
        "issues": [],
        "loop_count": 0,
    }

    for node in ast.walk(tree):

        # Collect all function names
        if isinstance(node, ast.FunctionDef):
            analysis["functions"].append(node.name)

        # Collect imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                analysis["imports"].append(alias.name)

        if isinstance(node, ast.ImportFrom):
            analysis["imports"].append(node.module)

        # Count all loops
        if isinstance(node, (ast.For, ast.While)):
            analysis["loop_count"] += 1

        # Detect nested loops
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if child is not node:
                    if isinstance(child, (ast.For, ast.While)):
                        analysis["nested_loops"] += 1
                        if "Nested loops detected" not in analysis["issues"]:
                            analysis["issues"].append(
                                "Nested loops detected — consider refactoring"
                            )

        # Check for error handling
        if isinstance(node, ast.Try):
            analysis["has_error_handling"] = True

    # Flag missing error handling
    if not analysis["has_error_handling"] and analysis["loop_count"] > 0:
        analysis["issues"].append(
            "No error handling found — consider adding try/except"
        )

    return analysis


# Test it directly
if __name__ == "__main__":
    test_code = """
def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i != j and arr[i] == arr[j]:
                if arr[i] not in duplicates:
                    duplicates.append(arr[i])
    return duplicates
"""
    result = analyze_with_ast(test_code)
    print("Functions:", result["functions"])
    print("Nested loops:", result["nested_loops"])
    print("Has error handling:", result["has_error_handling"])
    print("Issues found:", result["issues"])