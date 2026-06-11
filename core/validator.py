# core/validator.py

import json
from typing import Dict, Any, List


def validate_review(
    review: Dict[str, Any],
    ast_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Cross-checks LLM suggestions against AST data.
    Removes hallucinated function references.
    Ensures output structure is correct.
    """

    known_functions = ast_data.get("functions", [])

    # Validate suggestions
    validated_suggestions = []
    for suggestion in review.get("suggestions", []):
        # Check if suggestion references a non-existent function
        hallucinated = False
        words = suggestion.lower().split()

        for word in words:
            # If suggestion mentions "use function X" pattern
            if word in ["use", "call", "invoke"]:
                idx = words.index(word)
                if idx + 1 < len(words):
                    referenced_func = words[idx + 1].strip("()'\"")
                    # If it references a specific function
                    # that doesn't exist in code
                    if (referenced_func not in
                            [f.lower() for f in known_functions]
                            and len(referenced_func) > 2
                            and referenced_func.isidentifier()):
                        hallucinated = True

        if not hallucinated:
            validated_suggestions.append(suggestion)

    review["suggestions"] = validated_suggestions

    # Ensure quality score is valid
    score = review.get("quality_score", 5)
    if not isinstance(score, int) or score < 1 or score > 10:
        review["quality_score"] = 5

    # Ensure all required keys exist
    required_keys = {
        "bugs": [],
        "complexity": {
            "time": "Unknown",
            "space": "Unknown",
            "explanation": ""
        },
        "suggestions": [],
        "quality_score": 5,
        "improved_code": ""
    }

    for key, default in required_keys.items():
        if key not in review:
            review[key] = default

    return review