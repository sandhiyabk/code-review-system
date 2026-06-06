# data/style_rules.py

STYLE_RULES = [
    # Loops
    "Avoid nested loops deeper than 2 levels — use helper functions instead",
    "Use enumerate() instead of range(len()) when you need index and value",
    "Use list comprehensions for simple loop transforms instead of for loops",
    "Break complex loops into separate functions for readability",

    # Functions
    "Each function should do exactly one thing — single responsibility",
    "Functions longer than 20 lines should be broken into smaller functions",
    "Always add docstrings to explain what a function does",
    "Validate and check input parameters at the start of every function",
    "Use type hints for all function parameters and return values",

    # Error Handling
    "Always wrap file and network operations in try/except blocks",
    "Never use bare except — always specify the exception type",
    "Log errors properly instead of silently passing them",
    "Raise meaningful custom exceptions with clear error messages",

    # Variables
    "Use descriptive variable names — avoid single letters except in loops",
    "Avoid global variables — pass data as function parameters instead",
    "Use constants for magic numbers — give them meaningful names",
    "Use snake_case for variables and functions in Python",

    # Code Quality
    "Return early to avoid deeply nested if/else blocks",
    "Use f-strings instead of string concatenation or .format()",
    "Remove dead code, unused imports, and commented-out code",
    "Keep lines under 79 characters for PEP8 compliance",

    # Data Structures
    "Use dictionaries for O(1) lookup instead of searching lists",
    "Use sets when you need uniqueness and fast membership testing",
    "Use defaultdict or Counter from collections for counting patterns",

    # OOP
    "Use __slots__ in classes to reduce memory for many instances",
    "Prefer composition over inheritance for flexibility",
]