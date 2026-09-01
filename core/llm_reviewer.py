# core/llm_reviewer.py

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_llm_review(prompt: str) -> dict:
    """
    Sends enriched prompt to Groq LLM.
    Returns structured review as dict.
    """
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert Python code reviewer.
You MUST respond with ONLY valid JSON.
No explanation before or after JSON.
No markdown code blocks.
Just raw JSON.

Required format:
{
    "bugs": ["specific bug description"],
    "complexity": {
        "time": "O(n)",
        "space": "O(1)",
        "explanation": "brief explanation"
    },
    "suggestions": ["specific actionable suggestion"],
    "quality_score": 7,
    "improved_code": "the improved version of the code"
}"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=3000,
        )

        raw = response.choices[0].message.content

        # Clean any markdown if present
        clean = re.sub(r'```json|```', '', raw).strip()

        return json.loads(clean)

    except json.JSONDecodeError:
        # Return structured fallback if JSON parsing fails
        return {
            "bugs": ["Could not parse structured review"],
            "complexity": {
                "time": "Unknown",
                "space": "Unknown",
                "explanation": "Analysis failed"
            },
            "suggestions": [raw[:500]],
            "quality_score": 0,
            "improved_code": ""
        }
    except Exception as e:
        return {
            "bugs": [f"API Error: {str(e)}"],
            "complexity": {"time": "N/A", "space": "N/A", "explanation": ""},
            "suggestions": [],
            "quality_score": 0,
            "improved_code": ""
        }


# Test it directly
if __name__ == "__main__":
    test_prompt = """
Review this Python code:

def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i != j and arr[i] == arr[j]:
                duplicates.append(arr[i])
    return duplicates

Return ONLY valid JSON with keys:
bugs, complexity, suggestions, quality_score, improved_code
"""
    result = get_llm_review(test_prompt)
    print(json.dumps(result, indent=2))