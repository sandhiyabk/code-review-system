# core/chroma_store.py

import chromadb
from typing import List
from data.style_rules import STYLE_RULES


class StyleRuleStore:
    """
    Stores Python style rules as vector embeddings
    in ChromaDB for semantic retrieval.
    """

    def __init__(self):
        # Persistent client — data survives restarts
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="python_style_rules",
            metadata={"hnsw:space": "cosine"}
        )
        self._load_rules()

    def _load_rules(self):
        """Load style rules if collection is empty."""
        existing = self.collection.count()
        if existing == 0:
            print(f"Loading {len(STYLE_RULES)} style rules...")
            self.collection.add(
                documents=STYLE_RULES,
                ids=[f"rule_{i:03d}" for i in range(len(STYLE_RULES))]
            )
            print("Rules loaded successfully.")
        else:
            print(f"{existing} rules already loaded.")

    def get_relevant_rules(
        self,
        code: str,
        n_results: int = 4
    ) -> List[str]:
        """
        Find style rules most relevant to the given code
        using semantic similarity search.
        """
        results = self.collection.query(
            query_texts=[code],
            n_results=min(n_results, len(STYLE_RULES)),
            include=["documents", "distances"]
        )

        relevant = []
        for doc, dist in zip(
            results["documents"][0],
            results["distances"][0]
        ):
            # Only return rules with meaningful relevance
            # Distance < 0.7 means reasonably similar
            if dist < 0.7:
                relevant.append(doc)

        return relevant


# Test it directly
if __name__ == "__main__":
    store = StyleRuleStore()

    test_queries = [
        "for i in range(len(arr)): for j in range(len(arr)):",
        "x = 1\ny = 2\nz = x + y",
        "def f(a): return a * 2",
    ]

    for query in test_queries:
        print(f"\nCode: {query[:50]}...")
        rules = store.get_relevant_rules(query)
        print(f"Relevant rules found: {len(rules)}")
        for r in rules:
            print(f"  → {r[:60]}...")