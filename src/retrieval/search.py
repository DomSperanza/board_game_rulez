"""
Module: search
Purpose:
- Take user queries (e.g., "How do I upgrade a settlement?")
- Convert the query into a vector embedding.
- Search ChromaDB for the most contextually relevant chunks from the rulebook.
- Return the top N chunks to feed to the LLM.
"""


def search_rulebook(query: str, game_name: str, top_k: int = 3):
    pass
