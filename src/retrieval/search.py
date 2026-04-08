"""
Module: search
Purpose:
- Take user queries (e.g., "How do I upgrade a settlement?")
- Convert the query into a vector embedding.
- Search ChromaDB for the most contextually relevant chunks from the rulebook.
- Return the top N chunks to feed to the LLM.
"""
import os

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ingestion.rag_config import params_for_complexity
from ingestion.registry import get_game_complexity

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma_db")

# Second pass steers toward "building costs" / Requires: lines that pure semantic search often misses.
_COST_ANCHOR = (
    "building costs requires resources brick lumber ore wool grain "
    "settlement city road development card trade"
)


def _merge_results(
    primary: dict,
    secondary: dict,
    max_total: int,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for bucket in (primary, secondary):
        docs = bucket.get("documents") or []
        ids = bucket.get("ids") or []
        if not docs or not docs[0]:
            continue
        for i, doc_id in enumerate(ids[0]):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(docs[0][i])
            if len(out) >= max_total:
                return out
    return out


def search_rulebook(query: str, game_name: str, top_k: int | None = None) -> list[str]:
    """Retrieves merged vector hits so building-cost sections are not ranked out."""
    if top_k is None:
        top_k = params_for_complexity(get_game_complexity(game_name))["top_k"]
    client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        collection = client.get_collection(
            name="board_game_rules",
            embedding_function=sentence_transformer_ef,
        )
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return []

    where = {"game_name": game_name}
    n_fetch = max(8, top_k)

    r1 = collection.query(
        query_texts=[query],
        n_results=n_fetch,
        where=where,
    )
    r2 = collection.query(
        query_texts=[f"{query} {_COST_ANCHOR}"],
        n_results=n_fetch,
        where=where,
    )

    merged = _merge_results(r1, r2, max_total=top_k)
    return merged
