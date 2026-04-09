"""
Hybrid retrieval: multi-query vector search + BM25, fused with RRF, then cross-encoder rerank.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from ingestion.rag_config import params_for_complexity
from ingestion.registry import get_game_complexity

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma_db")

_COST_ANCHOR = (
    "building costs requires resources brick lumber ore wool grain "
    "settlement city road development card trade"
)

_RECALL_PER_QUERY = int(os.environ.get("RAG_RECALL_PER_QUERY", "28"))
_RRF_K = int(os.environ.get("RAG_RRF_K", "60"))
_CROSS_ENCODER = os.environ.get(
    "RAG_CROSS_ENCODER",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
_STOP = frozenset(
    "the a an do does is are was were what how when where why which who can could should would "
    "i my we you it its this that there they them he she or and but if in on at to from of for "
    "be as by with".split()
)

_cross_encoder_model = None


def _expand_queries(query: str) -> list[str]:
    q = " ".join(query.split())
    if not q:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    add(q)
    for sep in (". ", "? ", "! "):
        if sep in q:
            head = q.split(sep)[0].strip()
            if len(head) > 12:
                add(head)
            break
    add(f"{q} {_COST_ANCHOR}".strip())
    words = [w for w in q.lower().split() if w.strip(".,;:!?\"'()[]") not in _STOP]
    if len(words) >= 3:
        add(" ".join(words))
    return out


def _rrf(rank_lists: Sequence[Sequence[str]], k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        if not ranks:
            continue
        for i, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)
    return scores


def _vector_ranked_ids(
    collection,
    query_text: str,
    where: dict,
    n_results: int,
) -> list[str]:
    if n_results < 1:
        return []
    r = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
    )
    ids = (r.get("ids") or [[]])[0]
    return [i for i in ids if i]


def _bm25_ranked_ids(
    bm25: BM25Okapi,
    doc_ids: list[str],
    query_text: str,
    top_n: int,
) -> list[str]:
    if top_n < 1 or not doc_ids:
        return []
    q_tokens = query_text.lower().split()
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [doc_ids[i] for i in ranked_idx]


def _get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder_model = CrossEncoder(_CROSS_ENCODER)
    return _cross_encoder_model


def _rerank_cross_encoder(query: str, id_to_doc: dict[str, str], candidate_ids: list[str], top_n: int) -> list[str]:
    if top_n < 1 or not candidate_ids:
        return []
    if os.environ.get("RAG_NO_CROSS_ENCODER", "").lower() in ("1", "true", "yes"):
        return [id_to_doc[i] for i in candidate_ids[:top_n] if i in id_to_doc]
    try:
        ce = _get_cross_encoder()
        docs = [id_to_doc[i] for i in candidate_ids if i in id_to_doc]
        if not docs:
            return []
        pairs = [[query, d] for d in docs]
        try:
            scores = ce.predict(pairs, batch_size=16, show_progress_bar=False)
        except TypeError:
            scores = ce.predict(pairs, batch_size=16)
        order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:top_n]
        return [docs[j] for j in order]
    except Exception as e:
        print(f"Cross-encoder rerank skipped: {e}")
        return [id_to_doc[i] for i in candidate_ids[:top_n] if i in id_to_doc]


def search_rulebook(query: str, game_name: str, top_k: int | None = None) -> list[str]:
    """
    Multi-query dense + BM25 → RRF fusion → cross-encoder rerank → top_k rulebook strings.
    No extra LLM calls; cross-encoder and BM25 are local.
    """
    query = (query or "").strip()
    if not query:
        return []

    if top_k is None:
        top_k = params_for_complexity(get_game_complexity(game_name))["top_k"]
    top_k = max(1, int(top_k))

    client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    try:
        collection = client.get_collection(name="board_game_rules", embedding_function=ef)
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return []

    where = {"game_name": game_name}
    got = collection.get(where=where, include=["documents"])
    doc_ids: list[str] = list(got.get("ids") or [])
    documents: list[str] = list(got.get("documents") or [])
    if not doc_ids or not documents:
        return []

    id_to_doc = dict(zip(doc_ids, documents))
    tokenized = [(d.lower().split() if (d or "").strip() else ["_"]) for d in documents]
    bm25 = BM25Okapi(tokenized)

    n_mem = len(doc_ids)
    n_vec = min(_RECALL_PER_QUERY, n_mem)

    queries = _expand_queries(query)
    rank_lists: list[list[str]] = []
    for q in queries:
        rank_lists.append(_vector_ranked_ids(collection, q, where, n_vec))
        rank_lists.append(_bm25_ranked_ids(bm25, doc_ids, q, n_vec))

    fused = _rrf(rank_lists, k=_RRF_K)
    if not fused:
        return []

    ranked = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)
    pool_size = min(len(ranked), max(top_k * 4, 48), 120)

    return _rerank_cross_encoder(query, id_to_doc, ranked[:pool_size], top_k)
