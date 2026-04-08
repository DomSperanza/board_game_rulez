"""Map rulebook complexity (1=simple, 10=dense) to chunking and retrieval."""


def params_for_complexity(complexity: int) -> dict:
    """Higher complexity → smaller chunks, more overlap, more passages at query time."""
    c = max(1, min(10, int(complexity)))
    t = (c - 1) / 9.0
    chunk_size = int(round(1900 - 950 * t))
    overlap = int(round(280 + 240 * t))
    top_k = int(round(5 + 9 * t))
    return {"chunk_size": chunk_size, "overlap": overlap, "top_k": top_k}
