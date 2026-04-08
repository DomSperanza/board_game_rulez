"""Track ingested rulebooks so the same game or PDF file is not added twice."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = ROOT / "data" / "ingestion_registry.json"


def _game_key(name: str) -> str:
    return " ".join(name.strip().split()).casefold()


def _load() -> dict:
    if not REGISTRY_PATH.is_file():
        return {"games": {}}
    try:
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"games": {}}
    if "games" not in data:
        data["games"] = {}
    return data


def _save(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def check_can_ingest(display_name: str, pdf_sha256: str) -> str | None:
    """Return error message if ingest blocked; else None."""
    name = " ".join(display_name.strip().split())
    if not name:
        return "Enter a name for this game."
    data = _load()
    games: dict = data["games"]
    key = _game_key(name)
    for gid, meta in games.items():
        if gid == key:
            return f"A rulebook for “{meta['display_name']}” is already in your library."
        prev = meta.get("sha256") or ""
        if prev and prev == pdf_sha256:
            return f"This PDF was already added as “{meta['display_name']}”."
    return None


def register(
    display_name: str,
    pdf_sha256: str,
    source_filename: str,
    thumbnail: str | None = None,
) -> None:
    name = " ".join(display_name.strip().split())
    data = _load()
    entry = {
        "display_name": name,
        "sha256": pdf_sha256,
        "source_filename": source_filename,
    }
    if thumbnail:
        entry["thumbnail"] = thumbnail
    data["games"][_game_key(name)] = entry
    _save(data)


def game_exists(display_name: str) -> bool:
    name = " ".join(display_name.strip().split())
    return _game_key(name) in _load()["games"]


def pop_game(display_name: str) -> dict | None:
    """Remove game from registry; return removed entry or None."""
    name = " ".join(display_name.strip().split())
    key = _game_key(name)
    data = _load()
    entry = data["games"].pop(key, None)
    if entry:
        _save(data)
    return entry


def list_registered_games() -> list[str]:
    data = _load()
    return sorted(m["display_name"] for m in data["games"].values())


def list_library_games() -> list[dict]:
    """Sorted list of {display_name, thumbnail} for UI (thumbnail is basename or None)."""
    data = _load()
    rows = []
    for m in data["games"].values():
        rows.append(
            {
                "display_name": m["display_name"],
                "thumbnail": m.get("thumbnail") or None,
            }
        )
    return sorted(rows, key=lambda x: x["display_name"].casefold())


def sync_from_chroma_if_registry_empty() -> None:
    """If Chroma already has data but registry is new, copy game names (no hash dedupe until re-upload)."""
    data = _load()
    if data["games"]:
        return
    names = _unique_game_names_from_chroma()
    if not names:
        return
    for n in names:
        data["games"][_game_key(n)] = {
            "display_name": n,
            "sha256": "",
            "source_filename": "(existing database)",
        }
    _save(data)


def _unique_game_names_from_chroma() -> list[str]:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions

    db_path = str(ROOT / "data" / "chroma_db")
    if not os.path.isdir(db_path):
        return []
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        col = client.get_collection(name="board_game_rules", embedding_function=ef)
    except Exception:
        return []
    raw = col.get(include=["metadatas"])
    seen: set[str] = set()
    out: list[str] = []
    for m in raw.get("metadatas") or []:
        if not m:
            continue
        g = m.get("game_name")
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return sorted(out)
