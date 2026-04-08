"""Shared ingest path for CLI and Flask."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ingestion.embedder import create_embeddings_and_store, delete_embeddings_for_game
from ingestion.pdf_processor import chunk_text, extract_text
from ingestion.registry import check_can_ingest, file_sha256, game_exists, pop_game, register
from ingestion.thumbnail import THUMB_DIR, render_first_page_thumbnail


def ingest_pdf_file(pdf_path: str, game_display_name: str) -> tuple[bool, str]:
    """
    Returns (success, message). Message is error text or a short success note.
    """
    if not os.path.isfile(pdf_path):
        return False, "File not found."

    digest = file_sha256(pdf_path)
    err = check_can_ingest(game_display_name, digest)
    if err:
        return False, err

    name = " ".join(game_display_name.strip().split())
    text = extract_text(pdf_path)
    if not text:
        return False, "Could not read text from this PDF."

    chunks = chunk_text(text)
    if not chunks:
        return False, "No text chunks produced from this PDF."

    create_embeddings_and_store(chunks, name)
    thumb = render_first_page_thumbnail(pdf_path, name)
    register(name, digest, Path(pdf_path).name, thumbnail=thumb)
    return True, f"Added “{name}” ({len(chunks)} sections indexed)."


def ingest_uploaded_pdf(file_storage, game_display_name: str) -> tuple[bool, str]:
    if not file_storage or not getattr(file_storage, "filename", None):
        return False, "Choose a PDF to upload."
    if Path(file_storage.filename).suffix.lower() != ".pdf":
        return False, "Only PDF files are supported."
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        file_storage.save(tmp_path)
        return ingest_pdf_file(tmp_path, game_display_name)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def delete_game_completely(display_name: str) -> tuple[bool, str]:
    name = " ".join(display_name.strip().split())
    if not name:
        return False, "Pick a game to remove."
    if not game_exists(name):
        return False, f"No game named “{name}” in the library."
    try:
        delete_embeddings_for_game(name)
    except Exception as e:
        return False, f"Could not remove indexed data ({e})."
    entry = pop_game(name)
    if not entry:
        return False, f"No game named “{name}” in the library."
    tb = entry.get("thumbnail")
    if tb:
        try:
            (THUMB_DIR / tb).unlink(missing_ok=True)
        except OSError:
            pass
    return True, f"Removed “{name}” from your library."
