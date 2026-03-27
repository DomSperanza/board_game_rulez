"""Render PDF first page as a small cover image for the library UI."""
from __future__ import annotations

import hashlib
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent.parent
THUMB_DIR = ROOT / "data" / "game_thumbnails"

# Output: short deterministic name, avoids unsafe characters in game titles
MAX_EDGE_PX = 420
JPEG_QUALITY = 82


def thumbnail_basename(display_name: str) -> str:
    key = " ".join(display_name.strip().split()).casefold()
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return f"{h}.jpg"


def render_first_page_thumbnail(pdf_path: str, display_name: str) -> str | None:
    """Write JPEG of page 1. Returns basename on success."""
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    name = " ".join(display_name.strip().split())
    out = THUMB_DIR / thumbnail_basename(name)
    doc = None
    try:
        doc = fitz.open(pdf_path)
        if len(doc) < 1:
            return None
        page = doc[0]
        rect = page.rect
        if rect.width < 1 or rect.height < 1:
            return None
        zoom = min(MAX_EDGE_PX / rect.width, MAX_EDGE_PX / rect.height, 2.5)
        zoom = max(zoom, 0.5)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(out), output="jpg", jpg_quality=JPEG_QUALITY)
        return out.name
    except Exception:
        try:
            if out.is_file():
                out.unlink()
        except OSError:
            pass
        return None
    finally:
        if doc is not None:
            doc.close()
