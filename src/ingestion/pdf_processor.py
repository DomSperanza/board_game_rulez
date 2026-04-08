"""
Module: pdf_processor
Purpose:
- Load PDF files from the data directory.
- Extract raw text from the rulebooks.
- Clean and split the text into manageable "chunks" (e.g., small paragraphs).
"""
import re

import io

import fitz  # PyMuPDF

_SPARSE_PAGE_CHARS = 80
# Skip tiny PDF image placements (icons / bullets); area in PDF points²
_MIN_IMAGE_AREA_PT2 = 3600


def _sparse_page_text(text: str) -> bool:
    return len(text.strip()) < _SPARSE_PAGE_CHARS


def _ocr_pixmap(pix) -> str:
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


def _ocr_page_text(page) -> str:
    try:
        pix = page.get_pixmap(dpi=144)
        return _ocr_pixmap(pix).strip()
    except Exception:
        return ""


def _ocr_embedded_images(page) -> list[str]:
    """OCR each on-page image rectangle (figures, diagrams); skips very small placements."""
    snippets: list[str] = []
    seen: set[tuple[float, float, float, float]] = set()
    try:
        for item in page.get_images(full=True):
            xref = item[0]
            for rect in page.get_image_rects(xref):
                if rect.get_area() < _MIN_IMAGE_AREA_PT2:
                    continue
                key = (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
                if key in seen:
                    continue
                seen.add(key)
                pix = page.get_pixmap(clip=rect, dpi=144)
                s = _ocr_pixmap(pix).strip()
                if len(s) > 3:
                    snippets.append(s)
    except Exception:
        pass
    return snippets


def _merge_text_and_figure_ocr(base: str, figure_snippets: list[str]) -> str:
    if not figure_snippets:
        return base
    block = "\n\n".join(figure_snippets)
    if not base:
        return block
    return f"{base}\n\n[Figure text]\n{block}"


def extract_text(pdf_path: str, ocr_sparse_pages: bool = False) -> str:
    """Extract PDF text. With OCR on: embedded images are OCR'd and appended; sparse pages can use full-page OCR."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return ""

    try:
        parts: list[str] = []
        for page in doc:
            t = page.get_text().strip()
            if ocr_sparse_pages:
                fig = _ocr_embedded_images(page)
                t = _merge_text_and_figure_ocr(t, fig)
                if _sparse_page_text(t):
                    ocr_t = _ocr_page_text(page)
                    if len(ocr_t) > len(t):
                        t = ocr_t
            parts.append(t)
        return "\n".join(parts)
    finally:
        doc.close()

def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 400) -> list[str]:
    """Splits text into chunks with overlap; defaults sized to keep rule subsections together."""
    if not text:
        return []

    parts = []
    for block in re.split(r"\n\s*\n+", text.strip()):
        line = " ".join(block.split())
        if line:
            parts.append(line)
    text = " ".join(parts)
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        
    return chunks
