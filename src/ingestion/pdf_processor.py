"""
Module: pdf_processor
Purpose:
- Load PDF files from the data directory.
- Extract raw text from the rulebooks.
- Clean and split the text into manageable "chunks" (e.g., small paragraphs).
"""
import re

import fitz  # PyMuPDF

def extract_text(pdf_path: str) -> str:
    """Extracts raw text from a PDF."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return ""
    
    full_text = []
    for page in doc:
        full_text.append(page.get_text())
        
    return "\n".join(full_text)

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
