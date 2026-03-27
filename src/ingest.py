"""
Module: ingest
Purpose:
CLI tool for ingesting PDFs into the ChromaDB vector database.

Usage:
    python src/ingest.py --pdf data/raw_pdfs/catan_base_rules_2020_200707.pdf --game "Settlers of Catan"
"""
import argparse
import sys
import os

# Add the src directory to Python path so absolute imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.pipeline import ingest_pdf_file
from ingestion.registry import sync_from_chroma_if_registry_empty


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF rulebooks into Board Game Rulez database.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file.")
    parser.add_argument("--game", required=True, help="Name of the board game.")

    args = parser.parse_args()

    sync_from_chroma_if_registry_empty()

    print(f"Reading {args.pdf}...")
    ok, msg = ingest_pdf_file(args.pdf, args.game)
    if ok:
        print(msg)
        return
    print(f"Error: {msg}")
    sys.exit(1)

if __name__ == "__main__":
    main()
