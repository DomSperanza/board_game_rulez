"""Optional Tesseract OCR for PDF pages with little or no text layer."""


def ocr_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False
