from typing import List, Dict, Union
import fitz  # PyMuPDF — faster and more accurate than pdfplumber for plain text


def load_pdf(file_path: str) -> List[Dict[str, Union[int, str]]]:
    """
    Extract raw text from every page of a PDF alongside page numbers.

    Why PyMuPDF (fitz)?
    - Handles most PDF encodings reliably
    - Very fast — processes a 100-page doc in under a second
    - Returns clean text per page with 1-based page numbers for citation tracking.

    Returns:
        A list of dicts with page number (1-indexed) and extracted text:
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}]
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():  # skip blank/image-only pages
            pages.append({"page_num": page_num, "text": text})

    doc.close()
    return pages
