import fitz  # PyMuPDF — faster and more accurate than pdfplumber for plain text


def load_pdf(file_path: str) -> str:
    """
    Extract raw text from every page of a PDF.

    Why PyMuPDF (fitz)?
    - Handles most PDF encodings reliably
    - Very fast — processes a 100-page doc in under a second
    - Returns clean text per page so we can attach page numbers later (Phase 2)

    Returns the full document text as a single string.
    """
    doc = fitz.open(file_path)
    pages_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # skip blank/image-only pages
            pages_text.append(text)

    doc.close()
    return "\n".join(pages_text)