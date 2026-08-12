from typing import List, Dict, Any, Union


def chunk_text(
    pages: Union[List[Dict[str, Any]], str],
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Split document pages into overlapping chunks using a sliding window,
    tracking the source page number for each chunk.

    Why word-level (not character-level)?
    Character splits can cut a word in half mid-token, which confuses embedding models.
    Word-level splits are cleaner and chunk_size is easier to reason about.

    Why overlap?
    A sentence that spans two chunk boundaries would lose context without overlap.
    The tail of chunk N repeats at the head of chunk N+1, so meaning is preserved.

    Example (chunk_size=5, overlap=2):
        Input Page 1: [A B C D E F G H]
        Chunk 1: {"text": "A B C D E", "page_num": 1}
        Chunk 2: {"text": "D E F G H", "page_num": 1}   <- D E repeated from chunk 1

    Returns:
        A list of dicts containing chunk text and page number:
        [
            {"text": "chunk content...", "page_num": 1},
            {"text": "chunk content...", "page_num": 1},
            {"text": "chunk content...", "page_num": 2},
            ...
        ]
    """
    if isinstance(pages, str):
        pages = [{"text": pages, "page_num": 1}]

    chunks: List[Dict[str, Any]] = []
    step = max(1, chunk_size - overlap)

    for page_dict in pages:
        text = page_dict.get("text", "")
        page_num = page_dict.get("page_num", 1)

        words = text.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_content = " ".join(words[start:end])
            if chunk_content.strip():
                chunks.append({
                    "text": chunk_content,
                    "page_num": page_num,
                })
            start += step

    return chunks
