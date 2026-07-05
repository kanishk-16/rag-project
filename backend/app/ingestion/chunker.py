from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks using a sliding window.

    Why word-level (not character-level)?
    Character splits can cut a word in half mid-token, which confuses embedding models.
    Word-level splits are cleaner and chunk_size is easier to reason about.

    Why overlap?
    A sentence that spans two chunk boundaries would lose context without overlap.
    The tail of chunk N repeats at the head of chunk N+1, so meaning is preserved.

    Example (chunk_size=5, overlap=2):
        Input:  [A B C D E F G H]
        Chunk1: [A B C D E]
        Chunk2: [D E F G H]   <- D E repeated from chunk1

    In Phase 2 you'll experiment with different chunk_size/overlap values
    and measure their effect on Recall@K using your eval set.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Move forward by (chunk_size - overlap) so the next chunk
        # starts `overlap` words before where this one ended
        start += chunk_size - overlap

    return [c for c in chunks if c.strip()]  # drop any accidental empty chunks