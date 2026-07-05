"""
Tests for the chunker module.

Run with: pytest tests/test_chunker.py -v
"""
from app.ingestion.chunker import chunk_text


def test_basic_chunking():
    """Chunks should split text and each chunk should be non-empty."""
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    assert all(len(c.strip()) > 0 for c in chunks)


def test_overlap_exists():
    """The end of chunk N should overlap with the start of chunk N+1."""
    text = " ".join([f"word{i}" for i in range(50)])
    chunks = chunk_text(text, chunk_size=10, overlap=3)

    # Last 3 words of chunk 0 should appear at the start of chunk 1
    tail_words = chunks[0].split()[-3:]
    head_words = chunks[1].split()[:3]
    assert tail_words == head_words


def test_short_text_single_chunk():
    """Text shorter than chunk_size should produce exactly one chunk."""
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1


def test_no_empty_chunks():
    """No chunk should be empty or whitespace-only."""
    text = " ".join(["word"] * 200)
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    assert all(c.strip() for c in chunks)