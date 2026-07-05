from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import settings


# ── Singleton model ──────────────────────────────────────────────────────────
# Loading a HuggingFace model is expensive (~1–3 seconds + memory).
# We load it once at module import time and reuse across all requests.
# This is a simple version of the "expensive resource singleton" pattern
# you'll see in production services.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
    return _model


# ── Public API ───────────────────────────────────────────────────────────────

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of text strings into embedding vectors.

    Why batch instead of one-at-a-time?
    Embedding models are much faster when fed multiple texts at once
    because the GPU/CPU can process them in parallel.
    Always batch when ingesting documents.
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.
    Used at query time — only ever one string, so no batching needed.
    """
    return embed_texts([query])[0]