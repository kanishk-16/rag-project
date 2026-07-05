import os
import json
import numpy as np
from typing import List, Dict

# ── Storage paths ────────────────────────────────────────────────────────────
STORE_DIR = "./vector_store"
CHUNKS_FILE = os.path.join(STORE_DIR, "chunks.json")
EMBEDDINGS_FILE = os.path.join(STORE_DIR, "embeddings.npy")

os.makedirs(STORE_DIR, exist_ok=True)


# ── Load persisted data at startup ───────────────────────────────────────────
def _load_store():
    """Load chunks and embeddings from disk if they exist."""
    if os.path.exists(CHUNKS_FILE) and os.path.exists(EMBEDDINGS_FILE):
        with open(CHUNKS_FILE, "r") as f:
            chunks = json.load(f)
        embeddings = np.load(EMBEDDINGS_FILE)
        return chunks, embeddings
    return [], np.empty((0,), dtype=np.float32)

_chunks, _embeddings = _load_store()


def _save_store():
    """Persist chunks and embeddings to disk."""
    with open(CHUNKS_FILE, "w") as f:
        json.dump(_chunks, f)
    np.save(EMBEDDINGS_FILE, _embeddings)


# ── Public API ───────────────────────────────────────────────────────────────

def store_chunks(
    chunks: List[str],
    embeddings: List[List[float]],
    doc_id: str,
) -> None:
    """
    Store text chunks and their embedding vectors.

    Data is kept in two parallel structures:
    - _chunks: list of dicts with text + metadata
    - _embeddings: numpy array of shape (N, embedding_dim)

    Why parallel? So we can do fast numpy math on the embeddings
    while keeping the original text accessible by index.
    """
    global _chunks, _embeddings

    new_embeddings = np.array(embeddings, dtype=np.float32)

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        _chunks.append({
            "text": chunk,
            "metadata": {"doc_id": doc_id, "chunk_index": i}
        })

    if _embeddings.shape[0] == 0:
        _embeddings = new_embeddings
    else:
        _embeddings = np.vstack([_embeddings, new_embeddings])

    _save_store()
    print(f"[VectorStore] Stored {len(chunks)} chunks for doc_id={doc_id}. Total: {len(_chunks)}")


def retrieve_top_k(query_embedding: List[float], top_k: int = 5) -> List[Dict]:
    """
    Find the top-k most similar chunks using cosine similarity.

    Cosine similarity formula:
        similarity = (A · B) / (||A|| × ||B||)

    Why cosine and not euclidean distance?
    Cosine similarity ignores vector magnitude — it only looks at direction.
    This means a short sentence and a long paragraph covering the same topic
    will still score high similarity, which is what we want for text retrieval.

    Step-by-step:
    1. Normalize the query vector to unit length
    2. Normalize all stored vectors to unit length
    3. Dot product of normalized vectors = cosine similarity
    4. Pick the indices with the highest scores
    """
    if len(_chunks) == 0:
        return []

    query = np.array(query_embedding, dtype=np.float32)

    # Normalize query vector
    query_norm = query / (np.linalg.norm(query) + 1e-10)

    # Normalize all stored embeddings (row-wise)
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True) + 1e-10
    normed_embeddings = _embeddings / norms

    # Cosine similarity = dot product of normalized vectors
    similarities = normed_embeddings @ query_norm

    # Get top-k indices (sorted descending)
    k = min(top_k, len(_chunks))
    top_indices = np.argsort(similarities)[::-1][:k]

    results = []
    for idx in top_indices:
        results.append({
            "text": _chunks[idx]["text"],
            "metadata": _chunks[idx]["metadata"],
            "score": round(float(similarities[idx]), 4),
        })

    return results