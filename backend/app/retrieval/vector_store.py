import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import numpy as np
from rank_bm25 import BM25Okapi

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


# ── BM25 Index ───────────────────────────────────────────────────────────────
# BM25 is an exact/lexical matching algorithm based on term frequency and inverse
# document frequency. We maintain an in-memory index over whitespace-tokenized chunks.
def _build_bm25_index(chunks: List[Dict[str, Any]]) -> Optional[BM25Okapi]:
    """
    Build a BM25Okapi index from a list of chunk dictionaries.
    Tokenizes each chunk's text by whitespace.
    """
    if not chunks:
        return None
    tokenized_corpus = [chunk["text"].split() for chunk in chunks]
    return BM25Okapi(tokenized_corpus)

# Initialize BM25 index on startup from any loaded chunks
_bm25_index: Optional[BM25Okapi] = _build_bm25_index(_chunks)


# ── Cross-Encoder Singleton ──────────────────────────────────────────────────
# Loading a cross-encoder model can be computationally heavy and slow.
# We use lazy initialization to avoid overhead during import time and load it once
# only when first needed for reranking.
_cross_encoder_model= None

def _get_cross_encoder():
    """Lazily load the cross-encoder reranker singleton."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder
        print("[VectorStore] Loading CrossEncoder: cross-encoder/ms-marco-MiniLM-L-6-v2")
        _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder_model


# ── Score Normalization Helper ───────────────────────────────────────────────
def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalize score arrays into a [0, 1] range using min-max scaling.
    If all scores are identical, returns an array of zeros to avoid division by zero.
    """
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - min_val) / (max_val - min_val)


# ── Public API ───────────────────────────────────────────────────────────────

def store_chunks(
    chunks: Union[List[Dict[str, Any]], List[str]],
    embeddings: List[List[float]],
    doc_id: str,
    filename: Optional[str] = None,
    upload_time: Optional[Union[datetime, str]] = None,
) -> None:
    """
    Store text chunks, their embedding vectors, and metadata (page numbers, doc_id, filename, timestamp).

    Data is kept in two parallel structures:
    - _chunks: list of dicts with text + metadata (doc_id, chunk_index, page_num, filename, upload_time)
    - _embeddings: numpy array of shape (N, embedding_dim)

    Why parallel? So we can do fast numpy math on the embeddings
    while keeping the original text and metadata accessible by index.
    """
    global _chunks, _embeddings, _bm25_index

    new_embeddings = np.array(embeddings, dtype=np.float32)

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            page_num = chunk.get("page_num", chunk.get("page", 1))
        else:
            text = str(chunk)
            page_num = 1

        meta: Dict[str, Any] = {
            "doc_id": doc_id,
            "chunk_index": i,
            "page_num": page_num,
            "filename" : filename or "unknown",
            "upload_time" : upload_time.isoformat() if isinstance(upload_time, datetime) else str(upload_time or ""),
        }
        _chunks.append({
            "text": text,
            "metadata": meta,
        })

    if _embeddings.shape[0] == 0:
        _embeddings = new_embeddings
    else:
        _embeddings = np.vstack([_embeddings, new_embeddings])

    _save_store()

    # Rebuild the BM25 index from all stored chunk texts whenever new chunks are added
    _bm25_index = _build_bm25_index(_chunks)

    print(f"[VectorStore] Stored {len(chunks)} chunks for doc_id={doc_id}. Total: {len(_chunks)}")


def retrieve_top_k(
    query_embedding: List[float],
    top_k: int = 5,
    query_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Find the top-k most relevant chunks using hybrid retrieval and cross-encoder reranking.

    Why hybrid retrieval?
    - Semantic search captures synonyms and high-level conceptual similarity, but can miss
      exact keyword matches like acronyms, identifiers, or proper nouns.
    - BM25 captures exact term matches effectively but misses semantic paraphrasing.
    - Combining both provides higher candidate recall than either technique alone.

    Why cross-encoder reranking?
    - Bi-encoders (used for embeddings) encode queries and passages independently for fast retrieval.
    - Cross-encoders feed the query and passage together through attention layers, capturing deep
      token-level interactions and providing far more accurate relevance scoring for the top candidates.

    Pipeline Steps:
    1. Compute cosine similarity between query embedding and all chunk embeddings.
    2. Compute BM25 scores between tokenized query text and all chunks.
    3. Normalize both semantic and BM25 scores to [0, 1] using min-max normalization.
    4. Calculate combined hybrid scores: 0.7 * semantic + 0.3 * BM25.
    5. Retrieve top-20 candidates based on hybrid scores.
    6. Rerank the top-20 candidates using the CrossEncoder model.
    7. Return the top top_k reranked results.
    """
    if len(_chunks) == 0:
        return []

    # 1. Semantic Similarity Scores (Cosine Similarity)
    query = np.array(query_embedding, dtype=np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True) + 1e-10
    normed_embeddings = _embeddings / norms
    semantic_scores = (normed_embeddings @ query_norm).astype(np.float32)

    # 2. BM25 Keyword Matching Scores
    # If query_text is provided and BM25 index exists, score against whitespace-tokenized query
    if query_text.strip() and _bm25_index is not None:
        tokenized_query = query_text.split()
        bm25_scores = np.array(_bm25_index.get_scores(tokenized_query), dtype=np.float32)
    else:
        # Fallback to zero BM25 scores if no query string is provided
        bm25_scores = np.zeros(len(_chunks), dtype=np.float32)

    # 3. Min-Max Score Normalization
    # Rescaling both scores to [0, 1] prevents one modality with different scale from dominating
    norm_semantic = _min_max_normalize(semantic_scores)
    norm_bm25 = _min_max_normalize(bm25_scores)

    # 4. Hybrid Score Combination (70% semantic, 30% lexical)
    hybrid_scores = 0.7 * norm_semantic + 0.3 * norm_bm25

    # 5. Retrieve top-20 initial candidate indices
    candidate_k = min(20, len(_chunks))
    candidate_indices = np.argsort(hybrid_scores)[::-1][:candidate_k]

    # 6. Cross-Encoder Reranking
    # If query_text is available, rerank candidates using joint attention scoring
    if query_text.strip():
        cross_encoder = _get_cross_encoder()
        pairs = [(query_text, _chunks[idx]["text"]) for idx in candidate_indices]
        rerank_scores = cross_encoder.predict(pairs)

        # Pair candidate index with its cross-encoder score and sort descending
        scored_candidates = sorted(
            zip(candidate_indices, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )
    else:
        # If no query_text is supplied, preserve hybrid score ordering
        scored_candidates = [(idx, hybrid_scores[idx]) for idx in candidate_indices]

    # 7. Select top top_k results
    final_k = min(top_k, len(scored_candidates))
    top_candidates = scored_candidates[:final_k]

    results = []
    for idx, score in top_candidates:
        results.append({
            "text": _chunks[idx]["text"],
            "metadata": _chunks[idx]["metadata"],
            "score": round(float(score), 4),
        })

    return results
