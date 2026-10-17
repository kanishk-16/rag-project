import os
import json
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import numpy as np
from rank_bm25 import BM25Okapi

from app.core.config import settings

# ── Storage paths ────────────────────────────────────────────────────────────
STORE_DIR = settings.store_dir
CHUNKS_FILE = os.path.join(STORE_DIR, "chunks.json")
EMBEDDINGS_FILE = os.path.join(STORE_DIR, "embeddings.npy")

os.makedirs(STORE_DIR, exist_ok=True)
_store_lock = threading.Lock()


# ── Load persisted data at startup ───────────────────────────────────────────
def _load_store():
    """Load chunks and embeddings from disk if they exist."""
    if os.path.exists(CHUNKS_FILE) and os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            embeddings = np.load(EMBEDDINGS_FILE)
            if len(chunks) == len(embeddings):
                return chunks, embeddings
            print(f"[VectorStore] Warning: chunks count ({len(chunks)}) != embeddings count ({len(embeddings)}). Resetting store.")
        except Exception as e:
            print(f"[VectorStore] Error loading store: {e}")
    return [], np.empty((0,), dtype=np.float32)

_chunks, _embeddings = _load_store()


def _save_store():
    """Persist chunks and embeddings to disk atomically."""
    tmp_chunks = f"{CHUNKS_FILE}.tmp"
    tmp_embeddings = f"{EMBEDDINGS_FILE}.tmp"
    with open(tmp_chunks, "w", encoding="utf-8") as f:
        json.dump(_chunks, f)
    np.save(tmp_embeddings, _embeddings)
    os.replace(tmp_chunks, CHUNKS_FILE)
    os.replace(f"{tmp_embeddings}.npy" if not tmp_embeddings.endswith(".npy") and os.path.exists(f"{tmp_embeddings}.npy") else tmp_embeddings, EMBEDDINGS_FILE)


# ── BM25 Index ───────────────────────────────────────────────────────────────
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
_cross_encoder_model = None

def _get_cross_encoder():
    """Lazily load the cross-encoder reranker singleton."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder
        print(f"[VectorStore] Loading CrossEncoder: {settings.cross_encoder_model}")
        _cross_encoder_model = CrossEncoder(settings.cross_encoder_model)
    return _cross_encoder_model


# ── Score Normalization Helper ───────────────────────────────────────────────
def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalize score arrays into a [0, 1] range using min-max scaling.
    If all scores are identical, returns an array of zeros to avoid division by zero.
    """
    if len(scores) == 0:
        return np.empty((0,), dtype=np.float32)
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
    user_id: Optional[str] = None,
) -> None:
    """
    Store text chunks, their embedding vectors, and metadata.
    Thread-safe and supports multi-tenancy with user_id.
    """
    global _chunks, _embeddings, _bm25_index

    new_embeddings = np.array(embeddings, dtype=np.float32)

    with _store_lock:
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
                "filename": filename or "unknown",
                "upload_time": upload_time.isoformat() if isinstance(upload_time, datetime) else str(upload_time or ""),
                "user_id": user_id,
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
        _bm25_index = _build_bm25_index(_chunks)

    print(f"[VectorStore] Stored {len(chunks)} chunks for doc_id={doc_id} (user_id={user_id}). Total: {len(_chunks)}")


def retrieve_top_k(
    query_embedding: List[float],
    top_k: int = 5,
    query_text: str = "",
    user_id: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Find top-k relevant chunks using hybrid retrieval and cross-encoder reranking.
    Supports multi-tenant filtering by user_id and document filtering by doc_id.
    """
    if len(_chunks) == 0:
        return []

    # Filter indices based on tenant and optional document filter
    if user_id is not None or doc_id is not None:
        target_indices = [
            i for i, c in enumerate(_chunks)
            if (user_id is None or c.get("metadata", {}).get("user_id") == user_id)
            and (doc_id is None or c.get("metadata", {}).get("doc_id") == doc_id)
        ]
        if not target_indices:
            return []
    else:
        target_indices = list(range(len(_chunks)))

    target_embeddings = _embeddings[target_indices]

    # 1. Semantic Similarity Scores (Cosine Similarity)
    query = np.array(query_embedding, dtype=np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(target_embeddings, axis=1, keepdims=True) + 1e-10
    normed_embeddings = target_embeddings / norms
    semantic_scores = (normed_embeddings @ query_norm).astype(np.float32)

    # 2. BM25 Keyword Matching Scores
    if query_text.strip():
        # If filtering subset or using global index
        if len(target_indices) == len(_chunks) and _bm25_index is not None:
            tokenized_query = query_text.split()
            all_bm25 = np.array(_bm25_index.get_scores(tokenized_query), dtype=np.float32)
            bm25_scores = all_bm25[target_indices]
        else:
            subset_chunks = [_chunks[i] for i in target_indices]
            sub_index = _build_bm25_index(subset_chunks)
            tokenized_query = query_text.split()
            bm25_scores = np.array(sub_index.get_scores(tokenized_query), dtype=np.float32) if sub_index else np.zeros(len(target_indices), dtype=np.float32)
    else:
        bm25_scores = np.zeros(len(target_indices), dtype=np.float32)

    # 3. Min-Max Score Normalization
    norm_semantic = _min_max_normalize(semantic_scores)
    norm_bm25 = _min_max_normalize(bm25_scores)

    # 4. Hybrid Score Combination
    w_sem = settings.hybrid_semantic_weight
    w_bm25 = settings.hybrid_bm25_weight
    hybrid_scores = w_sem * norm_semantic + w_bm25 * norm_bm25

    # 5. Retrieve top candidate indices from the target subset
    candidate_limit = min(settings.cross_encoder_candidates, len(target_indices))
    top_subset_positions = np.argsort(hybrid_scores)[::-1][:candidate_limit]
    candidate_indices = [target_indices[p] for p in top_subset_positions]

    # 6. Cross-Encoder Reranking
    if query_text.strip():
        cross_encoder = _get_cross_encoder()
        pairs = [(query_text, _chunks[idx]["text"]) for idx in candidate_indices]
        rerank_scores = cross_encoder.predict(pairs)

        scored_candidates = sorted(
            zip(candidate_indices, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )
    else:
        scored_candidates = [(idx, hybrid_scores[pos]) for pos, idx in zip(top_subset_positions, candidate_indices)]

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
