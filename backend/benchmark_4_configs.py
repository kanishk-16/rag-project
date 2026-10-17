"""
Benchmark Runner measuring Recall, Hit Rate, and Latency (ms) across 4 configurations:
1. Dense only
2. BM25 only
3. Dense + BM25 + RRF
4. Dense + BM25 + RRF + Cross-Encoder
"""

import os
import sys
import json
import time
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from app.embeddings.embedder import embed_query
from app.retrieval.vector_store import (
    _chunks,
    _embeddings,
    _bm25_index,
    _get_cross_encoder,
)
from eval_runner import compute_context_recall, compute_hit_rate

EVAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval", "eval_set.json"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval", "results"))


def retrieve_dense_only(query_emb, top_k=5):
    query = np.array(query_emb, dtype=np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True) + 1e-10
    normed_embeddings = _embeddings / norms
    scores = (normed_embeddings @ query_norm).astype(np.float32)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"text": _chunks[idx]["text"], "metadata": _chunks[idx]["metadata"], "score": float(scores[idx])} for idx in top_indices]


def retrieve_bm25_only(query_text, top_k=5):
    if not query_text.strip() or _bm25_index is None:
        return []
    tokenized = query_text.split()
    scores = np.array(_bm25_index.get_scores(tokenized), dtype=np.float32)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"text": _chunks[idx]["text"], "metadata": _chunks[idx]["metadata"], "score": float(scores[idx])} for idx in top_indices]


def retrieve_rrf(query_emb, query_text, top_k=5, k=60):
    query = np.array(query_emb, dtype=np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True) + 1e-10
    normed_embeddings = _embeddings / norms
    dense_scores = (normed_embeddings @ query_norm).astype(np.float32)
    dense_ranked = np.argsort(dense_scores)[::-1]

    tokenized = query_text.split()
    bm25_scores = np.array(_bm25_index.get_scores(tokenized), dtype=np.float32)
    bm25_ranked = np.argsort(bm25_scores)[::-1]

    rrf_scores = defaultdict(float)
    for rank, idx in enumerate(dense_ranked):
        rrf_scores[idx] += 1.0 / (k + (rank + 1))
    for rank, idx in enumerate(bm25_ranked):
        rrf_scores[idx] += 1.0 / (k + (rank + 1))

    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    return [{"text": _chunks[idx]["text"], "metadata": _chunks[idx]["metadata"], "score": float(rrf_scores[idx])} for idx in sorted_indices]


def retrieve_rrf_cross_encoder(query_emb, query_text, top_k=5, candidate_k=20, k=60):
    query = np.array(query_emb, dtype=np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True) + 1e-10
    normed_embeddings = _embeddings / norms
    dense_scores = (normed_embeddings @ query_norm).astype(np.float32)
    dense_ranked = np.argsort(dense_scores)[::-1]

    tokenized = query_text.split()
    bm25_scores = np.array(_bm25_index.get_scores(tokenized), dtype=np.float32)
    bm25_ranked = np.argsort(bm25_scores)[::-1]

    rrf_scores = defaultdict(float)
    for rank, idx in enumerate(dense_ranked):
        rrf_scores[idx] += 1.0 / (k + (rank + 1))
    for rank, idx in enumerate(bm25_ranked):
        rrf_scores[idx] += 1.0 / (k + (rank + 1))

    candidates = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:candidate_k]

    cross_encoder = _get_cross_encoder()
    pairs = [(query_text, _chunks[idx]["text"]) for idx in candidates]
    ce_scores = cross_encoder.predict(pairs)

    scored = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"text": _chunks[idx]["text"], "metadata": _chunks[idx]["metadata"], "score": float(score)} for idx, score in scored]


def main():
    print(f"Loading eval set from: {EVAL_PATH}")
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    # Warm up cross encoder and embedder so first run doesn't skew latency numbers
    print("Warming up models...")
    _get_cross_encoder().predict([("test query", "test chunk")])
    _ = embed_query("test query")

    cached_queries = []
    print("Pre-embedding questions for pure retrieval timing (and separately tracking embedding latency)...")
    embed_latencies = []
    for item in eval_set:
        t0 = time.perf_counter()
        emb = embed_query(item["question"])
        embed_latencies.append((time.perf_counter() - t0) * 1000)
        cached_queries.append({"item": item, "emb": emb})

    avg_embed_latency_ms = float(np.mean(embed_latencies))
    print(f"Average query embedding latency: {avg_embed_latency_ms:.2f} ms")

    configs = [
        "1. Dense Only",
        "2. BM25 Only",
        "3. Dense + BM25 + RRF",
        "4. Dense + BM25 + RRF + Cross-Encoder",
    ]

    all_results = []
    for name in configs:
        print(f"\nRunning {name}...")
        total_recall = 0.0
        total_hits = 0
        retrieval_latencies = []
        end_to_end_latencies = []
        details = []

        for entry in cached_queries:
            item = entry["item"]
            emb = entry["emb"]
            q = item["question"]
            exp = item["expected_answer"]

            # Pure retrieval latency (using precomputed embedding for dense)
            t_start = time.perf_counter()
            if name.startswith("1."):
                chunks = retrieve_dense_only(emb, top_k=5)
            elif name.startswith("2."):
                chunks = retrieve_bm25_only(q, top_k=5)
            elif name.startswith("3."):
                chunks = retrieve_rrf(emb, q, top_k=5)
            else:
                chunks = retrieve_rrf_cross_encoder(emb, q, top_k=5)
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000

            # End-to-end includes query embedding time for configurations that use dense embeddings
            e2e_ms = t_elapsed_ms if name.startswith("2.") else (t_elapsed_ms + avg_embed_latency_ms)

            retrieval_latencies.append(t_elapsed_ms)
            end_to_end_latencies.append(e2e_ms)

            recall = compute_context_recall(exp, chunks)
            hit = compute_hit_rate(exp, chunks)
            total_recall += recall
            if hit:
                total_hits += 1

            details.append({
                "id": item["id"],
                "recall": recall,
                "hit": hit,
                "latency_retrieval_ms": round(t_elapsed_ms, 2),
                "latency_e2e_ms": round(e2e_ms, 2)
            })

        n = len(cached_queries)
        res = {
            "name": name,
            "mean_recall": round(total_recall / n, 4),
            "hit_rate": round(total_hits / n, 4),
            "hits": total_hits,
            "total": n,
            "avg_retrieval_latency_ms": round(float(np.mean(retrieval_latencies)), 2),
            "p95_retrieval_latency_ms": round(float(np.percentile(retrieval_latencies, 95)), 2),
            "avg_e2e_latency_ms": round(float(np.mean(end_to_end_latencies)), 2),
            "p95_e2e_latency_ms": round(float(np.percentile(end_to_end_latencies, 95)), 2),
            "details": details,
        }
        all_results.append(res)
        print(f"  -> Recall: {res['mean_recall']:.4f} | Hit Rate@5: {res['hit_rate']:.4f} ({res['hits']}/{n})")
        print(f"  -> Pure Retrieval Latency: avg={res['avg_retrieval_latency_ms']:.2f}ms, p95={res['p95_retrieval_latency_ms']:.2f}ms")
        print(f"  -> End-to-End Latency (incl. embedding): avg={res['avg_e2e_latency_ms']:.2f}ms, p95={res['p95_e2e_latency_ms']:.2f}ms")

    # Save to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, "pipeline_comparison_4_configs.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved updated metrics with latencies to: {out_file}")


if __name__ == "__main__":
    main()
