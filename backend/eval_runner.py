"""
Eval Runner — RAG Retrieval + Answer Quality Measurement
---------------------------------------------------------
Two modes:

  1. Retrieval-only (default, zero API calls, runs all 30 questions instantly)
     python eval_runner.py --mode retrieval

  2. Full eval with LLM judge (slow, uses Gemini API)
     python eval_runner.py --mode full

Why two modes?
  Gemini free tier = 20 requests/day. 30 questions x 2 LLM calls = 60 calls.
  Retrieval metrics (context recall, similarity scores) need ZERO API calls
  and are the most important metrics for measuring Phase 6 improvement anyway.
  LLM judge is supplementary — run it on a subset when quota allows.

Run from backend/ directory with venv active.
"""

import os
import sys
import json
import time
import argparse
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from app.embeddings.embedder import embed_query
from app.retrieval.vector_store import retrieve_top_k


# Metric: Context Recall
def compute_context_recall(expected_answer: str, retrieved_chunks: list) -> float:
    """
    Token overlap between expected answer and retrieved context.
    Score = matched_meaningful_tokens / total_meaningful_tokens_in_expected_answer
    Range: 0.0 (no overlap) to 1.0 (perfect recall)
    This is the most important metric for measuring Phase 6 improvement.
    """
    context = " ".join([c["text"] for c in retrieved_chunks]).lower()
    expected_tokens = set(expected_answer.lower().split())

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or",
        "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "it", "its", "this", "that", "be", "as", "from", "not",
        "than", "more", "also", "which", "when", "where", "who",
        "how", "what", "while", "then", "into", "about", "such", "each"
    }
    meaningful_tokens = expected_tokens - stop_words

    if not meaningful_tokens:
        return 0.0

    matched = sum(1 for token in meaningful_tokens if token in context)
    return round(matched / len(meaningful_tokens), 4)


# Metric: Hit Rate@K
def compute_hit_rate(expected_answer: str, retrieved_chunks: list, threshold: float = 0.3) -> bool:
    """
    Did at least one retrieved chunk contain enough of the expected answer?
    A hit = at least one chunk has context_recall >= threshold.
    """
    for chunk in retrieved_chunks:
        chunk_recall = compute_context_recall(expected_answer, [chunk])
        if chunk_recall >= threshold:
            return True
    return False


# LLM Judge with retry
def llm_judge_with_retry(question: str, expected: str, generated: str, model) -> dict:
    """LLM judge with automatic retry using delay from 429 error."""
    prompt = f"""You are evaluating a RAG system answer quality.

Question: {question}
Expected Answer: {expected}
Generated Answer: {generated}

Is the generated answer correct and consistent with the expected answer?
Respond with ONLY valid JSON, no other text:
{{"correct": true or false, "reason": "one sentence"}}"""

    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                match = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', error_str)
                wait = int(match.group(1)) + 5 if match else 70
                print(f"    [Rate limit] Waiting {wait}s before retry {attempt + 1}/3...")
                time.sleep(wait)
            else:
                return {"correct": False, "reason": f"Error: {error_str[:100]}"}

    return {"correct": False, "reason": "Failed after 3 retries — quota likely exhausted"}


def generate_answer_with_retry(question: str, chunks: list, model) -> str:
    """Generate answer with retry logic for rate limits."""
    context = "\n\n---\n\n".join([c["text"] for c in chunks])
    prompt = f"""Answer using ONLY the context below. Be concise. No markdown.

Context:
{context}

Question: {question}
Answer:"""

    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                match = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', error_str)
                wait = int(match.group(1)) + 5 if match else 70
                print(f"    [Rate limit] Waiting {wait}s before retry {attempt + 1}/3...")
                time.sleep(wait)
            else:
                return f"Error: {error_str[:100]}"

    return "Failed after 3 retries"


# Retrieval-only eval (no API calls)
def run_retrieval_eval(eval_path: str, output_path: str, top_k: int = 5):
    """
    Runs retrieval metrics only — zero API calls, all 30 questions in seconds.
    Metrics: Context Recall, Top-1 Similarity, Hit Rate@K
    These are the metrics that WILL change after Phase 6.
    """
    print(f"\n Loading eval set: {eval_path}")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f" Running retrieval eval on {len(eval_set)} questions (no API calls)...")
    print("-" * 60)

    results = []
    total_recall = 0.0
    total_top1 = 0.0
    total_hits = 0

    for i, item in enumerate(eval_set):
        q_id = item.get("id", f"q{i+1}")
        question = item["question"]
        expected = item["expected_answer"]

        query_emb = embed_query(question)
        chunks = retrieve_top_k(query_emb, top_k=top_k)

        if not chunks:
            print(f"  [{q_id}] No chunks retrieved")
            results.append({
                "id": q_id, "question": question,
                "context_recall": 0.0, "top1_score": 0.0,
                "hit": False, "retrieved_pages": [],
            })
            continue

        recall = compute_context_recall(expected, chunks)
        top1 = chunks[0]["score"]
        hit = compute_hit_rate(expected, chunks)
        pages = [c["metadata"].get("page_num") for c in chunks]

        total_recall += recall
        total_top1 += top1
        if hit:
            total_hits += 1

        results.append({
            "id": q_id, "question": question,
            "expected_answer": expected,
            "context_recall": recall, "top1_score": top1,
            "hit": hit, "retrieved_pages": pages,
        })

        hit_str = "HIT " if hit else "MISS"
        print(f"  [{q_id}] Recall: {recall:.2f} | Top-1: {top1:.4f} | {hit_str} | Pages: {pages}")

    n = len(eval_set)
    summary = {
        "run_date": datetime.utcnow().isoformat(),
        "mode": "retrieval-only",
        "total_questions": n,
        "top_k": top_k,
        "metrics": {
            "mean_context_recall": round(total_recall / n, 4),
            "mean_top1_similarity": round(total_top1 / n, 4),
            "hit_rate": round(total_hits / n, 4),
            "hits": total_hits,
            "total": n,
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVAL — BASELINE RESULTS")
    print("=" * 60)
    print(f"Total Questions      : {n}")
    print(f"Mean Context Recall  : {summary['metrics']['mean_context_recall']:.4f}")
    print(f"Mean Top-1 Similarity: {summary['metrics']['mean_top1_similarity']:.4f}")
    print(f"Hit Rate@{top_k}          : {summary['metrics']['hit_rate']:.4f}  ({total_hits}/{n} questions)")
    print(f"\nResults saved to: {output_path}")
    print("\nThese are your BASELINE numbers — save them.")
    print("Run again after Phase 6 (hybrid search + reranking) to measure improvement.")

    poor = sorted(results, key=lambda x: x["context_recall"])[:5]
    print(f"\nLowest recall questions (most room to improve):")
    for r in poor:
        print(f"  {r['id']}: recall={r['context_recall']:.2f} | {r['question'][:65]}...")


# Full eval with LLM judge
def run_full_eval(eval_path: str, output_path: str, top_k: int = 5, limit: int = None):
    """
    Full eval with LLM answer generation + LLM judge.
    WARNING: Uses Gemini API. Free tier = 20 requests/day.
    Each question uses 2 calls. Max 10 questions on free tier.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    if limit:
        eval_set = eval_set[:limit]

    needed = len(eval_set) * 2
    print(f"\n Questions: {len(eval_set)} | API calls needed: {needed} | Daily limit: 20")
    if needed > 18:
        print(f" WARNING: {needed} calls needed but only ~20 available per day.")
        print(f" Use --limit 9 to stay safely under quota.")
        ans = input("Continue anyway? (y/n): ")
        if ans.lower() != "y":
            return

    print(f"\n Running full eval (70s sleep between questions)...")
    print("-" * 60)

    results = []
    total_recall = 0.0
    total_top1 = 0.0
    total_correct = 0

    for i, item in enumerate(eval_set):
        q_id = item.get("id", f"q{i+1}")
        question = item["question"]
        expected = item["expected_answer"]

        print(f"\n[{i+1}/{len(eval_set)}] {q_id}: {question[:60]}...")

        query_emb = embed_query(question)
        chunks = retrieve_top_k(query_emb, top_k=top_k)

        if not chunks:
            results.append({"id": q_id, "question": question,
                            "context_recall": 0.0, "top1_score": 0.0,
                            "llm_correct": False, "error": "no_chunks"})
            continue

        recall = compute_context_recall(expected, chunks)
        top1 = chunks[0]["score"]
        total_recall += recall
        total_top1 += top1

        print(f"  Generating answer...")
        generated = generate_answer_with_retry(question, chunks, model)
        time.sleep(15)

        print(f"  Running LLM judge...")
        judgment = llm_judge_with_retry(question, expected, generated, model)

        if judgment["correct"]:
            total_correct += 1

        results.append({
            "id": q_id, "question": question,
            "expected_answer": expected,
            "generated_answer": generated,
            "context_recall": recall, "top1_score": top1,
            "llm_correct": judgment["correct"],
            "llm_reason": judgment["reason"],
            "retrieved_pages": [c["metadata"].get("page_num") for c in chunks],
        })

        status = "PASS" if judgment["correct"] else "FAIL"
        print(f"  Recall: {recall:.2f} | Top-1: {top1:.4f} | {status}")

        if i < len(eval_set) - 1:
            print(f"  Sleeping 70s...")
            time.sleep(70)

    n = len(eval_set)
    summary = {
        "run_date": datetime.utcnow().isoformat(),
        "mode": "full",
        "total_questions": n,
        "top_k": top_k,
        "metrics": {
            "mean_context_recall": round(total_recall / n, 4),
            "mean_top1_similarity": round(total_top1 / n, 4),
            "answer_correctness": round(total_correct / n, 4),
            "correct_count": total_correct,
            "total_count": n,
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("FULL EVAL RESULTS")
    print("=" * 60)
    print(f"Mean Context Recall  : {summary['metrics']['mean_context_recall']:.4f}")
    print(f"Mean Top-1 Similarity: {summary['metrics']['mean_top1_similarity']:.4f}")
    print(f"Answer Correctness   : {summary['metrics']['answer_correctness']:.4f} ({total_correct}/{n})")
    print(f"\nSaved to: {output_path}")


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Eval Runner")
    parser.add_argument("--mode", choices=["retrieval", "full"],
                        default="retrieval",
                        help="retrieval=no API calls (default) | full=LLM judge")
    parser.add_argument("--eval", default="../eval/eval_set.json")
    parser.add_argument("--output", default="../eval/results/baseline.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit questions in full mode (e.g. --limit 9)")
    args = parser.parse_args()

    if args.mode == "retrieval":
        run_retrieval_eval(args.eval, args.output, args.top_k)
    else:
        run_full_eval(args.eval, args.output, args.top_k, args.limit)