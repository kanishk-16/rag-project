# RAG Platform — Project Roadmap

> **Context:** Placement season has started. Priority is depth over breadth.
> A finished, deployed, measured system beats an ambitious half-built one every time.

---

## Current Status

**Live Demo:** https://rag-project-production-413e.up.railway.app/docs
**GitHub:** https://github.com/kanishk-16/rag-project

### Baseline Eval Numbers (semantic retrieval only)
| Metric | Score |
|---|---|
| Mean Context Recall | 0.6750 |
| Mean Top-1 Similarity | 0.4839 |
| Hit Rate@5 | 0.9667 (29/30 questions) |

Target after Phase 6: Context Recall > 0.80, Hit Rate@5 > 0.98

---

## Completed

### ✅ Phase 1 — Core RAG Pipeline
- PDF upload → PyMuPDF text extraction → sliding window chunker (500 words, 50 overlap)
- sentence-transformers `all-MiniLM-L6-v2` → 384-dim embeddings
- Numpy cosine similarity vector store (persistent on disk)
- `POST /upload`, `POST /query`, `GET /health`

### ✅ Phase 2 — Metadata + Citations
- Page number, filename, upload time stored on every chunk
- Page-level citations returned in every query response
- User can trace any answer back to its exact source page

### ✅ Phase 4 — Grounded Prompt
- Prompt template that restricts LLM to provided context only
- Explicit refusal behavior: "I couldn't find the answer in the provided documents"
- No markdown formatting in responses

### ✅ Phase 6 — Hybrid Search + Reranking
- BM25 keyword search (`rank-bm25`) + Dense cosine similarity (`all-MiniLM-L6-v2`)
- Min-max score normalization + weighted candidate pooling (top 20)
- Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) to top 5
- Benchmark results:
  - **Context Recall:** `0.6750` → **`0.7259`** (+7.5% relative gain)
  - **Hit Rate@5:** `96.67%` (29/30) → **`100.00%`** (30/30)
- Interview takeaway: *"Hybrid search + Cross-Encoder reranking rescued rare acronyms and technical names, achieving 100% Hit Rate@5 across our 30-question evaluation suite."*

### ✅ Phase 5 — Authentication & Multi-Tenancy
- SQLite + SQLAlchemy user persistence with bcrypt password hashing
- JWT-based authentication (`POST /api/v1/auth/register`, `POST /api/v1/auth/login`, `GET /api/v1/auth/me`)
- Protected `/api/v1/upload` and `/api/v1/query` routes via `Depends(get_current_user)`
- Strict multi-tenant isolation: document chunks indexed by `user_id`, vector store retrieval strictly scoped to the authenticated user's documents

### ✅ Phase 10 — Evaluation Suite
- 30-question ground-truth eval set from a real document (reviewed manually)
- `eval_runner.py` with two modes:
  - `--mode retrieval` — zero API calls, runs all 30 questions instantly
  - `--mode full` — LLM judge with retry logic for rate limits
- Baseline results captured in `eval/results/baseline.json`
- 4-configuration benchmark in `benchmark_4_configs.py` and `eval/results/pipeline_comparison_4_configs.json`

### ✅ Deployment
- `docker/Dockerfile` + `docker-compose.yml` with volume mounts
- Live on Railway — public HTTPS URL
- `.env.example` for clean local setup

---

## In Progress

### 🔄 Phase 8 — Streaming Responses (Next Step)
- FastAPI `StreamingResponse` for token-by-token output via Gemini stream API
- High visual impact for live technical demos
- Reduces time-to-first-token (TTFT) perception


---

## Future Work (explicitly cut this cycle)

| Feature | Reason cut |
|---|---|
| Phase 3: TXT + DOCX support | Low priority vs eval improvement |
| Phase 7: Conversation memory | Not enough time |
| Phase 9: LLMFactory (multi-provider) | Nice to have, not essential |
| Phase 11: Agentic RAG | Needs solid base RAG first |
| OCR for scanned PDFs | Significant effort, low interview ROI |
| PostgreSQL + async job queue | Right architecture, wrong timing |
| Admin dashboard | Cut entirely |
| Kubernetes | No real deployment problem that requires it |

"I scoped these out to prioritize a polished, evaluated, deployed system."
That is a good interview answer — it shows engineering judgment.

---

## What This Project Demonstrates

**Backend Engineering**
FastAPI · REST API design · Pydantic validation · dependency injection · Docker · Railway deployment

**AI/LLM Engineering**
RAG pipeline · embeddings · chunking strategies · vector similarity search · prompt engineering · grounded generation · hallucination reduction

**Retrieval Engineering**
Semantic search · cosine similarity · BM25 keyword search (Phase 6) · cross-encoder reranking (Phase 6)

**Evaluation**
Reproducible eval harness · Context Recall · Hit Rate@K · Mean Top-1 Similarity · before/after measurement

**System Design**
Modular architecture · separation of concerns · swappable vector store · persistent disk storage · environment-based configuration

---

## The 2-Minute Interview Explanation

"I built a RAG system from scratch — no LangChain, no LlamaIndex. The pipeline is:
PDF upload → PyMuPDF extraction → sliding window chunking with overlap → sentence-transformer
embeddings → cosine similarity retrieval → grounded Gemini answer with page-level citations.

I built a 30-question eval set and measured retrieval quality before and after changes.
Baseline: Context Recall 0.675, Hit Rate@5 96.7%.

The main weakness is Top-1 Similarity at 0.48 — the right chunk is found but not always
ranked first. I am adding hybrid search (BM25 + semantic) and cross-encoder reranking to fix this.

It is deployed live on Railway with Docker."

---

## The 10-Minute Interview Demonstration

1. Open live URL → show Swagger UI
2. Upload a PDF → show chunks_created and doc_id in response
3. Ask a specific question → show answer + page citations
4. Ask an unanswerable question → show refusal behavior
5. Open eval/eval_set.json → explain how the eval set was built
6. Run eval_runner.py --mode retrieval → show live metrics
7. Explain one engineering decision: why numpy over ChromaDB
8. Explain one limitation: tables lose structure after get_text()
9. Explain what Phase 6 will improve and why

---

## Git Commit Convention

feat: add hybrid BM25 + semantic search
fix: handle empty PDF pages in loader
eval: run baseline after hybrid search
docs: update eval results in README
test: add integration test for upload endpoint