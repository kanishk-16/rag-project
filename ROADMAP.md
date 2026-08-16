# 🚀 Enterprise RAG Platform — Roadmap (Placement Edition)

**Target:** Resume-ready, fully working, deployed RAG system before companies start arriving in August.
**Timeframe:** ~6–8 weeks, ~3–4 focused hours/day (alongside DSA/aptitude prep).

> Golden rule for this cycle: **a small project that's finished, tested, and deployed beats an ambitious one that's half-built.** Every phase below is scoped to be cuttable without embarrassment.

---

## 1. Vision

Build a working Retrieval-Augmented Generation platform that ingests documents, retrieves relevant context, generates grounded answers via an LLM, supports multiple users, and can be deployed and demoed live — with enough evaluation rigor to back up every claim made about it in an interview.

Demonstrates: Backend Engineering · LLM Engineering · Retrieval Engineering · Production Readiness · AI Evaluation · System Design.

---

## 2. The Cut Line (read this before writing code)

| Tier | Phases | Rule |
|---|---|---|
| 🟢 Must-ship | 1, 2, 3, 4, 5, 13 | Not done by deadline = not a resume project. No exceptions. |
| 🟡 Differentiators | 6 (hybrid search + rerank), 10 (evaluation) | These are what make the project sound senior, not tutorial-level. Protect this time. |
| 🟠 One bonus, max | 8 (streaming) *or* 7 (memory) | Pick one if there's a buffer week. Don't attempt both. |
| 🔴 Cut for this cycle | 9 (multi-LLM beyond a thin factory), 11 (agentic), OCR, audio, web search fallback, admin dashboard | "I scoped these out to prioritize a polished, evaluated system" is a good interview answer. |

If you fall behind: drop bonus → drop frontend polish → drop Phase 6/10 scope (not the phases entirely, just the depth) — in that order. Never drop the eval numbers entirely; even 3 sentences of "before/after" beats nothing.

---

## 3. Tech Stack

**Backend:** FastAPI, Python, Pydantic, Uvicorn
**LLMs:** Pick ONE to start (OpenAI or Gemini or Anthropic) → add a thin `LLMFactory` interface in Week 5 to swap providers
**Embeddings:** Start with `all-MiniLM-L6-v2` (fast, local, fine for dev). For the Week 5 comparison, use **BGE-M3** instead of Instructor-XL — Instructor-XL has fallen out of favor; BGE-M3 is the current standard self-hosted multilingual model and a more current name to say out loud in an interview.
**Vector DB:** ChromaDB (Qdrant/Pinecone are "future work," mention but don't build)
**Database:** PostgreSQL
**Cache:** Redis (only if time allows — cite as future work otherwise)
**Storage:** Local filesystem (S3 = future work)
**Frontend:** React or Next.js — minimal pages only
**Deployment:** Docker + Docker Compose, deployed to a free-tier host (Render/Railway/Fly.io/a VM)

---

## 4. Folder Structure

```
backend/
  app/
    api/
    auth/
    services/
    retrieval/
    llm/
    embeddings/
    ingestion/
    evaluation/
    database/
    models/
    utils/
  tests/
  uploads/
  logs/
frontend/
docker/
eval/
  eval_set.json
  results/
```

---

## 5. Week-by-Week Plan

### Week 1 — Phase 1: MVP pipeline, deployed ugly
- [ ] Upload PDF → extract text → chunk → embed (MiniLM) → store (Chroma)
- [ ] Retrieve top-k chunks → generate answer (one LLM provider)
- [ ] `POST /upload`, `POST /query`, `GET /health`
- [ ] Docker container running by end of week — even if rough

### Week 2 — Phase 2 + Phase 3 + start the eval set
- [ ] Chunk overlap, better chunking strategy
- [ ] Metadata: doc name, page number, chunk number, upload time, user ID
- [ ] Multi-file support: PDF, TXT, DOCX (skip HTML/Markdown/CSV)
- [ ] Write 20–30 Q&A pairs from your own test docs → `eval/eval_set.json`
- [ ] Log baseline Recall@k informally — this is your "before" number

### Week 3 — Phase 4 + Phase 5
- [ ] Prompt template with grounded-refusal behavior ("I couldn't find the answer.")
- [ ] Auth: register, login, JWT, protected routes, multi-user
- [ ] Most student RAG projects skip real auth — this is a cheap, high-signal differentiator

### Week 4 — Phase 6 (highest-leverage week)
- [ ] Hybrid search: BM25 + semantic
- [ ] Reranking: top-20 → top-5 via cross-encoder
- [ ] Re-run the eval set. Get the number: *"Recall@5 went from X to Y."*

### Week 5 — Phase 10 (formalize eval) + thin Phase 9
- [ ] Compare 2 chunk sizes, and MiniLM vs BGE-M3
- [ ] Measure Precision@K, Recall@K, rough faithfulness, latency
- [ ] Build a minimal `LLMFactory` interface to swap providers
- [ ] Write up results in `eval/results/`

### Week 6 — Frontend + Deployment + Polish
- [ ] Minimal pages: login, upload, chat (functional > pretty)
- [ ] Docker Compose, environment variables
- [ ] README with architecture diagram
- [ ] Deploy to a live, clickable instance
- [ ] Record a 2–3 minute demo video

### Weeks 7–8 (buffer, if available)
- [ ] Pick **one**: streaming (`StreamingResponse`, high visual impact for low effort) OR conversation memory
- [ ] Final polish, README pass, mock-interview your own project out loud

---

## 6. Backend Concepts Demonstrated
FastAPI · REST APIs · Authentication · Dependency Injection · Background Tasks · Logging · Exception Handling · Docker · PostgreSQL · (Redis if time allows)

## 7. AI Concepts Demonstrated
RAG · Embeddings · Chunking · Vector Databases · Semantic Search · Prompt Engineering · Hybrid Search · Reranking · LLM Evaluation · Hallucination Reduction

---

## 8. Final Deliverables Checklist
- [ ] Clean GitHub repo with meaningful commit history
- [ ] README with architecture diagram
- [ ] API documentation (FastAPI auto-docs is fine)
- [ ] Dockerized application
- [ ] Live deployed demo
- [ ] Eval results written up with concrete before/after numbers
- [ ] Basic unit + integration tests
- [ ] Demo video (2–3 min)
- [ ] A 2-minute and a 20-minute version of "how I'd explain this in an interview"

---

## 9. Explicitly Out of Scope This Cycle
Agentic RAG (planner/reasoner/verifier loop) · OCR for scanned docs · audio/lecture transcription · web-search fallback · admin dashboard · Kubernetes · multi-region storage.

These are real, good ideas — just not for a 6–8 week placement timeline. List them in the README under "Future Work" — it shows scope awareness, which interviewers like.
