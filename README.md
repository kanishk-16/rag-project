# RAG Platform — Evidence-First Knowledge Assistant

A production-oriented Retrieval-Augmented Generation system built from scratch — no LangChain, no LlamaIndex. Every component of the pipeline is implemented directly and explainable.

**Ask questions about your documents. Get cited, grounded answers.**

> Built by [Kanishk Vikram Singh](https://github.com/kanishk-16)

---

## What it does

- Upload PDF documents via a REST API
- Automatically extracts text, chunks it, and generates embeddings
- Ask natural language questions and get answers grounded strictly in the document
- Every answer includes **page-level citations** — the exact page the answer came from
- Refuses to answer when the information isn't in the document

---

## Architecture

```
POST /upload
    │
    ▼
PyMuPDF          → Extract text page by page
    │
    ▼
Chunker          → Sliding window (500 words, 50 word overlap)
    │
    ▼
Sentence Transformers (all-MiniLM-L6-v2)  → 384-dim embeddings
    │
    ▼
Numpy Vector Store  → Cosine similarity search (persistent on disk)
    │
POST /query
    │
    ▼
embed_query()    → Same embedding model
    │
    ▼
retrieve_top_k() → Normalize → dot product → argsort → top-5 chunks
    │
    ▼
Gemini 2.5 Flash → Grounded prompt → answer + citations
```

### Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| PDF Loading | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | Numpy (cosine similarity) |
| LLM | Google Gemini 2.5 Flash |
| Config | Pydantic Settings + python-dotenv |
| Deployment | Docker (coming soon) |

---

## Project Structure

```
rag-platform/
├── backend/
│   ├── app/
│   │   ├── api/routes.py          # FastAPI endpoints
│   │   ├── core/config.py         # Pydantic settings
│   │   ├── ingestion/
│   │   │   ├── loader.py          # PDF text extraction
│   │   │   └── chunker.py         # Sliding window chunker
│   │   ├── embeddings/embedder.py # HuggingFace embeddings
│   │   ├── retrieval/vector_store.py  # Numpy cosine similarity
│   │   ├── llm/client.py          # Gemini API + grounded prompt
│   │   └── main.py                # App entrypoint
│   ├── tests/
│   │   └── test_chunker.py
│   └── requirements.txt
├── eval/
│   ├── eval_set.json              # 30 reviewed Q&A pairs
│   └── results/
│       └── baseline.json          # Retrieval benchmark results
├── docker/Dockerfile
├── .env.example
└── ROADMAP.md
```

---

## Evaluation

Built a 30-question evaluation set from a real document to measure retrieval quality objectively.

### Baseline Results (Phase 2 — semantic retrieval only)

| Metric              | Score  |
|---------------------|--------|
| Total Questions     | 30     |
| Mean Context Recall | 0.6750 |
| Mean Top-1 Similarity | 0.4839 |
| Hit Rate@5          | 0.9667 |

### What the metrics mean

- **Context Recall** — what fraction of the expected answer's key terms appear in the retrieved chunks
- **Top-1 Similarity** — cosine similarity score of the most relevant retrieved chunk
- **Answer Correctness** — LLM judge rates whether the generated answer matches the expected answer

### Planned improvement
Phase 6 adds hybrid search (BM25 + semantic) and cross-encoder reranking.
Target: Context Recall > 0.80, Answer Correctness > 65%.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | Liveness check |
| POST | `/api/v1/upload` | Upload and index a PDF |
| POST | `/api/v1/query` | Ask a question |

### Upload a document
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "doc_id": "86edab5d-...",
  "filename": "your_document.pdf",
  "chunks_created": 42,
  "message": "Document processed and indexed successfully."
}
```

### Query the document
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "top_k": 5}'
```

Response:
```json
{
  "answer": "The document covers...",
  "sources": [
    {
      "doc_id": "86edab5d-...",
      "chunk_index": 3,
      "page_num": 4,
      "filename": "your_document.pdf",
      "score": 0.6488
    }
  ]
}
```

Interactive API docs available at `http://localhost:8000/docs`

---

## Local Setup

### Prerequisites
- Python 3.10+
- A Google API key (get one at https://aistudio.google.com/app/apikeys)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/kanishk-16/rag-project.git
cd rag-project

# 2. Create virtual environment
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp ../.env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 5. Run the server
uvicorn app.main:app --reload

# 6. Open API docs
# Visit http://localhost:8000/docs
```

---

## Key Engineering Decisions

**No LangChain or LlamaIndex**
Every component is implemented directly — loader, chunker, embedder, vector store, LLM client. This means every line is explainable and every design decision was made deliberately.

**Word-level chunking with overlap**
Character-level splits cut words mid-token which confuses embedding models. 50-word overlap prevents context loss at chunk boundaries. Chunk size (500 words) was chosen to stay within LLM context limits while preserving paragraph-level coherence.

**Cosine similarity over euclidean distance**
Embedding models don't guarantee consistent vector magnitudes. Cosine similarity measures directional alignment regardless of magnitude — semantically similar texts score high even if their vectors differ in length.

**Numpy vector store over ChromaDB**
ChromaDB requires C++ compilation on Windows which creates environment friction. The numpy implementation is transparent (the cosine similarity math is readable in the source), educational, and easily swappable for a production vector database.

---

## Limitations and Future Work

**Known limitations:**
- Tables in PDFs lose their row/column structure after text extraction
- Images and scanned PDFs are not supported
- Vector store is in-memory with disk persistence — not suitable for large document collections
- No authentication — single-user only

**Planned improvements:**
- Hybrid search (BM25 + semantic) + cross-encoder reranking → Phase 6
- PostgreSQL for document and conversation persistence → Phase 7
- Authentication and multi-user support → Phase 8
- Docker Compose deployment → Phase 9
- DOCX and TXT support → Phase 10

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for the full 6-week development plan with phase-by-phase breakdown.

---

## Author

**Kanishk Vikram Singh**
[GitHub](https://github.com/kanishk-16)