import os
import uuid
import shutil
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import embed_texts, embed_query
from app.retrieval.vector_store import store_chunks, retrieve_top_k
from app.llm.client import generate_answer

router = APIRouter()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Request / Response models ─────────────────────────────────────────────────
# Pydantic models give you automatic validation and OpenAPI docs for free.

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceChunk(BaseModel):
    doc_id: str
    chunk_index: int
    page_num: int | None = None
    filename: str | None = None
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Quick liveness check — useful for Docker healthchecks and monitoring."""
    return {"status": "ok"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF → extract text → chunk → embed → store in VectorStore.

    Returns the doc_id (keep this — you'll use it in Phase 2 to filter
    queries by document) and the number of chunks created.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported for now. DOCX/TXT support coming soon."
        )

    # 1. Save to disk
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2. Extract pages with page numbers
    pages = load_pdf(file_path)
    if not pages:
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from PDF. It may be a scanned/image-only PDF."
        )

    # 3. Chunk → Embed → Store (the core RAG ingestion pipeline)
    chunks = chunk_text(pages, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_texts(chunk_texts)
    store_chunks(
        chunks=chunks,
        embeddings=embeddings,
        doc_id=doc_id,
        filename=file.filename,
        upload_time=datetime.now(timezone.utc),
    )

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_created": len(chunks),
        "message": "Document processed and indexed successfully.",
    }


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question → embed it → retrieve relevant chunks → generate answer.

    The `sources` field in the response tells the user which chunks
    the answer was based on. This is the start of your citation system.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Embed the question
    query_embedding = embed_query(request.question)

    # 2. Retrieve top-k relevant chunks
    relevant_chunks = retrieve_top_k(query_embedding, top_k=request.top_k)
    if not relevant_chunks:
        raise HTTPException(
            status_code=404,
            detail="No documents found in the index. Please upload a document first."
        )

    # 3. Generate grounded answer
    answer = generate_answer(request.question, relevant_chunks)

    # 4. Build citation list for the response
    sources = [
        SourceChunk(
            doc_id=chunk["metadata"]["doc_id"],
            chunk_index=chunk["metadata"]["chunk_index"],
            page_num=chunk["metadata"].get("page_num"),
            filename=chunk["metadata"].get("filename"),
            score=chunk["score"],
        )
        for chunk in relevant_chunks
    ]

    return QueryResponse(answer=answer, sources=sources)