import os
import uuid
import shutil
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel

from app.core.config import settings
from app.ingestion.loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import embed_texts, embed_query
from app.retrieval.vector_store import store_chunks, retrieve_top_k
from app.llm.client import generate_answer
from app.api.auth import get_current_user
from app.models.user import User

router = APIRouter()

UPLOAD_DIR = settings.upload_dir
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    doc_id: Optional[str] = None


class SourceChunk(BaseModel):
    doc_id: str
    chunk_index: int
    page_num: Optional[int] = None
    filename: Optional[str] = None
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Quick liveness check — useful for Docker healthchecks and monitoring."""
    return {"status": "ok"}


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a PDF → extract text → chunk → embed → store in VectorStore.
    Tied to the authenticated user for multi-tenant data isolation.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. DOCX/TXT support coming in future releases."
        )

    # 1. File size validation
    file_peek = file.file.read(settings.max_upload_size_bytes + 1)
    if len(file_peek) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {settings.max_upload_size_bytes // (1024 * 1024)}MB."
        )
    file.file.seek(0)

    # 2. Save to disk
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 3. Extract pages with page numbers
    pages = load_pdf(file_path)
    if not pages:
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from PDF. It may be a scanned or image-only PDF."
        )

    # 4. Chunk → Embed → Store
    chunks = chunk_text(pages, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_texts(chunk_texts)
    store_chunks(
        chunks=chunks,
        embeddings=embeddings,
        doc_id=doc_id,
        filename=file.filename,
        upload_time=datetime.now(timezone.utc),
        user_id=current_user.id,
    )

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks_created": len(chunks),
        "user_id": current_user.id,
        "message": "Document processed and indexed successfully.",
    }


@router.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Ask a question → embed it → retrieve user's relevant chunks → generate grounded answer.
    Retrieval is strictly isolated to documents belonging to the authenticated user.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Embed question
    query_embedding = embed_query(request.question)

    # 2. Retrieve top-k relevant chunks (filtered by user_id and optional doc_id)
    relevant_chunks = retrieve_top_k(
        query_embedding,
        top_k=request.top_k,
        query_text=request.question,
        user_id=current_user.id,
        doc_id=request.doc_id,
    )
    if not relevant_chunks:
        raise HTTPException(
            status_code=404,
            detail="No matching documents found. Please upload a document first or verify your query."
        )

    # 3. Generate grounded answer
    answer = generate_answer(request.question, relevant_chunks)

    # 4. Build citations
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