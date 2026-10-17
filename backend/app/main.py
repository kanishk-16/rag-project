from fastapi import FastAPI
from app.db.session import engine, Base
from app.models.user import User  # Ensures user model registered with metadata
from app.api.routes import router as rag_router
from app.api.auth import router as auth_router

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="RAG Platform",
    description="Enterprise RAG Platform with Hybrid Search, Cross-Encoder Reranking, and Multi-Tenancy",
    version="0.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Authentication endpoints
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])

# Core RAG endpoints
app.include_router(rag_router, prefix="/api/v1", tags=["RAG"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)