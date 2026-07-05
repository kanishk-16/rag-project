from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="RAG Platform",
    description="Enterprise RAG Platform — Phase 1 MVP",
    version="0.1.0",
    docs_url="/docs",      # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",    # ReDoc UI at http://localhost:8000/redoc
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)