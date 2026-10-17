from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    google_api_key: str

    # Embedding model (HuggingFace)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Cross-encoder reranker
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking
    chunk_size: int = 500      # words per chunk
    chunk_overlap: int = 50    # words shared between adjacent chunks

    # Retrieval & Hybrid Search
    top_k: int = 5
    hybrid_semantic_weight: float = 0.7
    hybrid_bm25_weight: float = 0.3
    cross_encoder_candidates: int = 20

    # Paths & Storage
    store_dir: str = "./vector_store"
    upload_dir: str = "./uploads"
    max_upload_size_bytes: int = 25 * 1024 * 1024  # 25 MB

    # Database
    database_url: str = "sqlite:///./knowledge_vault.db"

    # Authentication & Security
    jwt_secret_key: str = "supersecret-jwt-key-for-rag-platform-production-32chars"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60 * 24  # 24 hours



# Single instance imported everywhere — don't instantiate Settings() again elsewhere
settings = Settings()