from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    google_api_key: str

    # Embedding model (HuggingFace)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking — tune these in Phase 2/4 and measure the eval impact
    chunk_size: int = 500      # words per chunk
    chunk_overlap: int = 50    # words shared between adjacent chunks

    # Retrieval
    top_k: int = 5


    class Config:
        env_file = ".env"


# Single instance imported everywhere — don't instantiate Settings() again elsewhere
settings = Settings()