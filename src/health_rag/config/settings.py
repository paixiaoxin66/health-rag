from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    # Application
    app_name: str = "health-rag"
    app_env: str = "development"
    debug: bool = True

    # Embedding
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_model_path: str = ""
    embedding_device: str = "cpu"
    embedding_local_only: bool = True

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_local_only: bool = True
    recall_k: int = 20

    # Vector store
    vector_store_type: str = "chroma"
    vector_store_path: str = "./data/vector_store"

    # LLM
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-v4-flash"
    deepseek_api_key: str = ""
    deepseek_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # Hugging Face
    hf_token: str = ""

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()