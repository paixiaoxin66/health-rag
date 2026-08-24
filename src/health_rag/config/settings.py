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
    embedding_device: str = "cpu"

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5

    # Vector store
    vector_store_type: str = "chroma"
    vector_store_path: str = "./data/vector_store"

    # LLM
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    deepseek_api_key: str = ""

    # Hugging Face
    hf_token: str = ""

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