from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for the Deal Analyzer pipeline."""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:14b"
    embedding_model: str = "nomic-embed-text"
    num_ctx: int = 8192
    num_gpu: int = 35

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    collection_name: str = "deal_documents"

    # Data paths (DATA_DIR set via docker-compose; None = local fallback)
    data_dir: Optional[Path] = None

    # Chunking
    chunk_size: int = 600
    chunk_overlap: int = 100
    max_chunk_size: int = 1200
    semantic_threshold: int = 85

    # Retrieval
    retriever_k: int = 30
    fetch_k_multiplier: int = 10
    rerank_top_n: int = 6
    mmr_lambda: float = 0.7
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    min_chunk_chars: int = 150
    batch_size: int = 50
    max_retries: int = 3

    model_config = SettingsConfigDict(env_file=".env.local", extra="ignore")

    @property
    def data_root(self) -> Path:
        if self.data_dir:
            return self.data_dir
        return Path(__file__).resolve().parent.parent / "data"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def chroma_db_dir(self) -> Path:
        return self.data_root / "chroma_db"

    @property
    def hash_db_path(self) -> Path:
        return self.data_root / "ingestion_state.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
