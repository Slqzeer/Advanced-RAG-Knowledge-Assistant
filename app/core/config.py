"""Every environment knob, read once.

The alternative is ``os.environ`` scattered across five modules, each with its
own default that drifts from the others. ``get_settings()`` is the whole
singleton story; it is cached so the ``.env`` file is parsed once per process.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Defaults are the local development setup; the environment overrides them."""

    # extra="ignore": .env carries keys for later phases (COHERE_API_KEY), and a
    # strict settings class would crash on them.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"
    # Not required: the test suite and every ingestion-free command must import
    # this module without a key present.
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_cache_path: Path = Path("data/processed/embeddings.sqlite")
    generation_model: str = "gpt-4o-mini"
    chunk_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    corpus_dir: Path = Path("data/raw")


@lru_cache
def get_settings() -> Settings:
    return Settings()
