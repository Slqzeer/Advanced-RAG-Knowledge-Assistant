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
    # Step 12 compared four: sentence won on Recall@5 (0.776 vs 0.713 recursive),
    # and 1000/200 was the peak of both the size and the overlap sweep.
    chunk_strategy: str = "sentence"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    # Step 14-16 compared three. The default changes only if the sweep meets the
    # rule in the step 14-16 design doc: dense | lexical | hybrid.
    retrieval_mode: str = "dense"
    # Per-branch depth before fusion. Fusing two top-10 lists cannot surface a
    # document neither branch ranked top-10, so this is the parameter that moves
    # recall; step 16 sweeps it.
    retrieval_candidates: int = 50
    # The constant from the original RRF paper. Larger flattens the rank
    # weighting, smaller sharpens it.
    rrf_k: int = 60
    corpus_dir: Path = Path("data/raw")


@lru_cache
def get_settings() -> Settings:
    return Settings()
