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

    # extra="ignore": .env carries keys for phases that have not landed yet, and
    # a strict settings class would crash on them.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"
    # Not required: the test suite and every ingestion-free command must import
    # this module without a key present.
    openai_api_key: str | None = None
    # Step 17's hosted reranker. Not required: the test suite and every
    # FlashRank-only run must import this module without a Cohere key present.
    cohere_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_cache_path: Path = Path("data/processed/embeddings.sqlite")
    generation_model: str = "gpt-4o-mini"
    # Step 12 compared four: sentence won on Recall@5 (0.776 vs 0.713 recursive),
    # and 1000/200 was the peak of both the size and the overlap sweep.
    chunk_strategy: str = "sentence"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    # Step 16 measured all three: dense | lexical | hybrid. Dense stays the
    # default because the pre-registered rule was not met — the best fusion row
    # (hybrid-k60-d20) reached Recall@5 0.737 against 0.776, and conceptual and
    # multi_doc each regressed about 0.10. The two settings below still matter:
    # --mode hybrid is measured, shippable and 0.044 better at Recall@10.
    retrieval_mode: str = "dense"
    # Per-branch depth before fusion. Fusing two top-10 lists cannot surface a
    # document neither branch ranked top-10, so this is the parameter that moves
    # recall; step 16 sweeps it.
    retrieval_candidates: int = 50
    # The constant from the original RRF paper. Larger flattens the rank
    # weighting, smaller sharpens it.
    rrf_k: int = 60
    # ~34 MB. The ~4 MB nano default trades away exactly the ranking precision
    # step 17 is measuring, which would make the measurement meaningless.
    flashrank_model: str = "ms-marco-MiniLM-L-12-v2"
    # Empty means off: unchanged behaviour until step 17's measurement earns the
    # change. flashrank | cohere — compared at step 17, table in the README.
    rerank_model: str = ""
    # How deep the retrieved pool goes into the cross-encoder. This is the
    # parameter that bounds what reranking can do: a reranker reorders, it
    # cannot retrieve, so the pool's own recall is its ceiling.
    rerank_candidates: int = 30
    corpus_dir: Path = Path("data/raw")


@lru_cache
def get_settings() -> Settings:
    return Settings()
