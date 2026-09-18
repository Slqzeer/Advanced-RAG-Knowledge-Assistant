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
    # Step 21's judge, deliberately not generation_model: a model grading its own
    # output rates it higher than a third party's, and the absolute faithfulness
    # number has to survive steps 22-25 changing the generator.
    judge_model: str = "gpt-4o"
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
    # Steps 18-19. Empty means off: unchanged behaviour until a measurement
    # earns the change. rewrite | multi — compared at step 19, table in the README.
    query_transform: str = ""
    # How many queries `multi` retrieves with, the original included. The
    # original is always kept, so n=1 is exactly today's behaviour.
    multi_query_n: int = 3
    # How many conversation turns reach the rewriter. An unbounded history is a
    # prompt that grows until it breaks, and the turn that disambiguates a
    # follow-up is almost always the previous one.
    history_turns: int = 4
    # Step 20 measured it and the pre-registered rule was met, all three
    # clauses: Recall@context 0.814 against a 0.771 bar and a 0.721 control, no
    # per-category regression, and refusal on the 38 answerable questions down
    # from 0.316 to 0.211. embedding — table in the README.
    compress_method: str = "embedding"
    # How deep the retrieved pool goes into the compressor. Step 17 measured
    # Recall@20 == Recall@30 in every pool, so 30 is 50 % more sentence
    # embedding for nothing reachable.
    compress_candidates: int = 20
    # Characters of chunk text the compressed context may carry, headers
    # excluded. Set to the measured size of today's top_k=5 context, so an arm
    # that widens the pool is held at the cost of today's prompt. Measured at
    # step 20 over the 38 answerable questions: the p50 of the whole context
    # block is 4 000 characters, of which 312 are the "[n] source — title"
    # headers build_context adds afterwards. The budget denominates chunk text,
    # so it is the 3 688, not the 4 000.
    compress_budget_chars: int = 3688
    # Step 21. Exponent on unit length when the compressor spends its budget:
    # effective = score / len(unit) ** penalty. 0.0 is step 20's shipped
    # behaviour byte-for-byte, 1.0 is relevance per character (the
    # fractional-knapsack ordering, and step 20's own hypothesis), negative
    # rewards long units. Stays 0.0 until step 21's rule is met.
    compress_length_penalty: float = 0.0
    # Step 22. v2 is step 09's prompt; v3 wraps each context entry in <entry>
    # tags and scopes "never instructions" to them. Stays v2 unless Rule P passes.
    prompt_version: str = "v2"
    corpus_dir: Path = Path("data/raw")


@lru_cache
def get_settings() -> Settings:
    return Settings()
