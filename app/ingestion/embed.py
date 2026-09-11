"""Turn chunk texts into vectors, with a persistent cache in front of the provider.

The cache is the point of this module. Steps 10 onward re-run the pipeline
constantly while changing retrieval, never embeddings: without it every run pays
for and waits on the same vectors again. One sqlite file, stdlib, survives
restarts.

The model is part of the cache key. Switching models must miss rather than serve
vectors from a different embedding space into the same index.
"""

import hashlib
import os
import sqlite3
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_CACHE_PATH = Path(os.getenv("EMBEDDING_CACHE_PATH", "data/processed/embeddings.sqlite"))
BATCH_SIZE = 100
# The SDK already retries 429 and 5xx with exponential backoff; five attempts is
# enough that a rate limit halfway through a corpus run does not lose the
# uncached work. Whatever it does lose, the cache hands back on the next run.
MAX_RETRIES = 5
# sqlite allows 999 host variables per statement on builds older than 3.32, and a
# corpus run asks for thousands of keys at once.
_MAX_VARIABLES = 500


def cache_key(model: str, text: str) -> str:
    """The cache identity of one text under one model."""
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


def _pack(vector: Sequence[float]) -> bytes:
    """Little-endian float32. A 1536-float vector is 6 KB packed, ~30 KB as JSON."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class EmbeddingCache:
    """A sqlite-backed text-to-vector store, keyed by ``cache_key``."""

    def __init__(self, path: Path = DEFAULT_CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(key TEXT PRIMARY KEY, model TEXT NOT NULL, vector BLOB NOT NULL)"
        )

    def get(self, key: str) -> list[float] | None:
        return self.get_many([key]).get(key)

    def get_many(self, keys: Iterable[str]) -> dict[str, list[float]]:
        """The hits only — a missing key is simply absent from the result."""
        wanted = list(keys)
        found: dict[str, list[float]] = {}
        for start in range(0, len(wanted), _MAX_VARIABLES):
            batch = wanted[start : start + _MAX_VARIABLES]
            placeholders = ",".join("?" * len(batch))
            rows = self._db.execute(
                f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", batch
            )
            found.update({key: _unpack(blob) for key, blob in rows})
        return found

    def put_many(self, items: Iterable[tuple[str, str, Sequence[float]]]) -> None:
        """Store ``(key, model, vector)`` triples in one transaction."""
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings (key, model, vector) VALUES (?, ?, ?)",
                [(key, model, _pack(vector)) for key, model, vector in items],
            )


def build_client() -> OpenAI:
    """The provider client, built from ``OPENAI_API_KEY`` in the environment."""
    return OpenAI(max_retries=MAX_RETRIES)


def embed_texts(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_MODEL,
    cache: EmbeddingCache | None = None,
    client: Any | None = None,
    batch_size: int = BATCH_SIZE,
) -> list[list[float]]:
    """One vector per text, in input order, embedding only what the cache misses.

    The output is assembled by key lookup rather than by zipping the provider
    response onto the input list: with a partial cache hit the two have different
    lengths, and an off-by-one there attaches the wrong vector to the wrong chunk
    silently — nothing downstream notices except an unexplained drop in recall.
    """
    if not texts:
        return []

    keys = [cache_key(model, text) for text in texts]
    cache = cache or EmbeddingCache()
    vectors = cache.get_many(keys)

    # dict, not list: duplicate texts collapse to one entry, first-seen order kept.
    pending = {key: text for key, text in zip(keys, texts, strict=True) if key not in vectors}
    if pending:
        client = client or build_client()
        fresh: dict[str, list[float]] = {}
        items = list(pending.items())
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            response = client.embeddings.create(input=[t for _, t in batch], model=model)
            # Sorted by index because the response order is the provider's promise,
            # not ours to assume.
            data = sorted(response.data, key=lambda datum: datum.index)
            for (key, _), datum in zip(batch, data, strict=True):
                fresh[key] = list(datum.embedding)
        _check_dimensions({**vectors, **fresh})
        cache.put_many((key, model, vector) for key, vector in fresh.items())
        vectors.update(fresh)

    return [vectors[key] for key in keys]


def embed_query(text: str, **kwargs: Any) -> list[float]:
    """A single query vector.

    Its own entry point even though `text-embedding-3-*` needs no instruction
    prefix: a BGE-family model later would make the query prefix mandatory, and
    this is where it goes.
    """
    return embed_texts([text], **kwargs)[0]


def _check_dimensions(vectors: dict[str, list[float]]) -> None:
    """Every vector in a run must be the same length. A wrong-dimension vector
    reaching Qdrant is rejected there at best, and silently unsearchable at worst."""
    sizes = {len(vector) for vector in vectors.values()}
    if len(sizes) > 1:
        raise ValueError(f"inconsistent embedding dimension: {sorted(sizes)}")
