"""Answer the same question twice for the price of once.

    cached_answer -> exact key -> nearest stored question -> answer_question -> store

A wrapper, not a pipeline stage: answer_question() never learns a cache exists,
and every benchmark calls answer_question() directly, so no measurement can be
served a stored answer. Step 22 learned why that matters from a gateway's cache.
"""

import hashlib
import json
import struct
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError
from redis import Redis

from app.core.config import Settings, get_settings
from app.generation.answer import answer_question
from app.models.answers import Answer
from app.observability.trace import QueryRecorder, record_answer, record_failure
from app.retrieval.search import default_embedder

PREFIX = "rag:cache:"
ANSWER_PREFIX = PREFIX + "answer:"
QVEC = PREFIX + "qvec"
CORPUS_REVISION = PREFIX + "corpus_revision"
# A dead Redis costs at most this per call, never a hang.
TIMEOUT_S = 0.2

Embedder = Callable[[str], list[float]]

# Settings that cannot change an answer. A denylist, not an allowlist: a setting
# added later changes the key by default, so forgetting one costs a miss, never a
# stale answer.
_UNKEYED = frozenset(
    {
        "qdrant_url",
        "redis_url",
        "cache_ttl_s",
        "cache_semantic_threshold",
        "trace_content",
        "generation_usd_per_mtok_input",
        "generation_usd_per_mtok_output",
        "corpus_dir",
    }
)


def keyed_settings(settings: Settings) -> dict[str, Any]:
    """Every setting that can shape an answer: secrets, local paths and the judge excluded."""
    return {
        name: value
        for name, value in settings.model_dump(mode="json").items()
        if name not in _UNKEYED
        and not name.endswith(("_api_key", "_path"))
        and not name.startswith("judge_")
    }


def _digest(payload: Mapping[str, Any]) -> str:
    # default=str: a filters mapping may hold tuples or enums, and a key that
    # cannot be built must not be the reason an answer fails.
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint(settings: Settings, kwargs: Mapping[str, Any], *, corpus_revision: int = 0) -> str:
    """Everything that shapes an answer except the question: a paraphrase may only
    hit an answer produced under the same fingerprint.

    kwargs are keyed as passed, not resolved against settings, so ``top_k=None``
    and ``top_k=5`` are two keys for one answer. That costs a miss, never a stale
    hit, and resolving them here would duplicate answer_question()'s defaults.
    """
    return _digest(
        {
            "settings": keyed_settings(settings),
            "kwargs": dict(kwargs),
            "corpus_revision": corpus_revision,
        }
    )


def cache_key(
    question: str,
    history: Sequence[Mapping[str, str]] | None,
    settings: Settings,
    kwargs: Mapping[str, Any],
    *,
    corpus_revision: int = 0,
) -> str:
    """The exact-layer key. Whitespace collapsed, case kept: ``Depends`` and
    ``depends`` are different things in these docs, and paraphrase is the
    semantic layer's job. History truncated exactly as contextualize() does."""
    turns = [dict(turn) for turn in list(history or [])[-settings.history_turns :]]
    return _digest(
        {
            "question": " ".join(question.split()),
            "history": turns,
            "fp": fingerprint(settings, kwargs, corpus_revision=corpus_revision),
        }
    )


def connect(settings: Settings) -> Redis:
    return Redis.from_url(
        settings.redis_url, socket_timeout=TIMEOUT_S, socket_connect_timeout=TIMEOUT_S
    )


def bump_corpus_revision(client: Redis | None = None, *, settings: Settings | None = None) -> int:
    """Atomically move future lookups to a fresh cache namespace."""
    settings = settings or get_settings()
    if client is None:
        if not settings.redis_url:
            return 0
        client = connect(settings)
    revision: Any = client.incr(CORPUS_REVISION)
    return int(revision)


def similarity_floor(threshold: float) -> float:
    """The VSIM score a cosine threshold maps to. VSIM scores (1 + cosine) / 2:
    1 identical, 0.5 orthogonal, 0 opposite. Pinned by a requires_redis test."""
    return (1 + threshold) / 2


def _command(client: Redis, *args: Any) -> Any:
    # The vector-set commands go through execute_command, which redis-py leaves
    # untyped; everything else it types as sync-or-async, and this client is sync.
    return client.execute_command(*args)  # type: ignore[no-untyped-call]


def _fp32(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _text(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _pairs(reply: Any) -> list[tuple[str, float]]:
    # redis-py 7.4 returns VSIM ... WITHSCORES as a flat [element, score, ...]
    # list, and an empty list for a missing key (probed against Redis 8.10).
    return [(_text(reply[i]), float(reply[i + 1])) for i in range(0, len(reply or []), 2)]


def _stored(client: Redis, key: str) -> Answer | None:
    raw: Any = client.get(ANSWER_PREFIX + key)
    if raw is None:
        return None
    try:
        return Answer.model_validate_json(raw)
    except ValidationError:
        # Written under an older Answer schema: a miss, and not one to repeat.
        client.delete(ANSWER_PREFIX + key)
        return None


def _nearest(client: Redis, vector: Sequence[float], fp: str, threshold: float) -> Answer | None:
    reply = _command(
        client,
        "VSIM",
        QVEC,
        "FP32",
        _fp32(vector),
        "WITHSCORES",
        "COUNT",
        1,
        "FILTER",
        f'.fp == "{fp}"',
    )
    pairs = _pairs(reply)
    if not pairs or pairs[0][1] < similarity_floor(threshold):
        return None
    element = pairs[0][0]
    hit = _stored(client, element)
    if hit is None:
        # The answer expired and its vector did not: vector-set elements have no
        # TTL of their own, so the lookup that finds one cleans it up.
        _command(client, "VREM", QVEC, element)
    return hit


def _warned(answer: Answer, message: str) -> Answer:
    return answer.model_copy(update={"warnings": [*answer.warnings, message]})


def cached_answer(
    question: str,
    *,
    history: Sequence[Mapping[str, str]] | None = None,
    settings: Settings | None = None,
    client: Redis | None = None,
    embedder: Embedder | None = None,
    answerer: Callable[..., Answer] = answer_question,
    trace: QueryRecorder | None = None,
    **kwargs: Any,
) -> Answer:
    """answer_question(), served from Redis when the same question, or a close
    enough paraphrase of it, was answered under the same settings.

    A hit returns one stored sample, frozen: at temperature 0 this generator
    still differs on identical input, and a cache is how that stops mattering.
    Refusals are never stored. Any Redis failure is a warning, never an error.
    """
    settings = settings or get_settings()
    owns_trace = trace is None and settings.trace_path is not None
    if owns_trace:
        trace = QueryRecorder(
            question,
            history or [],
            settings,
            mode=kwargs.get("mode"),
            transform=kwargs.get("transform"),
            rerank=kwargs.get("rerank"),
            compress=kwargs.get("compress"),
            top_k=kwargs.get("top_k"),
        )

    def finish(answer: Answer) -> Answer:
        return record_answer(trace, answer) if owns_trace and trace else answer

    def produce() -> Answer:
        try:
            trace_kwarg = {"trace": trace} if trace else {}
            return answerer(question, history=history, settings=settings, **kwargs, **trace_kwarg)
        except Exception as error:
            if owns_trace and trace:
                record_failure(trace, error)
            raise

    if client is None and not settings.redis_url:
        return finish(produce())

    started = time.perf_counter()
    client = client or connect(settings)

    def vector() -> list[float]:
        # Built only when needed: an exact hit, or a dead Redis, never opens the
        # sqlite embedding cache.
        return (embedder or default_embedder(settings))(question)

    # ponytail: follow-ups get exact lookups only. Assumption, not measurement: a
    # semantic hit on a follow-up needs the history resolved first (one LLM call,
    # ~1-2 s) before we know whether it hits, and we expect repeated follow-ups
    # to be rare. There is no traffic to measure that on yet; step 24's per-query
    # trace can count it.
    indexable = not history
    threshold = settings.cache_semantic_threshold

    try:
        revision_value: Any = client.get(CORPUS_REVISION)
        revision = int(revision_value or 0)
        key = cache_key(question, history, settings, kwargs, corpus_revision=revision)
        fp = fingerprint(settings, kwargs, corpus_revision=revision)
        hit, layer = _stored(client, key), "exact"
        if hit is None and indexable and threshold is not None:
            hit, layer = _nearest(client, vector(), fp, threshold), "semantic"
    except Exception as error:
        # Optional cache work includes embedding API and SQLite calls, not just Redis.
        answer = produce()
        return finish(_warned(answer, f"cache lookup failed, answered without it: {error}"))
    if hit is not None:
        elapsed = (time.perf_counter() - started) * 1000
        return finish(hit.model_copy(update={"cache": layer, "latency_ms": elapsed}))

    answer = produce()
    if answer.refusal is not None:
        return finish(answer)  # a refusal must not outlive the re-index that fixes it
    try:
        client.set(ANSWER_PREFIX + key, answer.model_dump_json(), ex=settings.cache_ttl_s)
        if indexable:
            # Indexed whether or not the semantic layer is on, so turning it on
            # later starts from a warm set rather than an empty one.
            _command(
                client,
                "VADD",
                QVEC,
                "FP32",
                _fp32(vector()),
                key,
                "NOQUANT",
                "SETATTR",
                json.dumps({"fp": fp}),
            )
    except Exception as error:
        return finish(_warned(answer, f"cache store failed: {error}"))
    return finish(answer)


def clear_cache(client: Redis | None = None, *, settings: Settings | None = None) -> int:
    """Delete every cached answer and the vector set; nothing outside the prefix.
    Returns how many keys were deleted. No Redis configured is zero, not an error."""
    settings = settings or get_settings()
    if client is None:
        if not settings.redis_url:
            return 0
        client = connect(settings)
    keys = [QVEC, *client.scan_iter(match=ANSWER_PREFIX + "*", count=500)]
    deleted: Any = client.delete(*keys)
    return int(deleted)
