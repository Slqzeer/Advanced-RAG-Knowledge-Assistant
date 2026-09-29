"""Answer the same question twice for the price of once.

    cached_answer -> exact key -> nearest stored question -> answer_question -> store

A wrapper, not a pipeline stage: answer_question() never learns a cache exists,
and every benchmark calls answer_question() directly, so no measurement can be
served a stored answer. Step 22 learned why that matters from a gateway's cache.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from app.core.config import Settings

# Settings that cannot change an answer. A denylist, not an allowlist: a setting
# added later changes the key by default, so forgetting one costs a miss, never a
# stale answer.
_UNKEYED = frozenset(
    {"qdrant_url", "redis_url", "cache_ttl_s", "cache_semantic_threshold", "corpus_dir"}
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


def fingerprint(settings: Settings, kwargs: Mapping[str, Any]) -> str:
    """Everything that shapes an answer except the question: a paraphrase may only
    hit an answer produced under the same fingerprint.

    kwargs are keyed as passed, not resolved against settings, so ``top_k=None``
    and ``top_k=5`` are two keys for one answer. That costs a miss, never a stale
    hit, and resolving them here would duplicate answer_question()'s defaults.
    """
    return _digest({"settings": keyed_settings(settings), "kwargs": dict(kwargs)})


def cache_key(
    question: str,
    history: Sequence[Mapping[str, str]] | None,
    settings: Settings,
    kwargs: Mapping[str, Any],
) -> str:
    """The exact-layer key. Whitespace collapsed, case kept: ``Depends`` and
    ``depends`` are different things in these docs, and paraphrase is the
    semantic layer's job. History truncated exactly as contextualize() does."""
    turns = [dict(turn) for turn in list(history or [])[-settings.history_turns :]]
    return _digest(
        {
            "question": " ".join(question.split()),
            "history": turns,
            "fp": fingerprint(settings, kwargs),
        }
    )
