from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.generation.cache import cache_key, fingerprint

# The spec's denylist, spelled out rather than imported: a setting joining it is
# a decision a reviewer should see in this diff.
UNKEYED = {
    "qdrant_url",
    "redis_url",
    "cache_ttl_s",
    "cache_semantic_threshold",
    "corpus_dir",
    "embedding_cache_path",
    "openai_api_key",
    "cohere_api_key",
    "generation_api_key",
    "judge_model",
    "judge_embedding_model",
}


def _other(name: str, value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return value + 1
    if isinstance(value, Path):
        return value / "other"
    if value is None:
        return 0.5 if name == "cache_semantic_threshold" else "other"
    return f"{value}-other"


@pytest.mark.parametrize("name", sorted(Settings.model_fields))
def test_every_setting_outside_the_denylist_changes_the_key(name: str) -> None:
    base = Settings(_env_file=None)
    other = base.model_copy(update={name: _other(name, getattr(base, name))})
    changed = cache_key("q", None, base, {}) != cache_key("q", None, other, {})
    assert changed is (name not in UNKEYED)


def test_whitespace_is_collapsed_and_case_is_kept() -> None:
    settings = Settings(_env_file=None)
    assert cache_key("  How do I   use Depends?\n", None, settings, {}) == cache_key(
        "How do I use Depends?", None, settings, {}
    )
    assert cache_key("depends", None, settings, {}) != cache_key("Depends", None, settings, {})


def test_history_is_truncated_as_contextualize_truncates_it() -> None:
    settings = Settings(_env_file=None, history_turns=2)
    turns = [{"role": "user", "content": str(n)} for n in range(5)]
    assert cache_key("q", turns, settings, {}) == cache_key("q", turns[-2:], settings, {})
    assert cache_key("q", turns, settings, {}) != cache_key("q", turns[-1:], settings, {})
    assert cache_key("q", [], settings, {}) == cache_key("q", None, settings, {})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 7},
        {"mode": "hybrid"},
        {"compress": ""},
        {"filters": {"doc_type": "tutorial"}},
        {"include_contexts": True},
        {"strict": True},
    ],
)
def test_each_per_call_kwarg_changes_the_key_and_the_fingerprint(kwargs: dict[str, Any]) -> None:
    settings = Settings(_env_file=None)
    assert cache_key("q", None, settings, kwargs) != cache_key("q", None, settings, {})
    assert fingerprint(settings, kwargs) != fingerprint(settings, {})


def test_the_fingerprint_ignores_the_question() -> None:
    settings = Settings(_env_file=None)
    assert fingerprint(settings, {}) == fingerprint(settings, {})
    assert cache_key("a", None, settings, {}) != cache_key("b", None, settings, {})
