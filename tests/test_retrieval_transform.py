"""Query transforms with a fake model: no network, no key, no spend."""

from typing import Any

import pytest

from app.core.config import Settings
from app.retrieval.transform import (
    TRANSFORMS,
    contextualize,
    expand,
    parse_queries,
)

SETTINGS = Settings(_env_file=None, generation_model="test-model", multi_query_n=3, history_turns=4)


class FakeLLM:
    """Answers with a canned completion and records how it was called."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def __call__(self, system: str, user: str, **kwargs: Any) -> tuple[str, dict[str, int]]:
        self.calls.append({"system": system, "user": user, **kwargs})
        return self.text, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


# --- the parser ------------------------------------------------------------


def test_plain_lines_become_queries() -> None:
    assert parse_queries("first query\nsecond query", limit=5) == ["first query", "second query"]


def test_numbered_and_bulleted_ordinals_are_stripped() -> None:
    text = "1. how to configure cors\n2) enabling cors middleware\n- cors origins setting\n* cors"
    assert parse_queries(text, limit=5) == [
        "how to configure cors",
        "enabling cors middleware",
        "cors origins setting",
        "cors",
    ]


def test_surrounding_quotes_are_stripped() -> None:
    assert parse_queries('"how do dependencies work"', limit=1) == ["how do dependencies work"]


def test_blank_lines_are_dropped() -> None:
    assert parse_queries("\n\nreal query\n   \n", limit=5) == ["real query"]


def test_a_preamble_line_ending_in_a_colon_is_dropped() -> None:
    """The single most common way a chatty model poisons the query list."""
    assert parse_queries("Here are three queries:\nfirst\nsecond", limit=5) == ["first", "second"]


def test_the_list_is_capped_at_the_limit() -> None:
    assert parse_queries("a\nb\nc\nd", limit=2) == ["a", "b"]


def test_output_with_nothing_usable_parses_to_an_empty_list() -> None:
    assert parse_queries("   \n\n  ", limit=3) == []


# --- expand() --------------------------------------------------------------


def test_rewrite_returns_exactly_one_query() -> None:
    llm = FakeLLM("how to configure memory limits for docker containers")
    assert expand("et pour docker ?", transform="rewrite", settings=SETTINGS, llm=llm) == [
        "how to configure memory limits for docker containers"
    ]


def test_multi_always_keeps_the_original_first() -> None:
    """N phrasings that all drift is how expansion loses ground the raw query held."""
    llm = FakeLLM("fastapi security authentication\nfastapi oauth2 authentication")
    queries = expand("How does auth work?", transform="multi", n=3, settings=SETTINGS, llm=llm)
    assert queries[0] == "How does auth work?"
    assert queries == [
        "How does auth work?",
        "fastapi security authentication",
        "fastapi oauth2 authentication",
    ]


def test_multi_asks_for_one_fewer_than_n_because_the_original_counts() -> None:
    llm = FakeLLM("a\nb\nc\nd\ne")
    assert len(expand("q", transform="multi", n=3, settings=SETTINGS, llm=llm)) == 3


def test_multi_at_n_one_is_todays_behaviour_and_makes_no_call() -> None:
    llm = FakeLLM("unused")
    assert expand("q", transform="multi", n=1, settings=SETTINGS, llm=llm) == ["q"]
    assert llm.calls == []


def test_multi_drops_a_paraphrase_identical_to_the_original() -> None:
    llm = FakeLLM("q\nsomething else")
    queries = expand("q", transform="multi", n=3, settings=SETTINGS, llm=llm)
    assert queries == ["q", "something else"]


def test_multi_deduplicates_repeated_paraphrases() -> None:
    llm = FakeLLM("same\nsame\nother")
    assert expand("q", transform="multi", n=5, settings=SETTINGS, llm=llm) == ["q", "same", "other"]


def test_garbage_output_falls_back_to_the_original_query() -> None:
    """An API hiccup mid-benchmark must degrade to today's behaviour, never zero a row."""
    assert expand("q", transform="rewrite", settings=SETTINGS, llm=FakeLLM("")) == ["q"]
    assert expand("q", transform="multi", n=3, settings=SETTINGS, llm=FakeLLM("  \n ")) == ["q"]


def test_an_unknown_transform_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="hyde"):
        expand("q", transform="hyde", settings=SETTINGS, llm=FakeLLM("x"))


def test_an_empty_query_is_rejected_before_any_call() -> None:
    llm = FakeLLM("x")
    with pytest.raises(ValueError, match="empty"):
        expand("   ", transform="rewrite", settings=SETTINGS, llm=llm)
    assert llm.calls == []


def test_n_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        expand("q", transform="multi", n=0, settings=SETTINGS, llm=FakeLLM("x"))


def test_n_defaults_to_the_setting() -> None:
    llm = FakeLLM("a\nb\nc\nd")
    settings = Settings(_env_file=None, generation_model="test-model", multi_query_n=2)
    assert expand("q", transform="multi", settings=settings, llm=llm) == ["q", "a"]


def test_the_registry_holds_exactly_the_two_measured_transforms() -> None:
    assert sorted(TRANSFORMS) == ["multi", "rewrite"]


# --- contextualize() -------------------------------------------------------


def test_empty_history_returns_the_question_and_makes_no_call() -> None:
    """Paying 500-1500 ms to rewrite a first-turn question into itself is the
    failure this parameter would otherwise ship on every single-turn request."""
    llm = FakeLLM("unused")
    assert contextualize("How does auth work?", [], settings=SETTINGS, llm=llm) == (
        "How does auth work?"
    )
    assert llm.calls == []


def test_a_follow_up_is_resolved_against_the_history() -> None:
    llm = FakeLLM("How are memory limits configured for Docker containers?")
    history = [
        {"role": "user", "content": "Comment limiter la memoire d'un container Kubernetes ?"},
        {"role": "assistant", "content": "Via resources.limits.memory [1]."},
    ]
    assert contextualize("et pour docker ?", history, settings=SETTINGS, llm=llm) == (
        "How are memory limits configured for Docker containers?"
    )


def test_only_the_last_history_turns_are_sent() -> None:
    llm = FakeLLM("resolved")
    history = [{"role": "user", "content": f"turn {i}"} for i in range(10)]
    contextualize("and that?", history, settings=SETTINGS, llm=llm)
    sent = llm.calls[0]["user"]
    assert "turn 9" in sent
    assert "turn 5" not in sent


def test_an_unusable_rewrite_falls_back_to_the_raw_question() -> None:
    llm = FakeLLM("")
    history = [{"role": "user", "content": "How does auth work?"}]
    assert contextualize("and oauth?", history, settings=SETTINGS, llm=llm) == "and oauth?"


def test_contextualize_rejects_an_empty_question() -> None:
    with pytest.raises(ValueError, match="empty"):
        contextualize("  ", [{"role": "user", "content": "x"}], settings=SETTINGS, llm=FakeLLM("y"))
