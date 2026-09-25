from types import SimpleNamespace
from typing import Any

from app.generation.llm import GENERATION_TIMEOUT_S, complete


def test_complete_bounds_every_call_with_a_timeout() -> None:
    """A gateway that accepts the connection and never answers must fail the call,
    not hold a whole benchmark arm: step 22's inj-v2 waited three hours on one."""
    seen: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        seen.update(kwargs)
        message = SimpleNamespace(content="pong")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert complete("s", "u", model="m", client=client)[0] == "pong"
    assert seen["timeout"] == GENERATION_TIMEOUT_S


def test_complete_asks_the_gateway_for_a_fresh_generation() -> None:
    """OmniRoute replayed 80 of inj-v2-detect's 90 answers from its cache, so the
    arm compared inj-v2 with itself. Every call must reach the model."""
    seen: dict[str, Any] = {}

    def create(**kwargs: Any) -> Any:
        seen.update(kwargs)
        message = SimpleNamespace(content="pong")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    complete("s", "u", model="m", client=client)
    assert seen["extra_headers"] == {
        "X-OmniRoute-No-Cache": "true",
        "X-OmniRoute-No-Memory": "true",
    }


def test_complete_waits_out_a_gateway_cooldown_instead_of_losing_the_question(
    monkeypatch: Any,
) -> None:
    """OmniRoute cools a credential for up to ~160 s, past the SDK's backoff.
    Step 21's first GitHub arm lost 37 of 38 questions to it."""
    import httpx
    from openai import RateLimitError

    import app.generation.llm as llm

    monkeypatch.setattr(llm, "RATE_LIMIT_WAIT_S", 0.0)
    calls = 0

    def create(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            response = httpx.Response(429, request=httpx.Request("POST", "http://gw"))
            raise RateLimitError("cooling down", response=response, body=None)
        message = SimpleNamespace(content="pong")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert complete("s", "u", model="m", client=client)[0] == "pong"
    assert calls == 2
