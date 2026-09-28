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


def _cooled(body: Any) -> Any:
    import httpx
    from openai import RateLimitError

    response = httpx.Response(429, request=httpx.Request("POST", "http://gw"))
    return RateLimitError("cooling down", response=response, body=body)


def test_a_cooldown_is_waited_for_as_long_as_the_gateway_says() -> None:
    """Retrying inside OmniRoute's cooldown extends it: step 21's grew to 1 046 s."""
    from app.generation.llm import COOLDOWN_MARGIN_S, cooldown_s

    assert cooldown_s(_cooled({"error": {"reset_seconds": 160}})) == 160 + COOLDOWN_MARGIN_S
    assert cooldown_s(_cooled({"reset_seconds": 58})) == 58 + COOLDOWN_MARGIN_S


def test_a_cooldown_without_a_reset_falls_back_and_one_past_the_cap_is_not_waited() -> None:
    from app.generation.llm import RATE_LIMIT_WAIT_S, cooldown_s

    assert cooldown_s(_cooled(None)) == RATE_LIMIT_WAIT_S
    # Antigravity's "reset after 133h" is an exhausted quota, not a cooldown.
    assert cooldown_s(_cooled({"error": {"reset_seconds": 133 * 3600}})) is None
