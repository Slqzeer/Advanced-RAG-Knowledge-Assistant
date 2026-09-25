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
