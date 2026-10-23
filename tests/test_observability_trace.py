from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.observability.trace import QueryTrace, TraceTimings, generation_cost, write_trace


def minimal_trace(question: str = "How does Depends work?") -> QueryTrace:
    return QueryTrace(
        trace_id="0123456789abcdef0123456789abcdef",
        started_at=datetime(2026, 10, 23, tzinfo=UTC),
        question=question,
        model="test-model",
        retrieval_mode="dense",
        prompt_version="v2",
        top_k=5,
        timings=TraceTimings(total_ms=12.5),
    )


def test_a_minimal_trace_has_a_versioned_validated_identity() -> None:
    trace = minimal_trace()

    assert trace.schema_version == 1
    assert trace.follow_up is False
    assert trace.chunks == []
    assert trace.cost_usd is None


@pytest.mark.parametrize("trace_id", ["ABCDEF", "xyz", "0" * 31, "0" * 33])
def test_trace_ids_are_otel_compatible_lowercase_hex(trace_id: str) -> None:
    with pytest.raises(ValidationError, match="trace_id"):
        QueryTrace(
            trace_id=trace_id,
            started_at=datetime.now(UTC),
            question="q",
            model="m",
            retrieval_mode="dense",
            prompt_version="v2",
            top_k=5,
            timings=TraceTimings(total_ms=1),
        )


def test_observability_is_off_and_unpriced_by_default() -> None:
    settings = Settings(_env_file=None)
    assert settings.trace_path is None
    assert settings.trace_content is False
    assert settings.generation_usd_per_mtok_input is None
    assert settings.generation_usd_per_mtok_output is None


def test_the_sink_appends_utf8_records_without_overwriting(tmp_path) -> None:
    path = tmp_path / "nested" / "traces.jsonl"

    write_trace(path, minimal_trace("Comment fonctionne Depends ?"))
    write_trace(path, minimal_trace("Et les dépendances ?"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [QueryTrace.model_validate_json(line).question for line in lines] == [
        "Comment fonctionne Depends ?",
        "Et les dépendances ?",
    ]


def test_cost_requires_both_prices() -> None:
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
    assert generation_cost(usage, Settings(_env_file=None)) is None
    assert generation_cost(
        usage,
        Settings(
            _env_file=None,
            generation_usd_per_mtok_input=0.15,
            generation_usd_per_mtok_output=0.60,
        ),
    ) == pytest.approx(0.45)
