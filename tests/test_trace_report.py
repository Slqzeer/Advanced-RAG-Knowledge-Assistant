import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from trace_report import load_traces, summarize  # noqa: E402

from app.observability.trace import QueryTrace, TraceTimings  # noqa: E402


def trace(**updates) -> QueryTrace:
    base = QueryTrace(
        trace_id="0123456789abcdef0123456789abcdef",
        started_at=datetime(2026, 10, 23, tzinfo=UTC),
        question="q",
        model="m",
        retrieval_mode="dense",
        prompt_version="v2",
        top_k=5,
        timings=TraceTimings(total_ms=10),
    )
    return base.model_copy(update=updates)


def test_load_traces_round_trips_jsonl_and_accepts_an_empty_file(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(trace().model_dump_json() + "\n", encoding="utf-8")
    assert load_traces(path) == [trace()]

    path.write_text("", encoding="utf-8")
    assert load_traces(path) == []


def test_load_traces_names_the_bad_line(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(trace().model_dump_json() + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2"):
        load_traces(path)


def test_summary_counts_outcomes_follow_up_repeats_tokens_and_known_cost() -> None:
    traces = [
        trace(
            response="answer",
            usage={"prompt_tokens": 100, "completion_tokens": 20},
            cost_usd=0.01,
        ),
        trace(cache="exact", follow_up=True, follow_up_repeat=True),
        trace(refusal="no_context", follow_up=True, timings=TraceTimings(total_ms=20)),
        trace(error_type="RuntimeError", error_message="down"),
    ]

    summary = summarize(traces)

    assert summary["total"] == 4
    assert summary["generated"] == 2
    assert summary["refused"] == 1 and summary["failed"] == 1
    assert summary["exact_hits"] == 1 and summary["semantic_hits"] == 0
    assert summary["follow_ups"] == 2
    assert summary["follow_up_repeats"] == 1
    assert summary["follow_up_repeat_rate"] == 0.5
    assert summary["prompt_tokens"] == 100 and summary["completion_tokens"] == 20
    assert summary["known_cost_usd"] == 0.01
    assert summary["content_captured"] == 1


def test_no_follow_ups_has_no_invented_repeat_rate() -> None:
    assert summarize([trace()])["follow_up_repeat_rate"] is None
