import json
from pathlib import Path

import pytest

from app.evaluation.dashboard import build_snapshot, load_jsonl, write_data_script


def write_jsonl(path: Path, *rows: dict[str, object]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_jsonl_names_the_bad_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    path.write_text('{"label": "ok"}\nnot json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"runs\.jsonl:2: invalid JSON"):
        load_jsonl(path)


def test_snapshot_keeps_latest_label_and_drops_question_evidence(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "results.jsonl",
        {
            "label": "dense",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "git_commit": "old",
            "config": {"mode": "dense"},
            "aggregate": {"recall@5": 0.5},
            "per_category": {"code": {"recall@5": 0.4}},
            "per_question": [{"question": "private"}],
            "latency_p50_ms": 10,
            "latency_p95_ms": 20,
            "questions": 45,
            "failures": [],
        },
        {
            "label": "dense",
            "timestamp": "2026-01-02T00:00:00+00:00",
            "git_commit": "new",
            "config": {"mode": "dense"},
            "aggregate": {"recall@5": 0.75},
            "per_category": {"code": {"recall@5": 0.8}},
            "per_question": [{"question": "still private"}],
            "latency_p50_ms": 12,
            "latency_p95_ms": 22,
            "questions": 45,
            "failures": [],
        },
    )
    for name in ("answers.jsonl", "cache.jsonl", "observability.jsonl"):
        write_jsonl(tmp_path / name)

    snapshot = build_snapshot(tmp_path)

    assert snapshot["counts"] == {
        "retrieval": 2,
        "answers": 0,
        "cache": 0,
        "observability": 0,
    }
    assert len(snapshot["retrieval"]) == 1
    assert snapshot["retrieval"][0]["git_commit"] == "new"
    assert snapshot["retrieval"][0]["aggregate"]["recall@5"] == 0.75
    assert "per_question" not in json.dumps(snapshot)
    assert "private" not in json.dumps(snapshot)


def test_snapshot_shapes_each_history_without_large_evidence(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "results.jsonl")
    write_jsonl(
        tmp_path / "answers.jsonl",
        {
            "label": "answer",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "config": {"compress": "embedding"},
            "questions": 38,
            "failures": 0,
            "refusals": 8,
            "refusal_rate": 0.21,
            "latency_p50_ms": 1000,
            "latency_p95_ms": 2000,
            "prompt_tokens_p50": 1200,
            "ragas": {"faithfulness": 0.86, "relevancy": 0.70},
            "ragas_by_category": {"code": {"relevancy": 0.72}},
            "per_question": [{"answer": "private"}],
        },
    )
    write_jsonl(
        tmp_path / "cache.jsonl",
        {
            "label": "cache",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "rule_e": True,
            "hits": 30,
            "per_probe": [{"query": "private"}],
        },
    )
    write_jsonl(
        tmp_path / "observability.jsonl",
        {
            "label": "trace",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "rule_o": True,
            "writes": 1000,
            "valid_lines": 1000,
            "p95_ms": 0.84,
        },
    )

    snapshot = build_snapshot(tmp_path)

    assert snapshot["answers"][0]["ragas"]["relevancy"] == 0.70
    assert snapshot["cache"][0] == {
        "label": "cache",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "rule_e": True,
        "hits": 30,
    }
    assert snapshot["observability"][0]["p95_ms"] == 0.84
    assert "per_probe" not in json.dumps(snapshot)


def test_write_data_script_is_deterministic_and_browser_ready(tmp_path: Path) -> None:
    snapshot = {"generated_at": "2026-01-01T00:00:00+00:00", "retrieval": []}
    destination = tmp_path / "nested" / "data.js"

    write_data_script(snapshot, destination)
    first = destination.read_text(encoding="utf-8")
    write_data_script(snapshot, destination)

    assert destination.read_text(encoding="utf-8") == first
    assert first.startswith("window.BENCHMARK_DATA = ")
    assert first.endswith(";\n")
