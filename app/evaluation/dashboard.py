"""Build the compact, content-free data snapshot used by the benchmark dashboard."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

HISTORY_FILES = {
    "retrieval": "results.jsonl",
    "answers": "answers.jsonl",
    "cache": "cache.jsonl",
    "observability": "observability.jsonl",
}

FIELDS = {
    "retrieval": (
        "label",
        "timestamp",
        "git_commit",
        "config",
        "ks",
        "questions",
        "answerable",
        "unanswerable",
        "aggregate",
        "per_category",
        "latency_p50_ms",
        "latency_p95_ms",
    ),
    "answers": (
        "label",
        "timestamp",
        "git_commit",
        "config",
        "questions",
        "failures",
        "refusals",
        "refusal_rate",
        "context_chars_p50",
        "prompt_tokens_p50",
        "answer_chars_p50",
        "latency_p50_ms",
        "latency_p95_ms",
        "ragas",
        "ragas_by_category",
        "ragas_failures",
        "attack_cases",
        "attack_successes",
        "control_refusals",
    ),
    "cache": (
        "label",
        "timestamp",
        "git_commit",
        "config",
        "candidate_threshold",
        "false_hits",
        "must_miss",
        "paraphrase_hits",
        "rule_s",
        "stored",
        "hits",
        "misses",
        "cold_refusals",
        "hit_p50_ms",
        "hit_p95_ms",
        "miss_p50_ms",
        "miss_p95_ms",
        "prompt_tokens",
        "completion_tokens",
        "usd_at_list_price",
        "rule_e",
    ),
    "observability": (
        "label",
        "timestamp",
        "git_commit",
        "writes",
        "valid_lines",
        "p50_ms",
        "p95_ms",
        "bytes_per_trace",
        "total_bytes",
        "rule_o",
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load object rows and report malformed input at its exact source line."""
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"{path}: cannot read: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _latest_by_label(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    latest: dict[str, tuple[str, int, Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        label = row.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("benchmark row has no non-empty label")
        candidate = (str(row.get("timestamp", "")), index, row)
        if label not in latest or candidate[:2] >= latest[label][:2]:
            latest[label] = candidate
    return [entry[2] for entry in sorted(latest.values(), key=lambda entry: entry[:2])]


def _compact(kind: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in FIELDS[kind] if field in row and row[field] is not None}


def build_snapshot(source_dir: Path) -> dict[str, Any]:
    """Return latest labelled aggregates from every committed history."""
    histories = {
        kind: load_jsonl(source_dir / filename) for kind, filename in HISTORY_FILES.items()
    }
    compact = {
        kind: [_compact(kind, row) for row in _latest_by_label(rows)]
        for kind, rows in histories.items()
    }
    timestamps = [
        str(row["timestamp"]) for rows in compact.values() for row in rows if row.get("timestamp")
    ]
    return {
        "generated_at": max(timestamps, default=None),
        "counts": {kind: len(rows) for kind, rows in histories.items()},
        **compact,
    }


def write_data_script(snapshot: Mapping[str, Any], destination: Path) -> None:
    """Write plain JavaScript so the dashboard also works from a file URL."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    destination.write_text(f"window.BENCHMARK_DATA = {payload};\n", encoding="utf-8")
