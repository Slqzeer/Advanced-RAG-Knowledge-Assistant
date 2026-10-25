import math
from pathlib import Path
from typing import Any

from app.observability.trace import QueryTrace


def load_traces(path: Path) -> list[QueryTrace]:
    traces: list[QueryTrace] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            traces.append(QueryTrace.model_validate_json(line))
        except Exception as error:
            raise ValueError(f"invalid trace at line {line_number}: {error}") from error
    return traces


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(traces: list[QueryTrace]) -> dict[str, Any]:
    follow_ups = sum(trace.follow_up for trace in traces)
    repeats = sum(trace.follow_up_repeat for trace in traces)
    return {
        "total": len(traces),
        "generated": sum(trace.cache is None and trace.error_type is None for trace in traces),
        "refused": sum(trace.refusal is not None for trace in traces),
        "failed": sum(trace.error_type is not None for trace in traces),
        "exact_hits": sum(trace.cache == "exact" for trace in traces),
        "semantic_hits": sum(trace.cache == "semantic" for trace in traces),
        "follow_ups": follow_ups,
        "follow_up_repeats": repeats,
        "follow_up_repeat_rate": repeats / follow_ups if follow_ups else None,
        "latency_p50_ms": _percentile([trace.timings.total_ms for trace in traces], 0.50),
        "latency_p95_ms": _percentile([trace.timings.total_ms for trace in traces], 0.95),
        "prompt_tokens": sum(trace.usage.get("prompt_tokens", 0) for trace in traces),
        "completion_tokens": sum(trace.usage.get("completion_tokens", 0) for trace in traces),
        "known_cost_usd": sum(trace.cost_usd or 0 for trace in traces),
        "content_captured": sum(trace.response is not None for trace in traces),
    }


def read_metrics(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return summarize([])
    return summarize(load_traces(path))
