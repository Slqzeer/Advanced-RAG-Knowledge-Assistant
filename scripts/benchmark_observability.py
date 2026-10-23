"""Measure the local JSONL trace sink without model, Redis or Qdrant calls."""

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.observability.trace import (  # noqa: E402
    QueryTrace,
    TraceChunk,
    TraceTimings,
    write_trace,
)

HISTORY = Path("data/eval/observability.jsonl")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def representative_trace() -> QueryTrace:
    return QueryTrace(
        trace_id="0123456789abcdef0123456789abcdef",
        started_at=datetime.now(UTC),
        question="How does FastAPI dependency injection work?",
        resolved_query="How does FastAPI dependency injection work?",
        model="gpt-4o-mini",
        retrieval_mode="dense",
        compressor="embedding",
        prompt_version="v2",
        top_k=5,
        chunks=[
            TraceChunk(
                chunk_id=f"fastapi:doc-{rank}#0",
                document_id=f"fastapi:doc-{rank}",
                rank=rank,
                score=0.8 - rank / 100,
                used=rank <= 5,
                cited=rank <= 2,
            )
            for rank in range(1, 21)
        ],
        system_prompt="Answer only from context.",
        user_prompt="x" * 4_000,
        response="FastAPI resolves dependencies before the endpoint runs [1].",
        usage={"prompt_tokens": 1_000, "completion_tokens": 80, "total_tokens": 1_080},
        cost_usd=0.000198,
        timings=TraceTimings(
            contextualize_ms=0.01,
            retrieval_ms=35,
            context_ms=79,
            generation_ms=2_500,
            total_ms=2_614,
        ),
    )


def rule_o(*, valid_lines: int, requested: int, p95_ms: float) -> bool:
    return valid_lines == requested and p95_ms < 5


def benchmark(*, count: int, directory: Path) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be at least 1")
    directory.mkdir(parents=True, exist_ok=True)
    trace = representative_trace()
    with tempfile.TemporaryDirectory(prefix="trace-bench-", dir=directory) as temporary:
        path = Path(temporary) / "traces.jsonl"
        latencies = []
        for _ in range(count):
            started = time.perf_counter()
            write_trace(path, trace)
            latencies.append((time.perf_counter() - started) * 1000)
        lines = path.read_text(encoding="utf-8").splitlines()
        valid = sum(QueryTrace.model_validate_json(line) is not None for line in lines)
        size = path.stat().st_size
    p50, p95 = _percentile(latencies, 0.50), _percentile(latencies, 0.95)
    return {
        "writes": count,
        "valid_lines": valid,
        "p50_ms": p50,
        "p95_ms": p95,
        "bytes_per_trace": size / count,
        "total_bytes": size,
        "rule_o": rule_o(valid_lines=valid, requested=count, p95_ms=p95),
    }


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    result = benchmark(count=args.count, directory=Path("data/processed"))
    row = {
        "label": "observability-jsonl",
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        **result,
    }
    for name, value in result.items():
        print(f"{name:<18} {value}")
    if not args.no_save:
        with HISTORY.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"appended to {HISTORY}")
    return 0 if result["rule_o"] else 1


if __name__ == "__main__":
    sys.exit(main())
