"""What did the answer cost, and did the model still answer at all?

    uv run python scripts/benchmark_answers.py --label answers-k5
    uv run python scripts/benchmark_answers.py --label answers-compress-d20 \
        --compress embedding --compress-candidates 20
    uv run python scripts/benchmark_answers.py --label ragas-d20-a0 --ragas

One row per run in `data/eval/answers.jsonl`. Three numbers per row: how many
characters of context the prompt carried, how many `prompt_tokens` the provider
actually billed, and how often the model refused a question the dataset says is
answerable.

Separate from `scripts/benchmark.py` for the reason `benchmark_conversations.py`
is separate: `run_benchmark`'s seam is `str -> list[ScoredChunk]` and produces
ranking metrics. This produces neither. Forcing it into `BenchmarkResult` would
mean inventing a `recall@5` for `summarise()` to render.

Refusal rate is a blunt binary signal and that is exactly its value at step 20:
it has no tuning surface, so it cannot be tuned into agreement. Step 21 adds
``--ragas``, which judges the answer itself — faithfulness, response relevancy
and context precision — on the same run and therefore the same row. The two
belonging to one row is the whole reason this lives here rather than in a fourth
benchmark script: step 20's finding is that they disagreed on ``q018``.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.evaluation.judge import (  # noqa: E402
    METRICS,
    JudgeSample,
    JudgeScores,
    judge,
    mean_scores,
)
from app.generation.answer import answer_question  # noqa: E402
from app.generation.compress import COMPRESSORS  # noqa: E402

HISTORY = Path("data/eval/answers.jsonl")
DEFAULT_DATASET = Path("data/eval/questions.jsonl")


def git_commit() -> str:
    """The commit a row was produced at. A number with no code beside it is a rumour."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def percentiles(values: list[float]) -> tuple[float, float]:
    """p50 and p95. One value is its own p50 and p95; none is zero."""
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    return statistics.median(ordered), ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--compress",
        choices=["", *sorted(COMPRESSORS)],
        default=None,
        help='sentence extractor; default: COMPRESS_METHOD, "" is off',
    )
    parser.add_argument("--compress-candidates", type=int, default=None)
    parser.add_argument("--compress-budget", type=int, default=None)
    parser.add_argument(
        "--unanswerable",
        action="store_true",
        help="run the out-of-corpus questions instead; a refusal there is correct",
    )
    parser.add_argument("--no-save", action="store_true", help="print only, append nothing")
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="judge each answer as well as counting refusals; costs judge calls",
    )
    parser.add_argument(
        "--ragas-metrics",
        default=",".join(sorted(METRICS)),
        help=f"comma-separated subset of {sorted(METRICS)}",
    )
    parser.add_argument("--judge-model", default=None, help="default: JUDGE_MODEL")
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=None,
        help="compressor budget exponent; default: COMPRESS_LENGTH_PENALTY",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    settings = get_settings()
    overrides = {
        key: value
        for key, value in (
            ("judge_model", args.judge_model),
            ("compress_length_penalty", args.length_penalty),
        )
        if value is not None
    }
    # model_copy rather than four more parameters threaded through
    # answer_question: both of these are already read from `settings` at the
    # bottom of the call stack, and the arm-per-run shape means one object per
    # run is the whole requirement.
    settings = settings.model_copy(update=overrides) if overrides else settings
    questions = [q for q in load_dataset(args.dataset) if not q.held_out]
    questions = [q for q in questions if (q.category == "unanswerable") == args.unanswerable]
    if not questions:
        raise SystemExit("no questions selected")

    rows: list[dict[str, Any]] = []
    samples: list[JudgeSample] = []
    for question in questions:
        started = time.perf_counter()
        try:
            answer = answer_question(
                question.question,
                top_k=args.top_k,
                compress=args.compress,
                compress_candidates=args.compress_candidates,
                compress_budget=args.compress_budget,
                settings=settings,
                include_contexts=args.ragas,
            )
        except Exception as error:  # noqa: BLE001 — one transient API error must not
            # cost a 38-question run; it is recorded as a row, never swallowed.
            rows.append({"question_id": question.question_id, "error": f"{error}"})
            continue
        rows.append(
            {
                "question_id": question.question_id,
                "category": question.category,
                "refused": answer.refusal is not None,
                "refusal": answer.refusal,
                # Per row, because step 22's Rule P compares citation warnings
                # between arms and no row before this one recorded them.
                "warnings": answer.warnings,
                "context_chars": answer.context_chars,
                # Relevancy penalises terse answers as incomplete: a correct
                # one-clause answer measures 0.278 where the same fact stated
                # fully measures 0.992. An arm that changes answer length
                # therefore moves relevancy for a reason that is not answer
                # quality, and reporting the length beside the score is the
                # only thing that lets a reader tell the two apart.
                "answer_chars": len(answer.answer),
                "prompt_tokens": answer.usage.get("prompt_tokens", 0),
                "retrieved": answer.retrieval.retrieved,
                "used": answer.retrieval.used,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
        if args.ragas:
            samples.append(
                JudgeSample(
                    question_id=question.question_id,
                    question=question.question,
                    answer=answer.answer,
                    contexts=answer.contexts,
                )
            )

    scored = [row for row in rows if "error" not in row]
    chars_p50, chars_p95 = percentiles([float(r["context_chars"]) for r in scored])
    answer_p50, answer_p95 = percentiles([float(r["answer_chars"]) for r in scored])
    tokens_p50, tokens_p95 = percentiles([float(r["prompt_tokens"]) for r in scored])
    latency_p50, latency_p95 = percentiles([float(r["latency_ms"]) for r in scored])
    refusals = sum(1 for row in scored if row["refused"])
    refusal_reasons = dict(Counter(str(r["refusal"]) for r in scored if r["refusal"]))
    warnings_total = sum(len(r["warnings"]) for r in scored)

    metric_names = [name.strip() for name in args.ragas_metrics.split(",") if name.strip()]
    judged = judge(samples, metrics=metric_names, settings=settings) if args.ragas else []
    by_id = {result.question_id: result for result in judged}
    for row in rows:
        judged_row = by_id.get(str(row.get("question_id")))
        if judged_row is None:
            continue
        row["ragas"] = judged_row.scores
        if judged_row.errors:
            row["ragas_errors"] = judged_row.errors

    # Grouped from `scored` rather than from `judged` so a category is named by
    # the dataset, not by which questions happened to survive the judge.
    per_category: dict[str, list[JudgeScores]] = {}
    for row in scored:
        judged_row = by_id.get(str(row["question_id"]))
        if judged_row is not None:
            per_category.setdefault(str(row["category"]), []).append(judged_row)

    ragas_overall = mean_scores(judged)
    ragas_by_category = {
        category: mean_scores(results) for category, results in sorted(per_category.items())
    }

    result = {
        "label": args.label,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "config": {
            "top_k": args.top_k or settings.top_k,
            "compress": args.compress if args.compress is not None else settings.compress_method,
            "compress_candidates": args.compress_candidates or settings.compress_candidates,
            "compress_budget": args.compress_budget or settings.compress_budget_chars,
            "collection": settings.qdrant_collection,
            "generation_model": settings.generation_model,
            "unanswerable": args.unanswerable,
            "judge_model": settings.judge_model,
            "length_penalty": settings.compress_length_penalty,
        },
        "questions": len(scored),
        "failures": len(rows) - len(scored),
        "refusals": refusals,
        "refusal_rate": refusals / len(scored) if scored else 0.0,
        "refusal_reasons": refusal_reasons,
        "warnings_total": warnings_total,
        "ragas": ragas_overall,
        "ragas_by_category": ragas_by_category,
        "ragas_failures": sum(len(r.errors) for r in judged),
        "context_chars_p50": chars_p50,
        "context_chars_p95": chars_p95,
        "answer_chars_p50": answer_p50,
        "answer_chars_p95": answer_p95,
        "prompt_tokens_p50": tokens_p50,
        "prompt_tokens_p95": tokens_p95,
        "latency_p50_ms": latency_p50,
        "latency_p95_ms": latency_p95,
        "per_question": rows,
    }

    print(f"\n## {args.label}  ({result['questions']} questions, {result['failures']} failed)\n")
    print(f"refusal rate     {result['refusal_rate']:.3f}  ({refusals} of {len(scored)})")
    print(f"refusal reasons  {refusal_reasons or '-'}")
    print(f"warnings         {warnings_total}")
    print(f"context chars    p50 {chars_p50:.0f}   p95 {chars_p95:.0f}")
    print(f"prompt tokens    p50 {tokens_p50:.0f}   p95 {tokens_p95:.0f}")
    print(f"latency ms       p50 {latency_p50:.0f}   p95 {latency_p95:.0f}")
    print(f"answer chars     p50 {answer_p50:.0f}   p95 {answer_p95:.0f}")

    for name, value in sorted(ragas_overall.items()):
        print(f"{name:<16} {value:.3f}")
    if ragas_by_category:
        print()
        for category, values in sorted(ragas_by_category.items()):
            rendered = "  ".join(f"{k} {v:.3f}" for k, v in sorted(values.items()))
            print(f"{category:<14} {rendered}")
    if result["ragas_failures"]:
        print(f"\n{result['ragas_failures']} judge failures — see ragas_errors in the row")

    if not args.no_save:
        with HISTORY.open("a", encoding="utf-8") as history:
            history.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"\nappended to {HISTORY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
