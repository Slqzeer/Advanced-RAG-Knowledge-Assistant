"""Can a stored answer serve a paraphrase, and what does a hit save?

    uv run python scripts/benchmark_cache.py --sweep
    uv run python scripts/benchmark_cache.py --arm cold
    uv run python scripts/benchmark_cache.py --arm exact
    uv run python scripts/benchmark_cache.py --arm semantic

One row per run in `data/eval/cache.jsonl`. Not `results.jsonl`: those rows are
ranking metrics and `summarise()` renders them; these are hits and latencies.

`--sweep` makes no generation call. Every cosine is between two question
embeddings the sqlite cache already holds or computes for about $0.0001.
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_answers import git_commit, percentiles  # noqa: E402 - sibling script

from app.core.config import get_settings  # noqa: E402
from app.evaluation.dataset import CachePair, EvalQuestion, load_dataset  # noqa: E402
from app.generation.answer import answer_question  # noqa: E402
from app.generation.cache import cached_answer, clear_cache, connect  # noqa: E402
from app.models.answers import Answer, RetrievalStats  # noqa: E402
from app.retrieval.search import default_embedder  # noqa: E402

HISTORY = Path("data/eval/cache.jsonl")
QUESTIONS = Path("data/eval/questions.jsonl")
PAIRS = Path("data/eval/cache_pairs.jsonl")
MARGIN = 0.02  # pre-registered in the spec: above the edge, not on it
MIN_PARAPHRASE_HITS = 19  # >= 50 % of 38


def anchors() -> list[EvalQuestion]:
    return [q for q in load_dataset(QUESTIONS) if not q.held_out and q.category != "unanswerable"]


def pairs() -> list[CachePair]:
    return [p for p in load_dataset(PAIRS, model=CachePair) if isinstance(p, CachePair)]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5
    return dot / norm


def sweep() -> dict[str, Any]:
    embed = default_embedder(get_settings())
    anchor_vectors = {a.question_id: embed(a.question) for a in anchors()}
    per_probe = []
    for pair in pairs():
        vector = embed(pair.question)
        scores = {aid: cosine(vector, av) for aid, av in anchor_vectors.items()}
        nearest = max(scores, key=scores.__getitem__)
        per_probe.append(
            {
                "id": pair.question_id,
                "kind": pair.kind,
                "anchor_id": pair.anchor_id,
                "own_cosine": scores[pair.anchor_id],
                "nearest_id": nearest,
                "nearest_cosine": scores[nearest],
            }
        )
    would_be_false = [
        p["nearest_cosine"]
        for p in per_probe
        if p["kind"] != "paraphrase" or p["nearest_id"] != p["anchor_id"]
    ]
    threshold = max(would_be_false) + MARGIN
    false_hits = sum(1 for c in would_be_false if c >= threshold)
    paraphrase_hits = sum(
        1
        for p in per_probe
        if p["kind"] == "paraphrase"
        and p["nearest_id"] == p["anchor_id"]
        and p["nearest_cosine"] >= threshold
    )
    return {
        "candidate_threshold": threshold,
        "false_hits": false_hits,
        "must_miss": len(would_be_false),
        "paraphrase_hits": paraphrase_hits,
        "rule_s": false_hits == 0 and paraphrase_hits >= MIN_PARAPHRASE_HITS,
        "per_probe": per_probe,
    }


def _no_spend(question: str, **_: Any) -> Answer:
    """The semantic arm's answerer: a miss must not cost a generation call, and a
    refusal is never stored, so a miss leaves the cache exactly as it found it."""
    return Answer(
        answer="(miss)",
        retrieval=RetrievalStats(retrieved=0, used=0, dropped=0),
        latency_ms=0.0,
        model="none",
        refusal="no_context",
    )


def arm(name: str, price_in: float | None, price_out: float | None) -> dict[str, Any]:
    settings = get_settings()
    if not settings.redis_url:
        raise SystemExit("REDIS_URL is empty; the arms measure the cache")
    if name == "cold":
        settings = settings.model_copy(update={"cache_semantic_threshold": None})
    client = connect(settings)
    client.ping()
    if name == "cold":
        clear_cache(client)
    if name == "semantic":
        if settings.cache_semantic_threshold is None:
            raise SystemExit("Rule S did not pass; there is no semantic arm to run")
        probes = [(p.question_id, p.question, p.kind, p.anchor_id) for p in pairs()]
        answerer = _no_spend
    else:
        probes = [(a.question_id, a.question, "anchor", a.question_id) for a in anchors()]
        answerer = answer_question

    rows = []
    for probe_id, text, kind, anchor_id in probes:
        started = time.perf_counter()
        answer = cached_answer(text, client=client, answerer=answerer, settings=settings)
        rows.append(
            {
                "id": probe_id,
                "kind": kind,
                "anchor_id": anchor_id,
                "cache": answer.cache,
                "answer": answer.answer,
                "wall_ms": (time.perf_counter() - started) * 1000,
                "prompt_tokens": answer.usage.get("prompt_tokens", 0),
                "completion_tokens": answer.usage.get("completion_tokens", 0),
                "refusal": answer.refusal,
                "warnings": answer.warnings,
            }
        )
        print(
            f"{probe_id:<10} {kind:<13} {answer.cache or 'miss':<9} {rows[-1]['wall_ms']:7.0f} ms"
        )

    hits = [r for r in rows if r["cache"]]
    misses = [r for r in rows if not r["cache"]]
    hit_p50, hit_p95 = percentiles([r["wall_ms"] for r in hits])
    miss_p50, miss_p95 = percentiles([r["wall_ms"] for r in misses])
    tokens_in = sum(r["prompt_tokens"] for r in misses)
    tokens_out = sum(r["completion_tokens"] for r in misses)
    usd = (
        (tokens_in * price_in + tokens_out * price_out) / 1e6
        if price_in is not None and price_out is not None
        else None
    )
    return {
        "hits": len(hits),
        "misses": len(misses),
        "hit_p50_ms": hit_p50,
        "hit_p95_ms": hit_p95,
        "miss_p50_ms": miss_p50,
        "miss_p95_ms": miss_p95,
        "prompt_tokens": tokens_in,
        "completion_tokens": tokens_out,
        "usd_at_list_price": usd,
        "per_probe": rows,
    }


def verdicts(name: str, row: dict[str, Any]) -> dict[str, Any]:
    if name == "cold":
        return {}
    past = [json.loads(line) for line in HISTORY.read_text(encoding="utf-8").splitlines()]
    if name == "exact":
        cold = next(r for r in reversed(past) if r["label"] == "cache-cold")
        stored = {r["id"]: r["answer"] for r in cold["per_probe"] if r["refusal"] is None}
        served = {r["id"]: r["answer"] for r in row["per_probe"] if r["cache"] == "exact"}
        return {
            # Every answer cold stored comes back, byte-identical; refusals were
            # never stored, so they are misses by design, not failures.
            "rule_e": served == stored and row["hit_p50_ms"] < 50,
            "stored": len(stored),
            "cold_refusals": len(cold["per_probe"]) - len(stored),
        }
    if name == "semantic":
        swept = next(r for r in reversed(past) if r["label"] == "cache-sweep")
        cold = next(r for r in reversed(past) if r["label"] == "cache-cold")
        # An anchor cold refused was never stored, so its paraphrase cannot hit.
        stored = {r["id"] for r in cold["per_probe"] if r["refusal"] is None}
        t = swept["candidate_threshold"]
        predicted = {
            p["id"]
            for p in swept["per_probe"]
            if p["kind"] == "paraphrase"
            and p["anchor_id"] in stored
            and p["nearest_id"] == p["anchor_id"]
            and p["nearest_cosine"] >= t
        }
        served = {r["id"] for r in row["per_probe"] if r["cache"] == "semantic"}
        false_hits = [r["id"] for r in row["per_probe"] if r["cache"] and r["kind"] != "paraphrase"]
        # A paraphrase served its own anchor's answer iff the answer text matches cold's.
        by_anchor = {r["id"]: r["answer"] for r in cold["per_probe"]}
        wrong_anchor = [
            r["id"]
            for r in row["per_probe"]
            if r["kind"] == "paraphrase" and r["cache"] and r["answer"] != by_anchor[r["anchor_id"]]
        ]
        return {
            "semantic_check": served == predicted and not false_hits and not wrong_anchor,
            "predicted_not_served": sorted(predicted - served),
            "served_not_predicted": sorted(served - predicted),
            "false_hits": false_hits,
            "wrong_anchor": wrong_anchor,
        }
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--arm", choices=["cold", "exact", "semantic"])
    parser.add_argument("--usd-per-mtok-in", type=float)
    parser.add_argument("--usd-per-mtok-out", type=float)
    parser.add_argument("--no-save", action="store_true", help="print only, append nothing")
    args = parser.parse_args()
    if args.sweep == bool(args.arm):
        parser.error("choose exactly one of --sweep or --arm")
    if args.arm:
        past = (
            [json.loads(line) for line in HISTORY.read_text(encoding="utf-8").splitlines()]
            if HISTORY.exists()
            else []
        )
        label = f"cache-{args.arm}"
        if any(row["label"] == label for row in past):
            parser.error(f"{label} already exists; refusing to repeat a billed arm")
        if args.arm != "cold" and not any(row["label"] == "cache-cold" for row in past):
            parser.error("run the cold arm first")
        settings = get_settings()
        result = arm(args.arm, args.usd_per_mtok_in, args.usd_per_mtok_out)
        result.update(verdicts(args.arm, result))
        row = {
            "label": label,
            "timestamp": datetime.now(UTC).isoformat(),
            "git_commit": git_commit(),
            "config": {
                "generation_model": settings.generation_model,
                "embedding_model": settings.embedding_model,
                "prompt_version": settings.prompt_version,
                "compress_method": settings.compress_method,
                "cache_semantic_threshold": settings.cache_semantic_threshold,
                "usd_per_mtok_in": args.usd_per_mtok_in,
                "usd_per_mtok_out": args.usd_per_mtok_out,
            },
            **result,
        }
        print(json.dumps({k: v for k, v in result.items() if k != "per_probe"}, indent=2))
        if not args.no_save:
            with HISTORY.open("a", encoding="utf-8") as history:
                history.write(json.dumps(row, ensure_ascii=False) + "\n")
        return 0

    result = sweep()
    kinds = ("paraphrase", "near_miss", "unanswerable")
    for kind in kinds:
        own = sorted(p["own_cosine"] for p in result["per_probe"] if p["kind"] == kind)
        p50, p95 = percentiles(own)
        print(
            f"{kind:<13} own-anchor cosine  min {own[0]:.3f}  p50 {p50:.3f}"
            f"  p95 {p95:.3f}  max {own[-1]:.3f}"
        )
    wrong = [
        p["id"]
        for p in result["per_probe"]
        if p["kind"] == "paraphrase" and p["nearest_id"] != p["anchor_id"]
    ]
    print(f"paraphrases nearer another anchor: {len(wrong)} {wrong}")
    print(f"candidate threshold {result['candidate_threshold']:.4f}")
    print(f"false hits          {result['false_hits']} of {result['must_miss']}")
    print(f"paraphrase hits     {result['paraphrase_hits']} of 38 (need {MIN_PARAPHRASE_HITS})")
    print(f"Rule S              {'PASS' if result['rule_s'] else 'FAIL'}")

    row = {
        "label": "cache-sweep",
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "config": {"embedding_model": get_settings().embedding_model, "margin": MARGIN},
        **result,
    }
    if not args.no_save:
        with HISTORY.open("a", encoding="utf-8") as history:
            history.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended to {HISTORY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
