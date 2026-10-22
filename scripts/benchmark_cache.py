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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_answers import git_commit, percentiles  # noqa: E402 - sibling script

from app.core.config import get_settings  # noqa: E402
from app.evaluation.dataset import CachePair, EvalQuestion, load_dataset  # noqa: E402
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="print only, append nothing")
    args = parser.parse_args()
    if not args.sweep:
        parser.error("choose --sweep")

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
