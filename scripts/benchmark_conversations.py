"""Does resolving a follow-up against its conversation actually retrieve better?

    uv run python scripts/benchmark_conversations.py --label conv-raw --raw
    uv run python scripts/benchmark_conversations.py --label conv-rewrite

Two runs, one delta. `--raw` searches the bare follow-up; without it the
follow-up is resolved by `contextualize` first. The gap between the two rows is
the number step 18 exists to produce, and it lands in the same
`data/eval/results.jsonl` as every other row.

Separate from `scripts/benchmark.py` because a conversation is a different
shape, not a different flag: `run_benchmark`'s retriever seam is
`str -> list[ScoredChunk]`, and four steps are built on that. The history is
bound into the closure and keyed on the question text, exactly as
`--oracle-filter` already does.

Ten conversations is a capability check, not a promotion criterion. Nothing
flips QUERY_TRANSFORM on the strength of this file.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.benchmark import run_benchmark  # noqa: E402
from app.evaluation.dataset import EvalConversation, load_dataset  # noqa: E402
from app.models.chunks import ScoredChunk  # noqa: E402
from app.retrieval.search import search  # noqa: E402
from app.retrieval.transform import contextualize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import HISTORY, KS, load_history, print_comparison, print_result  # noqa: E402

DEFAULT_DATASET = Path("data/eval/conversations.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--raw", action="store_true", help="search the bare follow-up, resolving nothing"
    )
    parser.add_argument("--compare", help="print a delta table against a previous label")
    parser.add_argument("--no-save", action="store_true", help="print only, append nothing")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    settings = get_settings()
    conversations = load_dataset(args.dataset, model=EvalConversation)
    # Keyed on the question text because run_benchmark hands the retriever a
    # string, exactly as --oracle-filter is. Exact as long as no two follow-ups
    # share a text, which the guard below enforces.
    histories = {c.question: getattr(c, "history", []) for c in conversations}
    if len(histories) != len(conversations):
        raise SystemExit("two follow-ups share the same text; the history lookup would be wrong")

    def retrieve(text: str) -> list[ScoredChunk]:
        query = text if args.raw else contextualize(text, histories[text], settings=settings)
        return search(query, top_k=args.top_k, settings=settings)

    result = run_benchmark(
        conversations,
        retrieve,
        ks=[k for k in KS if k <= args.top_k],
        label=args.label,
        config={
            "top_k": args.top_k,
            "mode": settings.retrieval_mode,
            "contextualize": not args.raw,
            "dataset": str(args.dataset),
            "collection": settings.qdrant_collection,
            "generation_model": settings.generation_model,
            "history_turns": settings.history_turns,
        },
    )
    print_result(result)

    status = 0
    if args.compare:
        status = print_comparison(result, args.compare, load_history(HISTORY))
    if not args.no_save:
        with HISTORY.open("a", encoding="utf-8") as history:
            history.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        print(f"\nappended to {HISTORY}")
    return status


if __name__ == "__main__":
    sys.exit(main())
