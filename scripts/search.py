"""Look at what the index actually returns.

    uv run python scripts/search.py "question" [--top-k 5] [--mode lexical]
                                              [--filter doc_type=tutorial]

Eyeballing results is not a substitute for step 11's metrics, but it is how you
notice that a keyword query retrieves nothing useful before you build four
steps on top of it.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.rerank import RERANKERS  # noqa: E402
from app.retrieval.search import RETRIEVERS, parse_filters, search  # noqa: E402
from app.retrieval.transform import TRANSFORMS  # noqa: E402

PREVIEW = 200


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable; e.g. --filter doc_type=tutorial --filter doc_type=tutorial,advanced",
    )
    parser.add_argument("--mode", choices=sorted(RETRIEVERS), help="default: RETRIEVAL_MODE")
    parser.add_argument(
        "--rerank",
        choices=["", *sorted(RERANKERS)],
        help='cross-encoder that reorders the pool; default: RERANK_MODEL, "" is off',
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        help="how deep the pool goes into the cross-encoder; default: RERANK_CANDIDATES",
    )
    parser.add_argument(
        "--transform",
        choices=["", *sorted(TRANSFORMS)],
        help='query transform applied before retrieval; default: QUERY_TRANSFORM, "" is off',
    )
    parser.add_argument(
        "--transform-n", type=int, help="queries `multi` produces, original included; MULTI_QUERY_N"
    )
    args = parser.parse_args()
    # The corpus is full of emoji and the Windows console defaults to cp1252.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    started = time.perf_counter()
    results = search(
        args.query,
        top_k=args.top_k,
        mode=args.mode,
        rerank=args.rerank,
        rerank_candidates=args.rerank_candidates,
        transform=args.transform,
        transform_n=args.transform_n,
        filters=parse_filters(args.filter),
    )
    elapsed = time.perf_counter() - started

    print(f'"{args.query}"  ->  {len(results)} results in {elapsed * 1000:.0f} ms\n')
    for scored in results:
        chunk = scored.chunk
        text = " ".join(chunk.text.split())
        # The asterisk marks a rerank score — a different scale from a cosine, and
        # comparing the two by eye is the mistake it exists to prevent.
        score = (
            f"{scored.rerank_score:.4f}*"
            if scored.rerank_score is not None
            else f"{scored.score:.4f}"
        )
        print(f"{scored.rank}. {score}  {chunk.document_id}  [{chunk.section or '-'}]")
        print(f"   {text[:PREVIEW]}{'...' if len(text) > PREVIEW else ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
