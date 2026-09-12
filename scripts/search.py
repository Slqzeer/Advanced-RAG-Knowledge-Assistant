"""Look at what the index actually returns.

    uv run python scripts/search.py "question" [--top-k 5] [--filter doc_type=tutorial]

Eyeballing results is not a substitute for step 11's metrics, but it is how you
notice that a keyword query retrieves nothing useful before you build four
steps on top of it.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.search import parse_filters, search  # noqa: E402

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
    args = parser.parse_args()
    # The corpus is full of emoji and the Windows console defaults to cp1252.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    started = time.perf_counter()
    results = search(args.query, top_k=args.top_k, filters=parse_filters(args.filter))
    elapsed = time.perf_counter() - started

    print(f'"{args.query}"  ->  {len(results)} results in {elapsed * 1000:.0f} ms\n')
    for scored in results:
        chunk = scored.chunk
        text = " ".join(chunk.text.split())
        print(f"{scored.rank}. {scored.score:.4f}  {chunk.document_id}  [{chunk.section or '-'}]")
        print(f"   {text[:PREVIEW]}{'...' if len(text) > PREVIEW else ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
