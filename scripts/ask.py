"""Ask the corpus a question and read the answer it grounds in.

    uv run python scripts/ask.py "question" [--top-k 5] [--mode hybrid]
                                           [--filter doc_type=tutorial] [--show-context]

``--show-context`` prints the exact prompt that was sent. It is the debugging
tool you will reach for every time an answer looks wrong: nine times out of ten
retrieval is the problem and the prompt says so at a glance.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.generation.answer import answer_question  # noqa: E402
from app.generation.context import build_context  # noqa: E402
from app.generation.llm import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402
from app.retrieval.search import RETRIEVERS, parse_filters, search  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable; e.g. --filter doc_type=tutorial --filter doc_type=tutorial,advanced",
    )
    parser.add_argument("--mode", choices=sorted(RETRIEVERS), help="default: RETRIEVAL_MODE")
    parser.add_argument(
        "--show-context", action="store_true", help="print the prompt that was sent"
    )
    parser.add_argument(
        "--strict", action="store_true", help="fail on an invented citation instead of warning"
    )
    args = parser.parse_args()
    # The corpus is full of emoji and the Windows console defaults to cp1252.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    filters = parse_filters(args.filter)

    if args.show_context:
        # Rebuilt rather than returned by answer_question: the prompt is a
        # debugging artefact, not part of the API contract step 25 serves.
        chunks = search(args.question, top_k=args.top_k or 5, mode=args.mode, filters=filters)
        context, _, _ = build_context(chunks)
        print(f"--- system ---\n{SYSTEM_PROMPT}\n")
        print(f"--- user ---\n{USER_TEMPLATE.format(context=context, question=args.question)}\n")

    answer = answer_question(
        args.question, top_k=args.top_k, mode=args.mode, filters=filters, strict=args.strict
    )

    print(f"{answer.answer}\n")
    # Above the sources, not below: a warning under a tidy citation list is a
    # warning nobody reads. Every number below appears in the answer, and every
    # number in the answer appears below — that is what step 09 buys.
    for warning in answer.warnings:
        print(f"!! {warning}")
    if answer.warnings:
        print()
    for source in answer.sources:
        section = f" / {source.section}" if source.section else ""
        print(f"[{source.index}] {source.title}{section}  ({source.score:.4f})")
        print(f"    {source.url or source.chunk_id}")

    stats = answer.retrieval
    tokens = answer.usage.get("total_tokens", 0)
    print(
        f"\n{stats.retrieved} retrieved, {stats.used} used, {stats.dropped} dropped,"
        f" {len(answer.sources)} cited"
        f"  |  {answer.model}  |  {tokens} tokens  |  {answer.latency_ms:.0f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
