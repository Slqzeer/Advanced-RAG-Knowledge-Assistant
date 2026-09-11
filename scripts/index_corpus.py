"""Index the corpus into Qdrant: raw -> clean -> chunk -> embed -> upsert.

    uv run python scripts/index_corpus.py [--source fastapi] [--limit 20] [--dry-run] [--recreate]

Re-running is free and idempotent: the step 05 cache serves the vectors and the
deterministic point ids overwrite the same points instead of adding new ones.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetch_corpus import SOURCES  # noqa: E402 - sibling script, after the path fix

from app.core.config import get_settings  # noqa: E402
from app.ingestion.chunk import chunk_documents  # noqa: E402
from app.ingestion.clean import clean_document  # noqa: E402
from app.ingestion.embed import EmbeddingCache, embed_texts  # noqa: E402
from app.ingestion.loader import load_documents  # noqa: E402
from app.retrieval.store import collection_stats, ensure_collection, get_client, upsert_chunks

BASE_URLS = {source.name: source.base_url for source in SOURCES}


def stage(label: str, count: int, started: float) -> float:
    """Print one summary line and return a fresh start time for the next stage."""
    print(f"  {label:<10} {count:>7}  {time.perf_counter() - started:6.1f}s")
    return time.perf_counter()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", help="repeatable; default: every corpus dir")
    parser.add_argument(
        "--limit", type=int, help="stop after N documents — a smoke run that costs nothing"
    )
    parser.add_argument("--dry-run", action="store_true", help="everything except the upsert")
    parser.add_argument("--recreate", action="store_true", help="drop the collection first")
    args = parser.parse_args()

    settings = get_settings()
    sources = args.source or sorted(p.name for p in settings.corpus_dir.iterdir() if p.is_dir())
    if not sources:
        raise SystemExit(f"no corpus under {settings.corpus_dir}; run scripts/fetch_corpus.py")

    started = total = time.perf_counter()
    documents = [
        document
        for name in sources
        for document in load_documents(settings.corpus_dir / name, name, BASE_URLS.get(name))
    ]
    if args.limit:
        documents = documents[: args.limit]
    started = stage("load", len(documents), started)

    cleaned = [c for d in documents if (c := clean_document(d)) is not None]
    started = stage("clean", len(cleaned), started)

    chunks = chunk_documents(cleaned, settings.chunk_size, settings.chunk_overlap)
    started = stage("chunk", len(chunks), started)
    if not chunks:
        raise SystemExit("nothing to index")

    vectors = embed_texts(
        [chunk.text for chunk in chunks],
        model=settings.embedding_model,
        cache=EmbeddingCache(settings.embedding_cache_path),
    )
    started = stage("embed", len(vectors), started)

    if args.dry_run:
        print(f"  dry run, nothing written  ({time.perf_counter() - total:.1f}s total)")
        return 0

    client = get_client(settings)
    ensure_collection(client, settings.qdrant_collection, len(vectors[0]), recreate=args.recreate)
    count = upsert_chunks(client, settings.qdrant_collection, chunks, vectors)
    stage("upsert", count, started)

    stats = collection_stats(client, settings.qdrant_collection)
    print(
        f"  {settings.qdrant_collection}: {stats['points']} points, "
        f"dim {stats['vector_size']}, {stats['distance']}  "
        f"({time.perf_counter() - total:.1f}s total)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
