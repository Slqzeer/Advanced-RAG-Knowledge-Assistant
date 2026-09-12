"""Index the corpus into Qdrant: raw -> clean -> chunk -> embed -> upsert.

    uv run python scripts/index_corpus.py [--source fastapi] [--limit 20] [--dry-run] [--recreate]
    uv run python scripts/index_corpus.py --strategy semantic --collection chunks_semantic

Re-running is free and idempotent: the step 05 cache serves the vectors and the
deterministic point ids overwrite the same points instead of adding new ones.
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetch_corpus import SOURCES  # noqa: E402 - sibling script, after the path fix

from app.core.config import get_settings  # noqa: E402
from app.ingestion.chunk import STRATEGIES, chunk_documents  # noqa: E402
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
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), help="default: CHUNK_STRATEGY")
    parser.add_argument("--collection", help="default: QDRANT_COLLECTION — one per strategy")
    parser.add_argument("--chunk-size", type=int, help="characters; default: CHUNK_SIZE")
    parser.add_argument("--overlap", type=int, help="characters; default: CHUNK_OVERLAP")
    args = parser.parse_args()

    settings = get_settings()
    strategy = args.strategy or settings.chunk_strategy
    collection = args.collection or settings.qdrant_collection
    chunk_size = args.chunk_size or settings.chunk_size
    overlap = settings.chunk_overlap if args.overlap is None else args.overlap
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

    # One cache for the sentence vectors `semantic` needs and the chunk vectors
    # every strategy needs: keyed on sha256(model + text), so identical spans
    # across strategies hit and only genuinely new text costs anything.
    cache = EmbeddingCache(settings.embedding_cache_path)
    embed = partial(embed_texts, model=settings.embedding_model, cache=cache)
    options = {"embed_batch": embed} if strategy == "semantic" else {}

    chunks = chunk_documents(cleaned, chunk_size, overlap, strategy, **options)
    started = stage("chunk", len(chunks), started)
    if not chunks:
        raise SystemExit("nothing to index")

    vectors = embed([chunk.text for chunk in chunks])
    started = stage("embed", len(vectors), started)

    # The parameters on screen, not just in the shell history: a chunk count with
    # no strategy next to it cannot be attributed once five runs are done.
    print(f"  strategy   {strategy} @ {chunk_size}/{overlap} -> {collection}")

    if args.dry_run:
        print(f"  dry run, nothing written  ({time.perf_counter() - total:.1f}s total)")
        return 0

    client = get_client(settings)
    ensure_collection(client, collection, len(vectors[0]), recreate=args.recreate)
    count = upsert_chunks(client, collection, chunks, vectors)
    stage("upsert", count, started)

    stats = collection_stats(client, collection)
    print(
        f"  {collection}: {stats['points']} points, "
        f"dim {stats['vector_size']}, {stats['distance']}  "
        f"({time.perf_counter() - total:.1f}s total)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
