"""Ranked chunks in, shorter ranked chunks out. The last stage before the prompt.

``information.md`` frames phase 9 as token reduction. On this corpus that
measures nothing: five ``sentence`` chunks are roughly 4 000 characters against a
``MAX_CONTEXT_CHARS`` of 12 000 that has never once bound, and 40 % off a
1 000-token prompt is $0.0001.

What compression is for *here* is the gap steps 17-19 proved no ranking stage can
close. Dense retrieval reaches 0.785 Recall@5 and 0.884 Recall@20; FlashRank
recovered +0.002 of it for 1 141 ms and multi-query only reordered the pool.
A stage that makes a chunk *cheaper* is the one remaining way rank 6-20 reaches
the model at today's budget — and half that gap is ``multi_doc``, which is short
of distinct documents per character, which is exactly what this buys.

A ``Compressor`` scores units and nothing else. ``compress`` owns splitting,
budgeting, re-assembly and the frozen-model rebuild, for the same reason
``expand()`` owns parsing and capping for both transforms: the only thing a
future BM25 or LLMLingua entry changes is how a sentence gets a number.
"""

import math
from collections.abc import Callable, Sequence

from app.core.config import Settings, get_settings
from app.ingestion.chunk import sentence_spans
from app.ingestion.embed import EmbeddingCache, embed_texts
from app.models.chunks import ScoredChunk

# Deliberately not `search.Embedder`, which is `Callable[[str], list[float]]` and
# embeds one query. This embeds every sentence of twenty chunks and must do it in
# one batched call; sharing the name would hide a different shape behind a
# familiar one.
BatchEmbedder = Callable[[Sequence[str]], list[list[float]]]

# One score per unit, in input order. Scoring only — see the module docstring.
Compressor = Callable[[str, Sequence[str], Settings, BatchEmbedder | None], list[float]]

# Mirrors RETRIEVERS, RERANKERS, TRANSFORMS and STRATEGIES: the registry is how
# this project compares N variants and promotes a winner, and it is what makes
# "add LLMLingua later" one function rather than a refactor.
COMPRESSORS: dict[str, Compressor] = {}

# Marks where text was cut. Without it the model reads two non-adjacent sentences
# as contiguous prose, and the confident answer stitched from two unrelated
# clauses is indistinguishable from a hallucination in the output. One string,
# and it is the difference between a compressed context and a misleading one.
GAP = " […] "


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Written out rather than assumed.

    ``text-embedding-3-*`` returns normalised vectors today, so a dot product
    would agree — and a BGE-family model later would not, silently.
    """
    norm = math.hypot(*left) * math.hypot(*right)
    return sum(x * y for x, y in zip(left, right, strict=True)) / norm if norm else 0.0


def _default_embedder(settings: Settings) -> BatchEmbedder:
    """The same sqlite cache the corpus was embedded with, so a repeated arm is free."""
    cache = EmbeddingCache(settings.embedding_cache_path)
    return lambda texts: embed_texts(list(texts), model=settings.embedding_model, cache=cache)


def _embedding(
    query: str, units: Sequence[str], settings: Settings, embedder: BatchEmbedder | None
) -> list[float]:
    """Cosine of each unit against the query, in one batched call.

    The query rides along in the same batch rather than in a call of its own: it
    is one more string, it lands in the same cache, and two calls where one will
    do is two chances for a transient error inside a 38-question run.
    """
    vectors = (embedder or _default_embedder(settings))([query, *units])
    return [_cosine(vectors[0], vector) for vector in vectors[1:]]


COMPRESSORS["embedding"] = _embedding


def _texts(units: Sequence[tuple[int, int, str]], kept: set[int]) -> dict[int, str]:
    """Chunk index -> compressed text: survivors in *document* order, GAP at cuts.

    Selection is by score; rendering is by position. Emitting in score order
    produces a paragraph that contradicts itself across sentence boundaries.
    """
    by_chunk: dict[int, list[int]] = {}
    for index in sorted(kept):  # units were built in document order
        by_chunk.setdefault(units[index][0], []).append(index)

    out: dict[int, str] = {}
    for chunk_index, indices in by_chunk.items():
        parts: list[str] = []
        for position, index in enumerate(indices):
            if position and units[index][1] != units[indices[position - 1]][1] + 1:
                # A unit span carries the whitespace that followed its sentence.
                # Left in place it would print as "one.  […] two", so the marker
                # supplies the spacing on both sides and the seam supplies none.
                parts[-1] = parts[-1].rstrip()
                parts.append(GAP)
            parts.append(units[index][2])
        text = "".join(parts).strip()
        if text:  # Chunk.text is min_length=1; an all-whitespace unit set is a drop
            out[chunk_index] = text
    return out


def compress(
    chunks: Sequence[ScoredChunk],
    query: str,
    *,
    method: str | None = None,
    budget_chars: int | None = None,
    settings: Settings | None = None,
    embedder: BatchEmbedder | None = None,
) -> list[ScoredChunk]:
    """The chunks, with only the sentences that answer ``query``, within a budget.

    ``method`` names an entry in ``COMPRESSORS``. ``None`` reads
    ``COMPRESS_METHOD`` and the empty string forces it off, which is how an
    uncompressed baseline stays runnable once a default flips — the convention
    ``rerank=`` and ``transform=`` already established.

    ``budget_chars`` counts **chunk text only**. Headers and the ``[n]`` numbering
    are added afterwards by ``build_context`` and are identical across arms, so
    excluding them keeps the fixed variable fixed. The number reported in the
    README is ``usage.prompt_tokens`` off the API response, not an estimate
    derived from this one.

    Rank order is preserved and never touched. Steps 16, 17 and 19 spent this
    project's entire ranking budget; a compressor that also reordered would
    confound the two in a single number.

    ``embedder`` is injectable so every unit test runs with no network, no key
    and no spend — the pattern ``answer_question``, ``search`` and ``expand`` use.
    """
    settings = settings or get_settings()
    method = settings.compress_method if method is None else method
    if not method:
        return list(chunks)
    if method not in COMPRESSORS:
        raise ValueError(f"unknown compressor {method!r}; have {sorted(COMPRESSORS)}")
    if not query.strip():
        # The empty string embeds fine and then scores every sentence equally,
        # which silently degrades to "keep the first ones that fit".
        raise ValueError("query is empty")
    if not chunks:
        return []

    budget = budget_chars or settings.compress_budget_chars
    if budget < 1:
        raise ValueError(f"budget_chars must be at least 1, got {budget}")

    # (chunk index, unit index within that chunk, text) for every unit.
    units = [
        (chunk_index, unit_index, scored.chunk.text[start:end])
        for chunk_index, scored in enumerate(chunks)
        for unit_index, (start, end) in enumerate(sentence_spans(scored.chunk.text))
    ]
    if sum(len(text) for _, _, text in units) <= budget:
        return list(chunks)  # nothing to cut, and nothing to pay an embedder for

    scores = COMPRESSORS[method](query, [text for _, _, text in units], settings, embedder)
    order = sorted(range(len(units)), key=lambda index: scores[index], reverse=True)

    # Greedy by score, charging each unit its own length. The first is kept even
    # if it alone blows the budget, for build_context's reason: an empty context
    # makes a wrong answer certain, an oversized one merely makes a long prompt.
    kept: set[int] = set()
    total = 0
    for index in order:
        cost = len(units[index][2])
        if kept and total + cost > budget:
            continue
        kept.add(index)
        total += cost

    # The assembled text also carries GAP markers, which the loop above did not
    # charge for. One trim pass enforces the real budget rather than an estimate
    # of it; it runs a handful of times at most.
    while len(kept) > 1 and sum(len(t) for t in _texts(units, kept).values()) > budget:
        kept.discard(min(kept, key=lambda index: scores[index]))

    texts = _texts(units, kept)
    return [
        # char_start/char_end are deliberately unchanged: they say where in the
        # document this chunk was cut from, which is what a citation needs and
        # what compression does not alter.
        scored.model_copy(update={"chunk": scored.chunk.model_copy(update={"text": texts[index]})})
        for index, scored in enumerate(chunks)
        if index in texts
    ]
