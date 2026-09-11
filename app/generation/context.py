"""Ranked chunks in, one numbered block of text out. No I/O, no model, no state.

Everything here is a pure function of its arguments, which is why the tests are
exhaustive: this is the last place the corpus is still verbatim. After it, the
text is inside a prompt and any damage done here is indistinguishable from the
model being wrong.
"""

from collections.abc import Sequence

from app.models.answers import Source
from app.models.chunks import ScoredChunk

# ~3 000 tokens for top_k=5. Characters, not tokens: a tokeniser is a dependency
# and a per-model one at that.
# ponytail: step 20 replaces this with a real token budget when compression
# arrives and the difference starts costing money.
MAX_CONTEXT_CHARS = 12_000
SEPARATOR = "\n\n"


def _header(index: int, chunk: ScoredChunk) -> str:
    """``[1] fastapi — Dependencies / First steps``.

    The number comes first because that is what the model is asked to cite.
    ``source`` is in there for step 13, when more than one corpus is indexed and
    "which docs said that?" stops being obvious.
    """
    title = f"{chunk.chunk.source} — {chunk.chunk.title}"
    if chunk.chunk.section:
        title += f" / {chunk.chunk.section}"
    return f"[{index}] {title}"


def build_context(
    chunks: Sequence[ScoredChunk], *, max_chars: int = MAX_CONTEXT_CHARS
) -> tuple[str, list[Source], int]:
    """Return ``(context_block, sources, dropped)`` for chunks in rank order.

    Chunks are kept whole or not at all. Over budget, the lowest-ranked ones are
    dropped — except the first, which is kept even if it alone blows the budget:
    an empty context makes a wrong answer certain, an oversized one merely makes
    a long prompt.
    """
    entries: list[str] = []
    sources: list[Source] = []
    total = 0

    for index, scored in enumerate(chunks, start=1):
        entry = f"{_header(index, scored)}\n{scored.chunk.text}"
        cost = len(entry) + (len(SEPARATOR) if entries else 0)
        if entries and total + cost > max_chars:
            break
        entries.append(entry)
        total += cost
        sources.append(
            Source(
                index=index,
                document_id=scored.chunk.document_id,
                title=scored.chunk.title,
                url=scored.chunk.url,
                section=scored.chunk.section,
                chunk_id=scored.chunk.chunk_id,
                score=scored.score,
            )
        )

    return SEPARATOR.join(entries), sources, len(chunks) - len(entries)
