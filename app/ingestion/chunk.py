"""Split cleaned documents into overlapping chunks.

Recursive character splitting, hand-written: try the most semantic separator
first, recurse into pieces still too long with the next one. Code is never cut
open — every span the step 03 regex calls code is a place no boundary may land.

Sizes are in characters, not tokens (~4 chars per token for English prose, worse
for code). A token-accurate unit only matters when a context limit binds; step
20 revisits it.
"""

import logging
import re
from collections.abc import Iterable
from itertools import pairwise

from app.ingestion.clean import CODE
from app.models.chunks import Chunk
from app.models.documents import RawDocument

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# Most semantic first. The "" terminal guarantees termination on a blob with no
# separator at all, instead of recursing forever.
SEPARATORS = ("\n## ", "\n### ", "\n\n", "\n", ". ", " ", "")
# A heading belongs to the text below it, so the cut goes before it, not after.
HEADING_SEPARATORS = frozenset({"\n## ", "\n### "})
SECTION = re.compile(r"^#{2,6}[ \t]+(.+?)[ \t]*#*$", re.MULTILINE)


def _boundary(separator: str, index: int) -> int:
    """Where the cut falls for an occurrence of ``separator`` starting at ``index``."""
    return index + 1 if separator in HEADING_SEPARATORS else index + len(separator)


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Half-open spans no boundary may fall inside: fenced, indented and inline code."""
    return [(m.start(), m.end()) for m in CODE.finditer(text)]


def _skip_protected(position: int, protected: Iterable[tuple[int, int]]) -> int:
    """``position``, pushed to the end of the code span it lands inside, if any."""
    for start, end in protected:
        if start >= position:
            break
        if position < end:
            return end
    return position


def _split_region(
    text: str,
    start: int,
    end: int,
    limit: int,
    protected: list[tuple[int, int]],
    depth: int = 0,
) -> list[tuple[int, int]]:
    """Tile ``[start, end)`` with contiguous spans of at most ``limit`` characters."""
    if end - start <= limit:
        return [(start, end)]
    if depth >= len(SEPARATORS) - 1:
        # Terminal: no separator left, cut on size alone — but never inside code.
        # ponytail: an oversized code fence becomes one oversized chunk; revisit if
        # the eval set shows those hurting recall.
        spans: list[tuple[int, int]] = []
        position = start
        while position < end:
            cut = min(
                max(_skip_protected(min(position + limit, end), protected), position + 1), end
            )
            spans.append((position, cut))
            position = cut
        return spans

    separator = SEPARATORS[depth]
    cuts: list[int] = []
    search = start
    while (index := text.find(separator, search, end)) != -1:
        search = index + len(separator)
        cut = _boundary(separator, index)
        if (
            start < cut < end
            and cut != (cuts[-1] if cuts else start)
            and _skip_protected(cut, protected) == cut
        ):
            cuts.append(cut)
    if not cuts:
        return _split_region(text, start, end, limit, protected, depth + 1)

    out: list[tuple[int, int]] = []
    for left, right in pairwise([start, *cuts, end]):
        if right - left <= limit:
            out.append((left, right))
        else:
            out.extend(_split_region(text, left, right, limit, protected, depth + 1))
    return out


def _merge(spans: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    """Glue adjacent spans back together while they still fit, so a `\n` split does
    not leave one chunk per line."""
    out: list[tuple[int, int]] = []
    for start, end in spans:
        if out and end - out[-1][0] <= limit:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def _snap_back(text: str, core_start: int, overlap: int, protected: list[tuple[int, int]]) -> int:
    """Where the overlap starts: the earliest separator in the window, so the chunk
    opens on a heading, paragraph or sentence rather than mid-word."""
    window = max(0, core_start - overlap)
    for separator in SEPARATORS[:-1]:
        index = text.find(separator, window, core_start)
        if index == -1:
            continue
        cut = _boundary(separator, index)
        if window <= cut < core_start and _skip_protected(cut, protected) == cut:
            return cut
    return core_start


def split_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[tuple[int, int]]:
    """Offset pairs, one per chunk. Offsets rather than strings: they make the
    coverage check possible and keep one copy of the text in memory.

    Successive chunks share up to ``overlap`` characters; dropping each overlap
    reproduces ``text`` exactly.
    """
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if not text.strip():
        return []

    protected = _protected_spans(text)
    stride = chunk_size - overlap
    cores = _merge(_split_region(text, 0, len(text), stride, protected), stride)

    spans: list[tuple[int, int]] = []
    for position, (start, end) in enumerate(cores):
        begin = start if position == 0 else _snap_back(text, start, overlap, protected)
        if end - begin > chunk_size:
            logger.warning("oversized chunk: %d characters, limit %d", end - begin, chunk_size)
        spans.append((begin, end))
    return spans


def chunk_document(
    document: RawDocument, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """Split a cleaned document, carrying its metadata onto every chunk."""
    text = document.text
    protected = _protected_spans(text)
    headings = [
        (m.start(), m.group(1).strip())
        for m in SECTION.finditer(text)
        if _skip_protected(m.start(), protected) == m.start()
    ]

    chunks: list[Chunk] = []
    for start, end in split_text(text, chunk_size, overlap):
        body = text[start:end]
        if not body.strip():
            continue
        chunks.append(
            Chunk(
                document_id=document.document_id,
                source=document.source,
                title=document.title,
                url=document.url,
                language=document.language,
                section=next((t for p, t in reversed(headings) if p <= start), None),
                chunk_index=len(chunks),
                text=body,
                char_start=start,
                char_end=end,
            )
        )
    return chunks


def chunk_documents(
    documents: Iterable[RawDocument],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    return [c for d in documents for c in chunk_document(d, chunk_size, overlap)]
