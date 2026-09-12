"""Split cleaned documents into overlapping chunks.

Four strategies behind one registry, all with the same ``text -> [(start, end)]``
contract so a benchmark run picks one by name and nothing downstream changes:

- ``recursive`` — try the most semantic separator first, recurse into pieces
  still too long with the next one. Code is never cut open: every span the step
  03 regex calls code is a place no boundary may land.
- ``fixed`` — hard cuts on the stride. The control, and deliberately naive.
- ``sentence`` — whole sentences packed to the size cap.
- ``semantic`` — sentences, cut where adjacent ones stop resembling each other.

Offsets rather than strings throughout: they make the coverage check possible,
keep one copy of the text in memory, and let the reconstruction property — drop
each overlap, get the original text back — be asserted against all four.

Sizes are in characters, not tokens (~4 chars per token for English prose, worse
for code). A token-accurate unit only matters when a context limit binds; step
20 revisits it.
"""

import logging
import math
import re
from collections.abc import Callable, Iterable
from itertools import pairwise
from statistics import quantiles
from typing import Any

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
# A sentence boundary candidate: terminal punctuation, its closing quotes, and
# the whitespace behind it, so the next sentence opens on a real character.
SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s+")
WORD_BEFORE = re.compile(r"[^\s(\[]*\Z")
# Words that end in a dot mid-sentence. Version numbers need no entry here:
# `0.115.2` has no dot followed by whitespace, so it never matches at all.
ABBREVIATIONS = frozenset({"etc", "vs", "cf", "al", "Dr", "Mr", "Mrs", "Ms", "St", "Fig", "No"})


def _check_overlap(chunk_size: int, overlap: int) -> None:
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")


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
    _check_overlap(chunk_size, overlap)
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


def split_fixed(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[tuple[int, int]]:
    """Hard cuts every ``chunk_size - overlap`` characters: mid-word, mid-fence.

    Deliberately naive, and it must stay that way. It is the floor the other
    three are measured against, and a ``fixed`` that quietly protected code would
    take the bottom out of the comparison. Do not teach it about separators.
    """
    _check_overlap(chunk_size, overlap)
    if not text.strip():
        return []

    spans: list[tuple[int, int]] = []
    position = 0
    while position < len(text):
        end = min(position + chunk_size, len(text))
        spans.append((position, end))
        if end == len(text):
            break
        position += chunk_size - overlap
    return spans


def _is_abbreviation(text: str, dot: int) -> bool:
    """True for ``e.g.``, ``U.S.`` and the short list of words that end in a dot."""
    match = WORD_BEFORE.search(text[max(0, dot - 12) : dot])
    word = match.group() if match else ""
    return word in ABBREVIATIONS or (len(word) > 1 and word[-2] == "." and word[-1].isalpha())


def _sentence_units(text: str, protected: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """One span per sentence, tiling ``text``. A boundary inside code is not one."""
    units: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        cut = match.end()
        if (
            cut <= start
            or _skip_protected(cut, protected) != cut
            or _is_abbreviation(text, match.start())
        ):
            continue
        units.append((start, cut))
        start = cut
    if start < len(text):
        units.append((start, len(text)))
    return units


def _cores(
    units: list[tuple[int, int]], stride: int, forced: frozenset[int] | set[int] = frozenset()
) -> list[tuple[int, int]]:
    """Index ranges over ``units``: close a chunk on a forced cut or on the size cap.

    Packed to ``stride`` rather than to ``chunk_size`` so the overlap snapped on
    afterwards still fits — the same arithmetic ``split_text`` uses.
    """
    cores: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(units)):
        if index in forced or units[index][1] - units[start][0] > stride:
            cores.append((start, index))
            start = index
    cores.append((start, len(units)))
    return cores


def _pack(
    units: list[tuple[int, int]],
    cores: list[tuple[int, int]],
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Core ranges to spans, each extended backwards by *whole* units for the overlap."""
    spans: list[tuple[int, int]] = []
    for position, (low, high) in enumerate(cores):
        begin = low
        if position:
            floor = cores[position - 1][0]
            while begin > floor and units[low][0] - units[begin - 1][0] <= overlap:
                begin -= 1
        start, end = units[begin][0], units[high - 1][1]
        if end - start > chunk_size:
            logger.warning("oversized chunk: %d characters, limit %d", end - start, chunk_size)
        spans.append((start, end))
    return spans


def split_sentences(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[tuple[int, int]]:
    """Whole sentences packed to ``chunk_size``, overlapping by whole sentences.

    A sentence longer than ``chunk_size`` becomes one oversized chunk: cutting it
    is precisely the thing this strategy exists not to do.
    """
    _check_overlap(chunk_size, overlap)
    if not text.strip():
        return []
    units = _sentence_units(text, _protected_spans(text))
    return _pack(units, _cores(units, chunk_size - overlap), chunk_size, overlap)


def _adjacent_distances(vectors: list[list[float]]) -> list[float]:
    """Cosine distance between each pair of neighbouring sentences."""
    norms = [math.hypot(*vector) for vector in vectors]
    distances: list[float] = []
    for (a, b), (left, right) in zip(pairwise(vectors), pairwise(norms), strict=True):
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        distances.append(1.0 - dot / (left * right) if left and right else 1.0)
    return distances


def split_semantic(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    *,
    embed_batch: Callable[[list[str]], list[list[float]]],
    percentile: int = 95,
) -> list[tuple[int, int]]:
    """Sentences packed as above, but cut early wherever two adjacent ones stop
    talking about the same thing.

    The threshold is this document's own ``percentile`` of adjacent-sentence
    cosine distance, never a global constant: 0.15 means one thing on a tutorial
    page and another on an API reference, and a percentile has no magic number to
    defend. ``embed_batch`` is an argument rather than an import so this module
    stays network-free and its tests need no key — and so the interesting
    question, where does it cut, is testable against scripted vectors.
    """
    _check_overlap(chunk_size, overlap)
    if not text.strip():
        return []

    units = _sentence_units(text, _protected_spans(text))
    # One call with every sentence: per-sentence calls would be the same tokens
    # at a hundred times the round trips.
    distances = _adjacent_distances(embed_batch([text[s:e] for s, e in units]))
    threshold = (
        quantiles(distances, n=100, method="inclusive")[percentile - 1]
        if len(distances) > 1
        else math.inf
    )
    forced = {index + 1 for index, distance in enumerate(distances) if distance > threshold}
    return _pack(units, _cores(units, chunk_size - overlap, forced), chunk_size, overlap)


Splitter = Callable[..., list[tuple[int, int]]]
# Step 12 measured all four on the step 10 question set: sentence won Recall@5
# 0.776 against recursive's 0.713. Mirrors `Settings.chunk_strategy`, as
# CHUNK_SIZE and CHUNK_OVERLAP mirror theirs.
DEFAULT_STRATEGY = "sentence"
# Selected by name from the command line, so a benchmark run names its splitter
# and `results.jsonl` records which one produced the number.
STRATEGIES: dict[str, Splitter] = {
    "recursive": split_text,
    "fixed": split_fixed,
    "sentence": split_sentences,
    "semantic": split_semantic,
}


def splitter(strategy: str) -> Splitter:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown chunking strategy {strategy!r}; expected one of {', '.join(STRATEGIES)}"
        )
    return STRATEGIES[strategy]


def chunk_document(
    document: RawDocument,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    strategy: str = DEFAULT_STRATEGY,
    **options: Any,
) -> list[Chunk]:
    """Split a cleaned document, carrying its metadata onto every chunk.

    ``options`` reaches the splitter untouched: ``semantic`` needs an
    ``embed_batch``, the other three need nothing.
    """
    text = document.text
    protected = _protected_spans(text)
    headings = [
        (m.start(), m.group(1).strip())
        for m in SECTION.finditer(text)
        if _skip_protected(m.start(), protected) == m.start()
    ]

    chunks: list[Chunk] = []
    for start, end in splitter(strategy)(text, chunk_size, overlap, **options):
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
    strategy: str = DEFAULT_STRATEGY,
    **options: Any,
) -> list[Chunk]:
    splitter(strategy)  # fail before the first document, not after the last
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size, overlap, strategy, **options)
    ]
