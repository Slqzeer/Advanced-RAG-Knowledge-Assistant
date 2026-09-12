"""Check that the answer cited what it claims to have cited.

Post-validation, not prompt hope. The prompt asks the model to cite; this module
parses what came back and reconciles it against the context that was supplied.
An unverified ``[1]`` is worse than no citation: it looks like provenance and is
not, and thirty lines here turn a claim into a guarantee.

Pure functions over strings and ``Source`` objects — no model, no I/O.
"""

import re
from collections.abc import Sequence

from app.ingestion.clean import CODE, LINK
from app.models.answers import Source

# ``[1]``, ``[1, 3]``, ``[1-3]``. Digits capped at three: a context entry number
# is single-digit in practice, and the cap keeps "[1-99999999]" from expanding
# into a range the size of a small country. Anything longer is not a citation.
CITATION = re.compile(r"\[[ \t]*(\d{1,3}(?:[ \t]*[,;–-][ \t]*\d{1,3})*)[ \t]*\]")
NUMBER = re.compile(r"\d{1,3}")
# Marks where a marker was removed, so the space it took goes with it. NUL never
# survives cleaning, so it cannot occur in a context entry the model echoes back.
GAP = "\x00"

# A refusal has nothing to cite and is the correct answer, so it must not be
# flagged as ungrounded. One list, reused: step 12's guardrails need the same one.
# ponytail: substring matching, English and French only. Replace with the
# structured refusal signal if step 22 makes refusal a first-class outcome.
REFUSAL_MARKERS = (
    "i do not have enough information",
    "i don't have enough information",
    "i could not find anything",
    "je ne dispose pas",
    "je n'ai pas suffisamment",
    "je ne trouve rien",
)


def _mask(text: str) -> str:
    """``text`` with code and Markdown links blanked out, character positions kept.

    ``list[int]`` in a fence and ``[the docs](https://x)`` in a sentence are not
    citations, and treating either as one fabricates provenance out of syntax.
    Both patterns come from step 03 rather than a third private copy.
    """
    chars = list(text)
    for pattern in (CODE, LINK):
        for match in pattern.finditer(text):
            chars[match.start() : match.end()] = " " * (match.end() - match.start())
    return "".join(chars)


def _numbers(group: str) -> list[int]:
    """The entry numbers one marker refers to: ``"1"``, ``"1, 3"`` or ``"1-3"``."""
    found = [int(n) for n in NUMBER.findall(group)]
    if len(found) == 2 and "," not in group and ";" not in group:
        low, high = sorted(found)
        return list(range(low, high + 1))
    return found


def parse_citations(text: str) -> list[int]:
    """Entry numbers cited in ``text``, deduplicated, in first-appearance order.

    Order is part of the contract: the returned source list is built from it.
    ``[0]`` is returned rather than dropped so validation can reject it out loud.
    """
    seen: dict[int, None] = {}
    for match in CITATION.finditer(_mask(text)):
        for number in _numbers(match.group(1)):
            seen.setdefault(number)
    return list(seen)


def _close_gaps(text: str) -> str:
    """Absorb the whitespace a removed marker leaves behind."""
    text = re.sub(rf"[ \t]*{GAP}[ \t]*(?=[.,;:!?)\]]|$)", "", text, flags=re.MULTILINE)
    text = re.sub(rf"(?:\A|(?<=\n))[ \t]*{GAP}[ \t]*", "", text)
    return re.sub(rf"[ \t]*{GAP}[ \t]*", " ", text)


def validate_citations(
    answer: str,
    sources: Sequence[Source],
    *,
    strict: bool = False,
) -> tuple[str, list[Source], list[str]]:
    """Return ``(cleaned_answer, cited_sources, warnings)``.

    Citations that resolve are renumbered contiguously from 1 **in the text and
    in the source list together**; a returned source was cited, and a citation in
    the returned text resolves to one. Citations that do not resolve are stripped
    and warned about, never renumbered onto a real document — that would map a
    fabricated claim to a real source, which is the worst outcome available.

    ``strict=True`` raises instead of warning: an evaluation run needs an invented
    citation to fail loudly rather than skew a metric by one quiet warning.
    """
    by_index = {source.index: source for source in sources}
    cited = parse_citations(answer)
    invalid = [number for number in cited if number not in by_index]

    if invalid and strict:
        raise ValueError(
            f"answer cites {', '.join(f'[{n}]' for n in invalid)} "
            f"but the context has entries {sorted(by_index) or 'none'}"
        )

    order = [number for number in cited if number in by_index]
    renumbered = {old: new for new, old in enumerate(order, start=1)}

    out: list[str] = []
    position = 0
    for match in CITATION.finditer(_mask(answer)):
        kept = [renumbered[n] for n in _numbers(match.group(1)) if n in renumbered]
        out.append(answer[position : match.start()])
        out.append(f"[{', '.join(map(str, kept))}]" if kept else GAP)
        position = match.end()
    out.append(answer[position:])
    text = _close_gaps("".join(out))

    warnings: list[str] = []
    if invalid:
        warnings.append(
            f"invalid_citations: {', '.join(f'[{n}]' for n in invalid)} "
            f"not in the context ({len(sources)} entries)"
        )
    if not order and not _is_refusal(answer):
        warnings.append("answer_without_citations")

    return (
        text,
        [by_index[old].model_copy(update={"index": new}) for old, new in renumbered.items()],
        warnings,
    )


def _is_refusal(answer: str) -> bool:
    """True when the answer declines rather than asserts."""
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)
