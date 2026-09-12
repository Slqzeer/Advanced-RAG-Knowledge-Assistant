"""Citations, parsed out of an answer and reconciled against the context.

An unverified ``[1]`` is worse than no citation at all: it looks like provenance
and is not. These tests are the guarantee, so they are written against the model
output shapes that actually occur, not the one the prompt asks for.
"""

import pytest

from app.generation.citations import parse_citations, validate_citations
from app.models.answers import Source


def source(index: int) -> Source:
    return Source(
        index=index,
        document_id=f"fastapi:doc-{index}",
        title=f"Doc {index}",
        url=None,
        section=None,
        chunk_id=f"fastapi:doc-{index}:0",
        score=1.0 - 0.1 * index,
    )


SOURCES = [source(i) for i in range(1, 6)]


# --- parsing ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Uses Depends [1].", [1]),
        ("Both agree [1][3] on this.", [1, 3]),
        ("Both agree [1, 3] on this.", [1, 3]),
        ("A whole run [1-3] of entries.", [1, 2, 3]),
        ("First [1]. Again [1].", [1]),
        ("Out of order [3] then [1].", [3, 1]),
    ],
)
def test_the_citation_shapes_a_model_actually_emits(text: str, expected: list[int]) -> None:
    assert parse_citations(text) == expected


def test_markdown_link_text_is_not_a_citation() -> None:
    """Without this every link in an answer becomes a fake source."""
    assert parse_citations("see [the docs](https://x) and [1](https://y)") == []


def test_brackets_inside_a_code_fence_are_not_citations() -> None:
    text = "Index it:\n\n```python\nitems[0]\nvalues: list[int] = []\n```\n"
    assert parse_citations(text) == []


def test_brackets_inside_an_inline_code_span_are_not_citations() -> None:
    assert parse_citations("call `items[0]` on it") == []


@pytest.mark.parametrize("text", ["[abc]", "[]", "[ ]", "[1a]", "nothing here"])
def test_non_numeric_brackets_are_not_citations(text: str) -> None:
    assert parse_citations(text) == []


def test_zero_is_parsed_so_that_validation_can_reject_it() -> None:
    """Dropping it here would let an invalid citation through unwarned."""
    assert parse_citations("Grounded [0].") == [0]


def test_citations_come_back_in_first_appearance_order() -> None:
    """Source ordering is built from this list, so the order is the contract."""
    assert parse_citations("[4] then [2] then [4] then [1]") == [4, 2, 1]


# --- validation ------------------------------------------------------------


def test_valid_citations_leave_the_text_alone_and_return_the_cited_subset() -> None:
    text, sources, warnings = validate_citations("Uses Depends [1] and caching [2].", SOURCES)
    assert text == "Uses Depends [1] and caching [2]."
    assert [s.document_id for s in sources] == ["fastapi:doc-1", "fastapi:doc-2"]
    assert warnings == []


def test_an_out_of_range_citation_is_stripped_and_warned_about() -> None:
    """``[7]`` with five entries means the model invented a source. It is never
    renumbered onto a real one — that maps a fabricated claim to a real document."""
    text, sources, warnings = validate_citations("Uses Depends [1]. It caches [7].", SOURCES)
    assert text == "Uses Depends [1]. It caches."
    assert [s.document_id for s in sources] == ["fastapi:doc-1"]
    assert len(warnings) == 1
    assert "7" in warnings[0]


def test_zero_is_out_of_range_too() -> None:
    text, sources, warnings = validate_citations("Grounded [0].", SOURCES)
    assert text == "Grounded."
    assert sources == []
    assert "0" in warnings[0]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Uses Depends [7].", "Uses Depends."),
        ("Uses [7] Depends.", "Uses Depends."),
        ("[7] Depends is the hook.", "Depends is the hook."),
        ("Uses Depends [7]", "Uses Depends"),
        ("Uses Depends [1][7].", "Uses Depends [1]."),
        ("One [1].\n\n[7] Two.", "One [1].\n\nTwo."),
    ],
)
def test_removing_a_marker_leaves_no_whitespace_artefacts(answer: str, expected: str) -> None:
    text, _, _ = validate_citations(answer, SOURCES)
    assert text == expected


def test_an_uncited_answer_is_flagged() -> None:
    """Zero citations plus a long confident answer is ungrounded generation
    wearing a RAG costume."""
    long_answer = "FastAPI resolves dependencies per request and caches the result. " * 3
    _, sources, warnings = validate_citations(long_answer, SOURCES)
    assert sources == []
    assert warnings == ["answer_without_citations"]


def test_an_uncited_refusal_is_not_flagged() -> None:
    """A genuine refusal has nothing to cite and is the correct answer."""
    refusal = "I do not have enough information in the provided context to answer this."
    text, sources, warnings = validate_citations(refusal, SOURCES)
    assert (text, sources, warnings) == (refusal, [], [])


def test_sources_are_renumbered_and_the_text_is_rewritten_to_match() -> None:
    """The subtle bug that makes a demo look fine and an audit fail: renumbering
    the source list without renumbering the answer that points at it."""
    text, sources, warnings = validate_citations("First [3]. Second [1].", SOURCES)
    assert text == "First [1]. Second [2]."
    assert [s.index for s in sources] == [1, 2]
    assert [s.document_id for s in sources] == ["fastapi:doc-3", "fastapi:doc-1"]
    assert warnings == []


def test_a_grouped_citation_is_renumbered_as_a_group() -> None:
    text, sources, _ = validate_citations("Both [2, 4] agree.", SOURCES)
    assert text == "Both [1, 2] agree."
    assert [s.document_id for s in sources] == ["fastapi:doc-2", "fastapi:doc-4"]


def test_every_citation_in_the_returned_text_resolves_to_a_returned_source() -> None:
    """The whole point of the step, asserted directly."""
    text, sources, _ = validate_citations("A [5][2]. B [7]. C [2-3].", SOURCES)
    assert set(parse_citations(text)) == {s.index for s in sources}


def test_strict_mode_raises_instead_of_warning() -> None:
    """Evaluation runs need the failure loud: a silent warning skews the metric."""
    with pytest.raises(ValueError, match=r"\[7\]"):
        validate_citations("Uses Depends [7].", SOURCES, strict=True)


def test_strict_mode_is_quiet_when_everything_resolves() -> None:
    text, sources, warnings = validate_citations("Uses Depends [1].", SOURCES, strict=True)
    assert (text, len(sources), warnings) == ("Uses Depends [1].", 1, [])
