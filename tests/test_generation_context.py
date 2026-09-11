"""build_context is pure, so this is thorough: it is where silent corruption
happens, and a context block is only ever read after the answer is already wrong.
"""

from app.generation.context import build_context
from app.models.chunks import Chunk, ScoredChunk


def scored(rank: int, text: str = "body", **overrides: object) -> ScoredChunk:
    fields: dict[str, object] = {
        "document_id": f"fastapi:doc-{rank}",
        "source": "fastapi",
        "title": "Dependencies",
        "url": "https://fastapi.tiangolo.com/tutorial/dependencies/",
        "section": "First steps",
        "chunk_index": 0,
        "text": text,
        "char_start": 0,
        "char_end": max(len(text), 1),
    }
    fields.update(overrides)
    return ScoredChunk(chunk=Chunk(**fields), score=1.0 - 0.1 * rank, rank=rank)  # type: ignore[arg-type]


def test_entries_are_numbered_from_one_in_rank_order() -> None:
    block, _, _ = build_context([scored(i, f"body {i}") for i in (1, 2, 3)])
    assert block.index("[1]") < block.index("[2]") < block.index("[3]")
    assert "[0]" not in block


def test_a_header_carries_the_title_and_the_section() -> None:
    block, _, _ = build_context([scored(1)])
    header = block.splitlines()[0]
    assert "Dependencies" in header
    assert "First steps" in header


def test_a_chunk_without_a_section_still_gets_a_clean_header() -> None:
    block, _, _ = build_context([scored(1, section=None)])
    header = block.splitlines()[0]
    assert header == "[1] fastapi — Dependencies"


def test_sources_mirror_the_numbers_in_the_block() -> None:
    chunks = [scored(i, f"body {i}") for i in (1, 2, 3)]
    block, sources, _ = build_context(chunks)
    assert [source.index for source in sources] == [1, 2, 3]
    for source, chunk in zip(sources, chunks, strict=True):
        assert source.chunk_id == chunk.chunk.chunk_id
        assert source.score == chunk.score
        assert f"[{source.index}] " in block


def test_a_tight_budget_drops_the_lowest_ranked_chunks_whole() -> None:
    chunks = [scored(i, "x" * 100) for i in (1, 2, 3)]
    block, sources, dropped = build_context(chunks, max_chars=300)
    assert (len(sources), dropped) == (2, 1)
    # Whole chunks only: a half chunk is a half fact the model completes for you.
    assert block.count("x" * 100) == 2
    assert "fastapi:doc-3" not in [source.document_id for source in sources]


def test_a_budget_smaller_than_the_first_chunk_still_keeps_it() -> None:
    """An empty context guarantees a wrong answer; an over-budget one only
    risks a truncated prompt, so the call happens anyway."""
    block, sources, dropped = build_context([scored(1, "x" * 5_000)], max_chars=100)
    assert len(sources) == 1
    assert dropped == 0
    assert "x" * 5_000 in block


def test_no_chunks_means_no_block_and_no_sources() -> None:
    assert build_context([]) == ("", [], 0)


def test_chunk_text_is_copied_byte_for_byte() -> None:
    """Brackets and code fences in the corpus must not be escaped or reflowed:
    the model is being shown documentation, not a rendering of it."""
    text = "See [1] and also:\n```python\nx = {'a': 1}\n```\n  indented\ttab"
    block, _, _ = build_context([scored(1, text)])
    assert text in block
