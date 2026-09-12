import hashlib
import logging
import random
from itertools import pairwise
from textwrap import dedent

import pytest

from app.ingestion.chunk import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    STRATEGIES,
    chunk_document,
    chunk_documents,
    split_fixed,
    split_semantic,
    split_sentences,
    split_text,
)
from app.models.chunks import Chunk
from app.models.documents import RawDocument


def document(text: str, **overrides: object) -> RawDocument:
    fields: dict[str, object] = {
        "document_id": "fastapi:tutorial/index",
        "source": "fastapi",
        "title": "Tutorial",
        "path": "tutorial/index.md",
        "doc_type": "tutorial",
        "url": "https://fastapi.tiangolo.com/tutorial/",
        "text": text,
        "content_hash": "deadbeef",
    }
    fields.update(overrides)
    return RawDocument(**fields)  # type: ignore[arg-type]


def sentences(count: int, word: str = "prose") -> str:
    return " ".join(f"{word} sentence number {n} carries a few words" for n in range(count)) + "."


def prose(count: int, opener: str = "Sentence") -> str:
    """`count` real sentences, unlike `sentences()` which builds one long one."""
    return " ".join(f"{opener} number {n} carries a few words here." for n in range(count))


def uncovered(text: str, spans: list[tuple[int, int]]) -> str:
    """The text rebuilt from the chunks with every overlap dropped."""
    out: list[str] = []
    end = 0
    for start, stop in spans:
        out.append(text[max(start, end) : stop])
        end = stop
    return "".join(out)


# --- the chunk model -------------------------------------------------------


def chunk(**overrides: object) -> Chunk:
    fields: dict[str, object] = {
        "document_id": "fastapi:index",
        "source": "fastapi",
        "title": "FastAPI",
        "url": None,
        "section": "Dependencies",
        "chunk_index": 3,
        "text": "body",
        "char_start": 10,
        "char_end": 20,
    }
    fields.update(overrides)
    return Chunk(**fields)  # type: ignore[arg-type]


def test_chunk_id_is_derived() -> None:
    assert chunk().chunk_id == "fastapi:index#3"
    assert "chunk_id" not in Chunk.model_fields
    assert chunk(chunk_id="whatever").chunk_id == "fastapi:index#3"


def test_chunk_rejects_an_empty_or_inverted_span() -> None:
    with pytest.raises(ValueError):
        chunk(char_end=10)
    with pytest.raises(ValueError):
        chunk(char_end=9)
    with pytest.raises(ValueError):
        chunk(text="")


def test_chunk_payload_is_flat_and_carries_the_id() -> None:
    payload = chunk().to_payload()
    assert payload["chunk_id"] == "fastapi:index#3"
    assert payload["section"] == "Dependencies"
    assert all(not isinstance(v, dict | list) for v in payload.values())


# --- the splitter ----------------------------------------------------------


def test_short_text_is_one_chunk() -> None:
    text = "Just a paragraph."
    assert split_text(text) == [(0, len(text))]


def test_no_chunk_exceeds_chunk_size() -> None:
    text = sentences(400)
    assert len(text) > 10_000
    assert all(end - start <= CHUNK_SIZE for start, end in split_text(text))


def test_chunks_cover_the_source_without_loss() -> None:
    text = "\n\n".join(f"## Section {n}\n\n{sentences(12)}" for n in range(20))
    assert len(text) > 10_000
    assert uncovered(text, split_text(text)) == text


def test_consecutive_chunks_overlap() -> None:
    text = sentences(400)
    spans = split_text(text)
    assert len(spans) > 3
    for (_, previous_end), (start, _) in zip(spans, spans[1:], strict=False):
        assert 0 < previous_end - start <= CHUNK_OVERLAP


def test_a_code_fence_stays_whole_in_one_chunk() -> None:
    fence = "```python\n" + "x = compute(status_code=422)\n" * 9 + "```"
    assert 250 < len(fence) < 350
    text = f"{sentences(40)}\n\n{fence}\n\n{sentences(40)}"
    chunks = [text[s:e] for s, e in split_text(text)]
    assert sum(fence in c for c in chunks) == 1
    assert all(c.count("```") % 2 == 0 for c in chunks)


def test_an_oversized_fence_becomes_one_oversized_chunk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fence = "```python\n" + "x = compute(status_code=422)  # a line\n" * 38 + "```"
    assert len(fence) > 1400
    text = f"{sentences(20)}\n\n{fence}\n\n{sentences(20)}"
    with caplog.at_level(logging.WARNING):
        chunks = [text[s:e] for s, e in split_text(text)]
    assert sum(fence in c for c in chunks) == 1
    assert any(len(c) > CHUNK_SIZE for c in chunks)
    assert "oversized chunk" in caplog.text
    assert uncovered(text, split_text(text)) == text


def test_splitting_prefers_heading_boundaries() -> None:
    text = "\n\n".join(f"## Section {n}\n\n{sentences(10)}" for n in range(6))
    chunks = [text[s:e] for s, e in split_text(text)]
    assert len(chunks) == 6
    assert all(c.lstrip().startswith("## Section") for c in chunks)


def test_degenerate_inputs_terminate() -> None:
    assert split_text("") == []
    assert split_text("   \n\n\t ") == []
    blob = "a" * 5000
    spans = split_text(blob)
    assert uncovered(blob, spans) == blob
    assert all(end - start <= CHUNK_SIZE for start, end in spans)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        split_text("whatever", chunk_size=100, overlap=100)


def test_random_documents_always_reconstruct() -> None:
    rng = random.Random(0)
    for _ in range(20):
        parts = []
        for n in range(rng.randint(1, 30)):
            roll = rng.random()
            if roll < 0.2:
                parts.append(f"## Heading {n}")
            elif roll < 0.4:
                parts.append("```py\n" + "code = 1\n" * rng.randint(1, 60) + "```")
            else:
                parts.append(sentences(rng.randint(1, 30)))
        text = "\n\n".join(parts)
        assert uncovered(text, split_text(text)) == text


# --- documents -------------------------------------------------------------


def test_chunk_indices_and_offsets_slice_the_source_back() -> None:
    text = "# Title\n\n" + "\n\n".join(f"## Section {n}\n\n{sentences(12)}" for n in range(8))
    chunks = chunk_document(document(text))
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(text[c.char_start : c.char_end] == c.text for c in chunks)
    assert all(c.chunk_id == f"fastapi:tutorial/index#{c.chunk_index}" for c in chunks)


def test_section_is_the_nearest_enclosing_heading() -> None:
    text = dedent(
        """\
        # Dependencies

        {intro}

        ## Sub-dependencies

        {body}
        """
    ).format(intro=sentences(30), body=sentences(30))
    chunks = chunk_document(document(text))
    assert chunks[0].section is None
    assert chunks[-1].section == "Sub-dependencies"


def test_metadata_is_carried_onto_every_chunk() -> None:
    chunks = chunk_document(document(prose(200), language="fr"))
    assert len(chunks) > 1
    assert all(c.source == "fastapi" and c.title == "Tutorial" for c in chunks)
    assert all(c.url == "https://fastapi.tiangolo.com/tutorial/" for c in chunks)
    assert all(c.language == "fr" for c in chunks)


def test_chunk_documents_concatenates() -> None:
    a, b = document(sentences(200)), document(sentences(150), document_id="fastapi:other")
    assert len(chunk_documents([a, b])) == len(chunk_document(a)) + len(chunk_document(b))
    assert chunk_documents([]) == []


# --- the strategy registry -------------------------------------------------


def uniform(texts: list[str]) -> list[list[float]]:
    """Every sentence identical: no distance spike anywhere."""
    return [[1.0, 0.0] for _ in texts]


def call(strategy: str, text: str, **kwargs: object) -> list[tuple[int, int]]:
    """Every strategy behind one signature; `semantic` gets a scripted embedder."""
    if strategy == "semantic":
        kwargs.setdefault("embed_batch", uniform)
    return STRATEGIES[strategy](text, **kwargs)  # type: ignore[operator]


MIXED = "\n\n".join(
    [
        "## Overview",
        prose(12),
        "```py\nvalue = compute(status_code=422)\n" * 4 + "```",
        prose(20, "Explanation"),
        "## Details",
        prose(30, "Detail"),
    ]
)


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_every_strategy_tiles_the_text(strategy: str) -> None:
    spans = call(strategy, MIXED)
    assert len(spans) > 1
    assert spans[0][0] == 0
    assert spans[-1][1] == len(MIXED)
    for (_, previous_end), (start, _) in pairwise(spans):
        assert start <= previous_end


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_every_strategy_reconstructs_the_input(strategy: str) -> None:
    assert uncovered(MIXED, call(strategy, MIXED)) == MIXED


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_every_strategy_returns_non_empty_in_range_spans(strategy: str) -> None:
    assert all(0 <= start < end <= len(MIXED) for start, end in call(strategy, MIXED))


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_every_strategy_ignores_blank_input(strategy: str) -> None:
    assert call(strategy, "") == []
    assert call(strategy, "   \n\n\t ") == []


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_every_strategy_rejects_an_overlap_it_cannot_honour(strategy: str) -> None:
    with pytest.raises(ValueError):
        call(strategy, MIXED, chunk_size=100, overlap=100)


# --- fixed: the control, deliberately naive --------------------------------


def test_fixed_cuts_on_the_stride_regardless_of_content() -> None:
    text = prose(200)
    assert len(text) > 5000
    spans = split_fixed(text, chunk_size=1000, overlap=200)
    assert [start for start, _ in spans] == list(range(0, len(text) - 200, 800))
    assert all(end - start <= 1000 for start, end in spans)


def test_fixed_splits_a_code_fence_open() -> None:
    """Asserted, not tolerated: `fixed` is the floor of the comparison."""
    fence = "```python\n" + "x = compute(status_code=422)\n" * 6 + "```"
    text = prose(4) + "\n\n" + fence + "\n\n" + prose(4)
    chunks = [text[s:e] for s, e in split_fixed(text, chunk_size=200, overlap=40)]
    assert any(chunk.count("```") % 2 == 1 for chunk in chunks)


# --- sentence --------------------------------------------------------------


def test_sentence_boundaries_skip_abbreviations_and_version_numbers() -> None:
    text = "Use FastAPI 0.115.2 for this. It works, e.g. with Depends. Done."
    chunks = [text[s:e] for s, e in split_sentences(text, chunk_size=40, overlap=0)]
    assert chunks == ["Use FastAPI 0.115.2 for this. ", "It works, e.g. with Depends. Done."]


def test_an_oversized_sentence_is_its_own_chunk() -> None:
    long = "word " * 100 + "end."
    text = "Short one. " + long + " Tail here."
    chunks = [text[s:e] for s, e in split_sentences(text, chunk_size=200, overlap=50)]
    assert any(long in chunk for chunk in chunks)
    assert any(len(chunk) > 200 for chunk in chunks)


@pytest.mark.parametrize("strategy", ["sentence", "semantic"])
def test_sentence_strategies_never_cut_inside_code(strategy: str) -> None:
    fence = "```python\n" + "x = 1. y = 2. z = 3.\n" * 8 + "```"
    text = prose(6) + "\n\n" + fence + "\n\n" + prose(6)
    chunks = [text[s:e] for s, e in call(strategy, text, chunk_size=250, overlap=50)]
    assert sum(fence in chunk for chunk in chunks) == 1
    assert all(chunk.count("```") % 2 == 0 for chunk in chunks)


def test_sentence_overlap_is_whole_sentences() -> None:
    text = prose(60)
    spans = split_sentences(text, chunk_size=300, overlap=100)
    assert len(spans) > 3
    for start, _ in spans[1:]:
        assert text[:start].rstrip()[-1] in ".!?"


# --- semantic (scripted vectors: no network, no key) -----------------------


def test_semantic_cuts_where_the_meaning_changes() -> None:
    text = "Alpha one here. Alpha two here. Beta three here."

    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if t.startswith("Alpha") else [0.0, 1.0] for t in texts]

    chunks = [text[s:e] for s, e in split_semantic(text, overlap=0, embed_batch=embed)]
    assert chunks == ["Alpha one here. Alpha two here. ", "Beta three here."]


def test_semantic_makes_no_cut_when_every_sentence_is_alike() -> None:
    text = prose(10)
    assert len(text) < 1000
    assert split_semantic(text, overlap=0, embed_batch=uniform) == [(0, len(text))]


def test_semantic_still_honours_the_size_cap_without_a_distance_spike() -> None:
    text = prose(50)
    spans = split_semantic(text, chunk_size=300, overlap=0, embed_batch=uniform)
    assert len(spans) > 1
    assert all(end - start <= 300 for start, end in spans)


def test_semantic_embeds_every_sentence_in_one_call() -> None:
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return uniform(texts)

    split_semantic(prose(20), embed_batch=embed)
    assert len(calls) == 1
    assert len(calls[0]) == 20


# --- the registry ----------------------------------------------------------


def test_the_registry_holds_exactly_the_four_strategies() -> None:
    assert sorted(STRATEGIES) == ["fixed", "recursive", "semantic", "sentence"]
    assert STRATEGIES["recursive"] is split_text


def test_an_unknown_strategy_names_the_valid_ones() -> None:
    with pytest.raises(ValueError, match="recursive"):
        chunk_documents([document(prose(10))], strategy="nope")
    with pytest.raises(ValueError, match="semantic"):
        chunk_documents([], strategy="nope")


def test_recursive_is_byte_identical_to_the_step_11_baseline() -> None:
    """The baseline the whole comparison is measured against must not drift."""
    rng = random.Random(12)
    parts = []
    for n in range(40):
        roll = rng.random()
        if roll < 0.2:
            parts.append(f"## Heading {n}")
        elif roll < 0.4:
            parts.append("```py\n" + "code = 1\n" * rng.randint(1, 40) + "```")
        else:
            parts.append(sentences(rng.randint(1, 30)))
    text = "\n\n".join(parts)
    spans = split_text(text)
    digest = hashlib.sha256(repr(spans).encode()).hexdigest()
    assert digest == "93885e1e65ffb90c421013202c16391fdad91e35324371dc5327c1b6b9f61b14"
    chunks = chunk_document(document(text), strategy="recursive")
    kept = [span for span in spans if text[slice(*span)].strip()]
    assert [(c.char_start, c.char_end) for c in chunks] == kept
