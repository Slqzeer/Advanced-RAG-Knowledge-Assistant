import logging
import random
from textwrap import dedent

import pytest

from app.ingestion.chunk import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    chunk_document,
    chunk_documents,
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
        "url": "https://fastapi.tiangolo.com/tutorial/",
        "text": text,
        "content_hash": "deadbeef",
    }
    fields.update(overrides)
    return RawDocument(**fields)  # type: ignore[arg-type]


def sentences(count: int, word: str = "prose") -> str:
    return " ".join(f"{word} sentence number {n} carries a few words" for n in range(count)) + "."


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
    chunks = chunk_document(document(sentences(200), language="fr"))
    assert len(chunks) > 1
    assert all(c.source == "fastapi" and c.title == "Tutorial" for c in chunks)
    assert all(c.url == "https://fastapi.tiangolo.com/tutorial/" for c in chunks)
    assert all(c.language == "fr" for c in chunks)


def test_chunk_documents_concatenates() -> None:
    a, b = document(sentences(200)), document(sentences(150), document_id="fastapi:other")
    assert len(chunk_documents([a, b])) == len(chunk_document(a)) + len(chunk_document(b))
    assert chunk_documents([]) == []
