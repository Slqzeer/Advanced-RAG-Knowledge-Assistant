from pathlib import Path

import pytest
from pydantic import ValidationError

from app.ingestion.loader import doc_type_of, load_document, load_documents
from app.models.documents import RawDocument


def _document(**overrides: str) -> RawDocument:
    fields = {
        "document_id": "fastapi:docs/index",
        "source": "fastapi",
        "title": "FastAPI",
        "path": "docs/index.md",
        "doc_type": "docs",
        "url": None,
        "text": "hello",
        "content_hash": "0" * 64,
    }
    fields.update(overrides)
    return RawDocument(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["document_id", "text"])
def test_raw_document_rejects_empty(field: str) -> None:
    with pytest.raises(ValidationError):
        _document(**{field: ""})


CORPUS_ROOT = Path(__file__).parent / "data" / "corpus" / "fastapi"


def _load() -> list[RawDocument]:
    return load_documents(CORPUS_ROOT, "fastapi", "https://fastapi.tiangolo.com")


def test_only_markdown_files_are_loaded() -> None:
    assert len(_load()) == 3


def test_document_id_and_url_use_forward_slashes() -> None:
    document = next(d for d in _load() if d.title == "Dependencies")
    assert document.document_id == "fastapi:docs/tutorial/dependencies"
    assert document.path == "docs/tutorial/dependencies.md"
    assert document.url == "https://fastapi.tiangolo.com/docs/tutorial/dependencies/"


def test_index_url_drops_the_index_segment() -> None:
    document = next(d for d in _load() if d.document_id == "fastapi:docs/index")
    assert document.url == "https://fastapi.tiangolo.com/docs/"


def test_front_matter_does_not_leak_into_title() -> None:
    titles = {d.title for d in _load()}
    assert titles == {"First Steps", "Dependencies", "FastAPI"}


def test_title_falls_back_to_filename_stem(tmp_path: Path) -> None:
    path = tmp_path / "query-params.md"
    path.write_text("no heading here\n", encoding="utf-8")
    assert load_document(path, tmp_path, "fastapi").title == "Query Params"


def test_content_hash_is_stable_and_distinguishes_documents() -> None:
    first, second = _load(), _load()
    hashes = [d.content_hash for d in first]
    assert hashes == [d.content_hash for d in second]
    assert len(set(hashes)) == 3


def test_documents_are_sorted_by_document_id() -> None:
    ids = [d.document_id for d in _load()]
    assert ids == sorted(ids)
    assert ids == [d.document_id for d in _load()]


def test_doc_type_is_the_first_path_segment() -> None:
    assert doc_type_of("tutorial/first-steps.md") == "tutorial"


def test_a_nested_path_still_reports_its_top_segment() -> None:
    assert doc_type_of("tutorial/dependencies/classes-as-dependencies.md") == "tutorial"


def test_a_file_at_the_corpus_root_is_root() -> None:
    assert doc_type_of("index.md") == "root"


def test_loading_a_document_sets_its_doc_type(tmp_path: Path) -> None:
    (tmp_path / "deployment").mkdir()
    (tmp_path / "deployment" / "docker.md").write_text("# Docker\n\nbody\n", encoding="utf-8")
    [document] = load_documents(tmp_path, "fastapi")
    assert document.doc_type == "deployment"
