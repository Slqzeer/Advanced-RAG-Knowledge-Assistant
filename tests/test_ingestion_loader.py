import pytest
from pydantic import ValidationError

from app.models.documents import RawDocument


def _document(**overrides: str) -> RawDocument:
    fields = {
        "document_id": "fastapi:docs/index",
        "source": "fastapi",
        "title": "FastAPI",
        "path": "docs/index.md",
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
