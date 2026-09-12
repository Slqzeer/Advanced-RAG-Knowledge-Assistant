from pathlib import Path

import pytest

from app.evaluation.dataset import EvalQuestion, load_dataset, validate_dataset

VALID = Path(__file__).parent / "data" / "eval" / "valid.jsonl"


def _write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _row(**overrides: object) -> str:
    fields: dict[str, object] = {
        "question_id": "q001",
        "question": "How does dependency injection work?",
        "category": "conceptual",
        "relevant_document_ids": ["fastapi:tutorial/dependencies/index"],
    }
    fields.update(overrides)
    import json

    return json.dumps(fields)


def test_a_valid_file_loads_every_row() -> None:
    questions = load_dataset(VALID)
    assert [q.question_id for q in questions] == ["q001", "q002", "q003"]
    assert all(isinstance(q, EvalQuestion) for q in questions)


def test_blank_lines_and_comments_are_skipped() -> None:
    assert len(load_dataset(VALID)) == 3


def test_defaults_are_filled_in() -> None:
    first = load_dataset(VALID)[0]
    assert first.relevant_sections == []
    assert first.notes is None
    assert first.held_out is False


def test_malformed_json_names_the_line_number(tmp_path: Path) -> None:
    path = _write(tmp_path, _row(), "{not json", _row(question_id="q003"))
    with pytest.raises(ValueError, match=r":2:"):
        load_dataset(path)


def test_duplicate_question_id_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, _row(), _row())
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)


def test_unknown_category_lists_the_allowed_values(tmp_path: Path) -> None:
    path = _write(tmp_path, _row(category="factoid"))
    with pytest.raises(ValueError, match="unanswerable"):
        load_dataset(path)


def test_empty_question_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, _row(question="  "))
    with pytest.raises(ValueError):
        load_dataset(path)


def test_an_answerable_question_needs_at_least_one_document(tmp_path: Path) -> None:
    path = _write(tmp_path, _row(relevant_document_ids=[]))
    with pytest.raises(ValueError, match="relevant_document_ids"):
        load_dataset(path)


def test_an_unanswerable_question_must_have_no_documents(tmp_path: Path) -> None:
    path = _write(tmp_path, _row(category="unanswerable"))
    with pytest.raises(ValueError, match="unanswerable"):
        load_dataset(path)


def test_categories_filter(tmp_path: Path) -> None:
    assert [q.question_id for q in load_dataset(VALID, categories=["exact"])] == ["q002"]
    assert load_dataset(VALID, categories=["code"]) == []


def test_validate_dataset_reports_unknown_document_ids() -> None:
    questions = load_dataset(VALID)
    problems = validate_dataset(questions, {"fastapi:tutorial/dependencies/index"})
    assert len(problems) == 1
    assert "q002" in problems[0]
    assert "fastapi:tutorial/handling-errors" in problems[0]


def test_validate_dataset_is_silent_when_every_id_exists() -> None:
    questions = load_dataset(VALID)
    known = {"fastapi:tutorial/dependencies/index", "fastapi:tutorial/handling-errors"}
    assert validate_dataset(questions, known) == []
