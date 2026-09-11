from pathlib import Path

import pytest

from app.core.config import Settings, get_settings

# The repo .env sets most of these; a defaults test has to work from nothing.
OVERRIDES = (
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "OPENAI_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_CACHE_PATH",
    "CHUNK_SIZE",
    "TOP_K",
    "CORPUS_DIR",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in OVERRIDES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_apply_to_a_clean_environment(clean_env: None) -> None:
    settings = Settings(_env_file=None)
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "chunks"
    assert settings.chunk_size == 1000
    assert settings.top_k == 5
    assert settings.corpus_dir == Path("data/raw")


def test_the_api_key_is_optional(clean_env: None) -> None:
    """Importing settings must never require a key: the test suite has none."""
    assert Settings(_env_file=None).openai_api_key is None


def test_an_environment_variable_overrides_a_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION", "somewhere_else")
    assert Settings(_env_file=None).qdrant_collection == "somewhere_else"


def test_keys_for_later_phases_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "not-a-field-yet")
    assert Settings(_env_file=None).qdrant_collection == "chunks"


def test_get_settings_returns_the_same_object_twice() -> None:
    assert get_settings() is get_settings()
