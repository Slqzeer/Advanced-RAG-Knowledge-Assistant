from importlib.util import find_spec

RUNTIME_MODULES = (
    "app",
    "fastapi",
    "pydantic_settings",
    "qdrant_client",
    "uvicorn",
)


def test_runtime_modules_are_importable() -> None:
    missing = [name for name in RUNTIME_MODULES if find_spec(name) is None]

    assert missing == [], f"Modules introuvables : {', '.join(missing)}"
