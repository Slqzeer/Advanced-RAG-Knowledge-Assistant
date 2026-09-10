from importlib import import_module

RUNTIME_MODULES = (
    "app",
    "fastapi",
    "pydantic_settings",
    "qdrant_client",
    "uvicorn",
)


def test_runtime_modules_are_importable() -> None:
    for module_name in RUNTIME_MODULES:
        module = import_module(module_name)
        assert module.__name__ == module_name
