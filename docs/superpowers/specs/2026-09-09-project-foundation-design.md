# Project Foundation Design

## Purpose

Establish a reproducible development foundation for Advanced RAG Knowledge
Assistant without implementing the RAG pipeline, API endpoints, ingestion, or
retrieval behavior. The repository must be understandable to a new contributor
and ready for the first functional phase.

## Scope

This foundation includes:

- a Python 3.12 project managed and locked with `uv`;
- the target application package boundaries described in `information.md`;
- local code-quality and test tooling;
- a single-node Qdrant service for local development;
- environment-variable documentation;
- a polished, navigable, evolutionary README;
- a smoke test that validates the Python environment only.

It explicitly excludes FastAPI routes, RAG behavior, document ingestion,
embeddings, collections, LLM calls, CI workflows, Redis, an application Docker
image, and deployment configuration.

## Repository Structure

```text
.
|-- app/
|   |-- api/
|   |-- core/
|   |-- evaluation/
|   |-- generation/
|   |-- ingestion/
|   |-- models/
|   `-- retrieval/
|-- data/
|   |-- processed/
|   `-- raw/
|-- docker/
|-- docs/
|-- notebooks/
|-- scripts/
|-- tests/
|-- .editorconfig
|-- .env.example
|-- .gitignore
|-- .pre-commit-config.yaml
|-- .python-version
|-- compose.yaml
|-- information.md
|-- pyproject.toml
|-- README.md
`-- uv.lock
```

Python package directories contain `__init__.py`. Non-package directories that
would otherwise be empty contain `.gitkeep`. The repository stays flat and
follows the layout requested in `information.md`; introducing a `src/` layout
would add churn without solving a current problem.

## Python Environment

`.python-version` pins the development line to Python 3.12. `pyproject.toml`
requires Python 3.12 or newer and configures `uv` as a non-packaged application
project. `uv.lock` is committed so every contributor resolves the same package
versions.

Runtime dependencies are limited to the foundation needed by the next phase:

- `fastapi`;
- `uvicorn` with its standard extras;
- `pydantic-settings`;
- `qdrant-client`.

The `dev` dependency group contains:

- `mypy`;
- `pre-commit`;
- `pytest`;
- `ruff`.

LangChain, embedding providers, rerankers, RAGAS, OpenTelemetry, and Redis
clients are added only in the phase that uses them.

## Quality Tooling

Ruff is the single formatter and linter. It targets Python 3.12 and checks the
application, scripts, and tests. Mypy checks `app` with strict settings while
allowing dependency packages without type information. Pytest discovers tests
under `tests/`.

Pre-commit runs lightweight repository checks, Ruff formatting, Ruff linting,
and an `uv lock --check` guard. Hooks use pinned revisions. Installation remains
explicit (`uv run pre-commit install`) so `uv sync` does not mutate a
contributor's Git configuration.

## Local Infrastructure

`compose.yaml` defines only Qdrant, using the official
`qdrant/qdrant:v1.19.1` image. It publishes REST port `6333` and gRPC port
`6334`, persists data in a named Docker volume, and includes a healthcheck.
The instance is intentionally unauthenticated and bound for local development;
the README warns that this configuration is not production-ready.

No application container or custom Dockerfile is created yet because there is
no runnable application. The `docker/` directory is retained as the documented
home for later container assets.

## Configuration

`.env.example` documents non-secret local defaults, including the Qdrant URL,
REST port, gRPC port, and empty provider-key placeholders required by future
phases. `.env` and local data are ignored by Git. No real credential appears in
the repository.

## README Design

The README is written in French and acts as the evolving front page of the
project. It contains:

1. a concise title and value proposition;
2. a clickable table of contents;
3. project goals and learning principles;
4. a clear "current state" section;
5. a Mermaid diagram of the target RAG pipeline;
6. the technical stack;
7. prerequisites and quick-start commands;
8. development and quality commands;
9. the repository structure;
10. a detailed roadmap with completed, current, and planned phases;
11. an empty results framework for Recall@K, MRR, latency, and cost;
12. a quality and CI section that clearly states CI is planned;
13. links to project documentation.

The README never presents planned behavior as implemented. Result cells use an
em dash until measurements exist. Badges are added only when their underlying
workflow or published artifact exists, so the initial README has no misleading
CI badge.

## Verification

The setup is accepted when all of these checks pass:

```text
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run mypy app
uv run pytest
docker compose config --quiet
docker compose up -d qdrant
docker compose ps
```

The smoke test verifies that the top-level application package and the four
runtime dependencies can be imported. It contains no business logic and makes
no network call. Qdrant readiness is verified through the Compose healthcheck,
not through the default Python test suite, keeping tests deterministic when
Docker is unavailable.

## Evolution Rules

Each future project phase updates the README in the same change that implements
the phase: mark the roadmap item complete, update the current-state section,
document new commands, and add measured results when available. Tooling and
dependencies are introduced only when an implemented phase needs them.
