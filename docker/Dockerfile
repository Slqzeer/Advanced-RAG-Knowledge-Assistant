# syntax=docker/dockerfile:1

FROM python:3.12.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.3 /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid rag --create-home rag \
    && apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN chown rag:rag /app

COPY --chown=rag:rag pyproject.toml uv.lock ./

USER rag

RUN --mount=type=cache,target=/home/rag/.cache/uv,uid=10001,gid=10001 \
    uv sync --locked --no-dev --no-install-project

COPY --chown=rag:rag app ./app
COPY --chown=rag:rag scripts ./scripts

RUN mkdir -p data/raw data/processed

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/openapi.json', timeout=2)"]

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
