"""The whole pipeline in one function: retrieve, assemble, ask, report.

    search -> build_context -> prompt -> LLM -> validate_citations -> Answer

Deliberately boring and deliberately short. Every arrow above is a function this
project owns, which is the point of the exercise; steps 14-20 replace what is
behind the arrows without changing this shape.
"""

import time
from collections.abc import Callable

from app.core.config import Settings, get_settings
from app.generation.citations import validate_citations
from app.generation.context import MAX_CONTEXT_CHARS, build_context
from app.generation.llm import SYSTEM_PROMPT, USER_TEMPLATE, complete
from app.models.answers import Answer, RetrievalStats
from app.models.chunks import ScoredChunk
from app.retrieval.search import search

Retriever = Callable[..., list[ScoredChunk]]
Completer = Callable[..., tuple[str, dict[str, int]]]

# Not a setting: it never varies by environment. Step 22 will reuse it when
# retrieval is merely weak rather than empty.
NO_CONTEXT_ANSWER = (
    "I could not find anything about this in the indexed documentation, so I cannot answer it."
)


def answer_question(
    question: str,
    *,
    top_k: int | None = None,
    source: str | None = None,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    strict: bool = False,
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    llm: Completer | None = None,
) -> Answer:
    """Answer ``question`` from the indexed corpus, or admit that it cannot.

    ``retriever`` and ``llm`` are injectable so the tests run with no server, no
    key and no spend — which is also how step 11 will swap in a different
    retriever without touching this function.

    ``strict`` turns an invented citation from a warning into a ``ValueError``,
    which is what an evaluation run wants and what a demo does not.
    """
    if not question.strip():
        raise ValueError("question is empty")

    settings = settings or get_settings()
    retriever = retriever or search
    llm = llm or complete
    started = time.perf_counter()

    chunks = retriever(question, top_k=top_k or settings.top_k, source=source, settings=settings)
    context, sources, dropped = build_context(chunks, max_chars=max_context_chars)
    # Counted before validation trims ``sources`` to the cited subset: how many
    # chunks reached the model is a retrieval fact, not a citation one.
    used = len(sources)
    warnings: list[str] = []

    if chunks:
        text, usage = llm(
            SYSTEM_PROMPT,
            USER_TEMPLATE.format(context=context, question=question),
            model=settings.generation_model,
        )
        if not text.strip():
            # A blank completion is a failure — a length stop, a refusal, a
            # filtered response. Returning it hides that behind a valid object.
            raise ValueError("the model returned an empty answer")
        text, sources, warnings = validate_citations(text, sources, strict=strict)
    else:
        # No call: with an empty context the only thing a model can produce is an
        # invention, and it would be charged for. Nothing to validate either —
        # a refusal with no context to cite is the right answer, not a warning.
        text, usage, sources = NO_CONTEXT_ANSWER, {}, []

    return Answer(
        answer=text,
        sources=sources,
        retrieval=RetrievalStats(retrieved=used + dropped, used=used, dropped=dropped),
        latency_ms=(time.perf_counter() - started) * 1000,
        model=settings.generation_model,
        usage=usage,
        warnings=warnings,
    )
