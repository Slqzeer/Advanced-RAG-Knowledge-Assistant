"""The whole pipeline in one function: retrieve, assemble, ask, report.

    search -> build_context -> prompt -> LLM -> validate_citations -> Answer

Deliberately boring and deliberately short. Every arrow above is a function this
project owns, which is the point of the exercise; steps 14-20 replace what is
behind the arrows without changing this shape.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from app.core.config import Settings, get_settings
from app.generation.citations import is_refusal, validate_citations
from app.generation.compress import BatchEmbedder
from app.generation.compress import compress as compress_chunks
from app.generation.context import MAX_CONTEXT_CHARS, build_context
from app.generation.llm import SYSTEM_PROMPTS, USER_TEMPLATE, Completer, complete
from app.models.answers import Answer, RetrievalStats
from app.models.chunks import ScoredChunk
from app.retrieval.search import Filters, search
from app.retrieval.transform import contextualize

Retriever = Callable[..., list[ScoredChunk]]

# Not a setting: it never varies by environment. Step 22 designed a second fixed
# refusal for weak retrieval and rejected it offline — no score floor separates
# this corpus's answerable questions from its near-miss unanswerable ones.
NO_CONTEXT_ANSWER = (
    "I could not find anything about this in the indexed documentation, so I cannot answer it."
)


def answer_question(
    question: str,
    *,
    history: Sequence[Mapping[str, str]] | None = None,
    transform: str | None = None,
    transform_n: int | None = None,
    top_k: int | None = None,
    mode: str | None = None,
    rerank: str | None = None,
    rerank_candidates: int | None = None,
    compress: str | None = None,
    compress_candidates: int | None = None,
    compress_budget: int | None = None,
    embedder: BatchEmbedder | None = None,
    filters: Filters | None = None,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    include_contexts: bool = False,
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

    ``mode`` selects the retriever and ``rerank`` the cross-encoder that reorders
    its output; both are threaded straight through so the measured winners of
    steps 16 and 17 reach the answer, not only the benchmark.

    ``history`` is the conversation so far, as ``{"role", "content"}`` mappings.
    Given one, the question is resolved into a standalone query before retrieval
    — ``"et pour docker ?"`` becomes a question that means something on its own.
    This is the one place that happens: ``search()`` takes a query string and is
    never told a conversation exists.

    ``transform`` and ``transform_n`` are threaded straight through to the
    retriever, exactly as ``mode`` and ``rerank`` already are, so the measured
    winner of step 19 reaches the answer and not only the benchmark.

    ``compress`` names a sentence extractor from ``compress.COMPRESSORS``. It
    runs *after* retrieval and *before* ``build_context``: a retriever that
    rewrites chunk text has stopped being one, and ``build_context`` commits in
    its own docstring to no I/O and no model. ``None`` reads ``COMPRESS_METHOD``
    and the empty string forces it off.

    ``compress_candidates`` is the pool depth retrieved when compression is on —
    the whole point of the stage, since step 17 measured the pool holding
    documents no ranking could surface. With compression off, ``top_k`` is
    retrieved unchanged and nothing about this function's behaviour differs from
    step 19's.

    ``include_contexts`` returns the chunk texts that reached the model on the
    ``Answer``. Off by default: step 21's judge needs them, step 25's endpoint
    does not, and rebuilding them outside this function by re-running search and
    compression could diverge from what was actually sent.
    """
    if not question.strip():
        raise ValueError("question is empty")

    settings = settings or get_settings()
    if settings.prompt_version not in SYSTEM_PROMPTS:
        # Before retrieval: an unknown version must cost nothing to discover.
        raise ValueError(
            f"unknown prompt version {settings.prompt_version!r}; have {sorted(SYSTEM_PROMPTS)}"
        )
    retriever = retriever or search
    llm = llm or complete
    started = time.perf_counter()

    # Before retrieval and above search(): resolving a follow-up needs the
    # conversation, and pushing that into the retrieval seam would carry it on
    # into step 23's cache key and step 25's endpoint. With no history this
    # returns the question and makes no call.
    query = contextualize(question, history or [], settings=settings, llm=llm)

    method = settings.compress_method if compress is None else compress
    wanted = top_k or settings.top_k
    depth = wanted
    if method:
        depth = compress_candidates or settings.compress_candidates
        if depth < wanted:
            # A pool shallower than the answer is a compressor with nothing to
            # choose between; the same guard search() applies to rerank_candidates.
            raise ValueError(f"compress_candidates ({depth}) must be at least top_k ({wanted})")

    chunks = retriever(
        query,
        top_k=depth,
        mode=mode,
        transform=transform,
        transform_n=transform_n,
        rerank=rerank,
        rerank_candidates=rerank_candidates,
        filters=filters,
        settings=settings,
    )
    # Counted before compression: how many chunks the retriever produced is a
    # retrieval fact, and a d20 run that compresses to seven must report 20.
    pool = len(chunks)
    chunks = compress_chunks(
        chunks,
        query,
        method=method,
        budget_chars=compress_budget,
        settings=settings,
        embedder=embedder,
    )
    context, sources, _ = build_context(
        chunks, max_chars=max_context_chars, tagged=settings.prompt_version != "v2"
    )
    # Counted before validation trims ``sources`` to the cited subset: how many
    # chunks reached the model is a retrieval fact, not a citation one.
    used = len(sources)
    # Sliced to ``used``, not to ``chunks``: build_context keeps chunks whole or
    # not at all and stops at max_chars, so a longer list would claim the model
    # saw something it never received.
    contexts = [scored.chunk.text for scored in chunks[:used]] if include_contexts else []
    warnings: list[str] = []

    refusal: Literal["no_context", "model_declined"] | None
    if chunks:
        text, usage = llm(
            SYSTEM_PROMPTS[settings.prompt_version],
            USER_TEMPLATE.format(context=context, question=question),
            model=settings.generation_model,
        )
        if not text.strip():
            # A blank completion is a failure — a length stop, a refusal, a
            # filtered response. Returning it hides that behind a valid object.
            raise ValueError("the model returned an empty answer")
        text, sources, warnings = validate_citations(text, sources, strict=strict)
        refusal = "model_declined" if is_refusal(text) else None
    else:
        # No call: with an empty context the only thing a model can produce is an
        # invention, and it would be charged for. Nothing to validate either —
        # a refusal with no context to cite is the right answer, not a warning.
        text, usage, sources = NO_CONTEXT_ANSWER, {}, []
        refusal = "no_context"

    return Answer(
        answer=text,
        sources=sources,
        retrieval=RetrievalStats(retrieved=pool, used=used, dropped=pool - used),
        context_chars=len(context),
        contexts=contexts,
        latency_ms=(time.perf_counter() - started) * 1000,
        model=settings.generation_model,
        usage=usage,
        warnings=warnings,
        refusal=refusal,
    )
