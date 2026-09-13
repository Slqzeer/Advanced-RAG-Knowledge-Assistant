"""Work on the question, not on the index.

Step 17 measured the alternative and closed it: a cross-encoder reorders a pool
and is capped by that pool's recall, and every pool on this corpus saturates by
depth 20. A transform changes the *query*, so it can retrieve a document no pool
ever held. That is the one lever left.

A transform is a ``Transform``: it takes a query and returns a list of queries.
``rewrite`` returns one, replacing the original; ``multi`` returns several with
the original kept first. Parsing, capping, de-duplication and the fallback to
the original happen once, here, for every entry — the only thing that genuinely
differs between them is the prompt.

``contextualize`` lives in this file because it is a query transform by nature,
and is deliberately **not** a ``TRANSFORMS`` key: it takes a conversation and
returns a single string, so it does not fit a registry whose contract is
``str -> list[str]``, and it must never be reachable from ``search()``. Retrieval
does not get to learn what a conversation is — that dependency would propagate
into step 23's cache key and step 25's endpoint.
"""

import re
from collections.abc import Callable, Mapping, Sequence

from app.core.config import Settings, get_settings
from app.generation.llm import Completer, complete

Transform = Callable[[str, int, Settings, Completer], list[str]]

# Mirrors search.RETRIEVERS, rerank.RERANKERS and chunk.STRATEGIES: the registry
# is how this project compares N variants and promotes a winner. One at a time,
# by decision — QUERY_TRANSFORM takes one value and there is no chaining here.
TRANSFORMS: dict[str, Transform] = {}

# The corpus is English and the evaluation set asks in French, so a rewriter that
# also translates is the realistic transform rather than a separate feature. It
# does mean the `rewrite-standalone` row confounds reformulation with
# translation; task 3 records that and names the run that would separate them.
REWRITE_SYSTEM = """\
You rewrite a user's question into a single search query for a documentation search engine.

Rules:
- Output exactly one line: the rewritten query. No preamble, no numbering, no quotes, no explanation.
- Keep every proper noun, symbol, error code and number from the question.
- Prefer the vocabulary technical documentation uses.
- Write the query in English even when the question is not."""  # noqa: E501 — one rule per line, wrapped is worse

MULTI_SYSTEM = """\
You generate alternative phrasings of a question for a documentation search engine.

Rules:
- Output exactly {n} lines, one query per line. No preamble, no numbering, no quotes, no explanation.
- Each line must be a complete, standalone search query.
- Vary the vocabulary: use the words technical documentation would use, not only the user's.
- Keep every proper noun, symbol, error code and number from the question.
- Write the queries in English even when the question is not."""  # noqa: E501 — one rule per line, wrapped is worse

# Deliberately does NOT translate, unlike REWRITE_SYSTEM. This transform is
# measured as a delta (conv-raw against conv-rewrite), so it must do exactly one
# thing: resolve the reference. Folding translation in would make the delta
# measure two changes at once, which is the mistake llm.py's version marker
# exists to prevent.
CONTEXTUALIZE_SYSTEM = """\
You rewrite a follow-up question into a standalone one, using the conversation before it.

Rules:
- Output exactly one line: the standalone question. No preamble, no quotes, no explanation.
- Resolve every pronoun and every ellipsis ("and for X?", "what about that?") against the conversation.
- Keep the follow-up's own language. Do not translate.
- If the follow-up already stands alone, return it unchanged."""  # noqa: E501 — one rule per line, wrapped is worse

CONTEXTUALIZE_TEMPLATE = """Conversation:
{history}

Follow-up: {question}"""

# "1. ", "2) ", "- ", "* ", "• " — every way a model numbers a list it was asked
# not to number.
ORDINAL = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


def parse_queries(text: str, *, limit: int) -> list[str]:
    """One query per line, cleaned, at most ``limit`` of them.

    Line-oriented rather than JSON on purpose: structured output would mean
    adding a ``response_format`` parameter to ``complete()``, the one function in
    the project that talks to a model, kept to one signature so that swapping
    providers is editing its body. The failure this would protect against is
    handled by ``expand``'s fallback, which is needed either way.
    """
    queries: list[str] = []
    for line in text.splitlines():
        cleaned = ORDINAL.sub("", line).strip().strip("\"'").strip()
        # A trailing colon is a preamble ("Here are three queries:"), not a
        # query. It is the single most common way a chatty model poisons a list.
        if cleaned and not cleaned.endswith(":"):
            queries.append(cleaned)
    return queries[:limit]


def _rewrite(query: str, n: int, settings: Settings, llm: Completer) -> list[str]:
    """One query, replacing the original. ``n`` is ignored: a rewrite is a rewrite."""
    text, _ = llm(REWRITE_SYSTEM, query, model=settings.generation_model)
    return parse_queries(text, limit=1)


TRANSFORMS["rewrite"] = _rewrite


def _multi(query: str, n: int, settings: Settings, llm: Completer) -> list[str]:
    """``n`` queries, the original first and always kept.

    Keeping the original makes the raw ranking a floor that fusion can only build
    on: N phrasings that all drift the same way is how an expansion loses ground
    the user's own words already held. It also makes n=1 exactly today's
    behaviour, which is a free sanity check.
    """
    if n < 2:
        return [query]
    text, _ = llm(MULTI_SYSTEM.format(n=n - 1), query, model=settings.generation_model)
    # dict.fromkeys de-duplicates while preserving order; a model asked for three
    # phrasings returns the same one twice more often than you would hope.
    paraphrases = dict.fromkeys(parse_queries(text, limit=n - 1))
    paraphrases.pop(query, None)
    return [query, *paraphrases]


TRANSFORMS["multi"] = _multi


def expand(
    query: str,
    *,
    transform: str,
    n: int | None = None,
    settings: Settings | None = None,
    llm: Completer | None = None,
) -> list[str]:
    """The queries ``transform`` wants to retrieve with. Never empty.

    ``llm`` is injectable so every unit test runs with no network and no key —
    the pattern ``answer_question`` already uses.

    **A transform that produces nothing usable returns ``[query]``.** An API
    hiccup or a chatty preamble in the middle of a 38-question benchmark must
    degrade to exactly today's behaviour, never zero a question. The fallback is
    not silent: ``scripts/benchmark.py`` records the raw completion per question,
    so a row where it fired says so on inspection.
    """
    if transform not in TRANSFORMS:
        raise ValueError(f"unknown transform {transform!r}; have {sorted(TRANSFORMS)}")
    if not query.strip():
        # The empty string expands into plausible-looking garbage and then
        # retrieves against it, which search() already refuses one layer down.
        raise ValueError("query is empty")
    settings = settings or get_settings()
    # `n or settings.multi_query_n` would read 0 as "unset" and serve the default
    # instead of rejecting it — the same sentinel trap `rerank=""` already taught
    # this project to write out longhand.
    n = settings.multi_query_n if n is None else n
    if n < 1:
        raise ValueError(f"transform_n must be at least 1, got {n}")
    return TRANSFORMS[transform](query, n, settings, llm or complete) or [query]


def contextualize(
    question: str,
    history: Sequence[Mapping[str, str]],
    *,
    settings: Settings | None = None,
    llm: Completer | None = None,
) -> str:
    """``"et pour docker ?"`` plus what came before it, as one standalone question.

    ``history`` is a sequence of ``{"role", "content"}`` mappings — OpenAI's own
    message shape, because it is what ``complete()`` already speaks and what step
    25's endpoint will receive off the wire.

    An empty history returns the question unchanged **without an LLM call**.
    Every single-turn question in the project takes that path, and paying a
    billed call to rewrite a question into itself is the failure this function
    would otherwise ship by default.
    """
    if not question.strip():
        raise ValueError("question is empty")
    if not history:
        return question
    settings = settings or get_settings()
    turns = list(history)[-settings.history_turns :]
    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
    text, _ = (llm or complete)(
        CONTEXTUALIZE_SYSTEM,
        CONTEXTUALIZE_TEMPLATE.format(history=transcript, question=question),
        model=settings.generation_model,
    )
    resolved = parse_queries(text, limit=1)
    return resolved[0] if resolved else question
