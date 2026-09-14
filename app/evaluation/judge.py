"""Faithfulness, relevancy and context precision: what Recall@context cannot see.

Every metric this project owns scores retrieval. Step 20 measured the limit of
that on a named question: ``q018`` went from a cited answer to a refusal while
its category's Recall@context did not move, because the right documents were in
the context both times and the code blocks that answered it were not. Refusal
rate caught that one only because the failure happened to be a refusal; it would
have missed a confidently wrong answer entirely.

The only module in this project that imports ``ragas``, for the reason
``rerank.py`` is the only importer of ``flashrank``: a third-party evaluation
framework with its own async model and its own pydantic schemas should reach
exactly one file. Nothing ragas-shaped crosses this boundary in either direction
— ``judge()`` takes ``JudgeSample`` and returns plain floats.

Reference-free on purpose. ``EvalQuestion`` carries document-level ground truth
and no reference answers, by the step 10 decision that chunk-level labels do not
survive a re-chunk. So faithfulness, response relevancy and context precision
without reference are the three that are reachable, and context recall and
answer correctness are an em dash in the README until reference answers exist.
"""

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    Faithfulness,
)

from app.core.config import Settings, get_settings

# ponytail: four at a time, no backoff. A 429 inside a 38-question run fails that
# one sample and is recorded as an error, which is survivable because the arm is
# re-runnable. Add a retry when a run actually loses rows to it.
CONCURRENCY = 4

# One scorer per metric, and it owns *calling* its ragas metric as well as
# constructing it. The three `ascore` signatures do not agree — AnswerRelevancy
# takes no `retrieved_contexts` — so a single uniform `await` at the call site
# would have to branch on the metric name, which is the registry's job.
Scorer = Callable[["JudgeSample"], Awaitable[float]]

# ``Any`` on both sides deliberately: these are ragas' LLM and embedding objects
# and naming their types here would spread the import this module exists to contain.
MetricFactory = Callable[[Any, Any], Scorer]

# Mirrors RETRIEVERS, RERANKERS, TRANSFORMS, COMPRESSORS and STRATEGIES: the
# registry is how this project names N variants and reports them in one table.
#
# Factories rather than instances because both LLM metrics need the *same*
# configured judge, and constructing at import time would build a judge — and
# read an API key — in every process that imports this module, the test suite
# included.
METRICS: dict[str, MetricFactory] = {}


class JudgeSample(BaseModel):
    """One question, the answer it got, and the contexts that answer was built from.

    ``contexts`` is the *compressed* chunk text that actually reached the model,
    never the retrieved pool: faithfulness is a claim about what the model was
    shown, and context precision over chunks the model never saw would score a
    prompt that does not exist.
    """

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    contexts: list[str] = Field(default_factory=list)


class JudgeScores(BaseModel):
    """What one sample scored, and what failed to score.

    Two dicts rather than one with sentinel values: a metric that failed is not a
    metric that scored zero, and a row that cannot tell the difference averages
    the failure into the arm.
    """

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(min_length=1)
    scores: dict[str, float] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)


def _faithfulness(llm: Any, embeddings: Any) -> Scorer:
    """Is every claim in the answer supported by the contexts? Needs no reference."""
    metric = Faithfulness(llm=llm)

    async def score(sample: JudgeSample) -> float:
        result = await metric.ascore(
            user_input=sample.question,
            response=sample.answer,
            retrieved_contexts=list(sample.contexts),
        )
        return float(result.value)

    return score


def _relevancy(llm: Any, embeddings: Any) -> Scorer:
    """Does the answer address the question? Generates questions from the answer
    and takes their cosine against the original, so it needs embeddings too —
    and takes no contexts at all, which is why each entry owns its own call."""
    metric = AnswerRelevancy(llm=llm, embeddings=embeddings)

    async def score(sample: JudgeSample) -> float:
        result = await metric.ascore(user_input=sample.question, response=sample.answer)
        return float(result.value)

    return score


def _context_precision(llm: Any, embeddings: Any) -> Scorer:
    """How much of the context was worth sending, judged against the response."""
    metric = ContextPrecisionWithoutReference(llm=llm)

    async def score(sample: JudgeSample) -> float:
        result = await metric.ascore(
            user_input=sample.question,
            response=sample.answer,
            retrieved_contexts=list(sample.contexts),
        )
        return float(result.value)

    return score


METRICS["faithfulness"] = _faithfulness
METRICS["relevancy"] = _relevancy
METRICS["context_precision"] = _context_precision


def _build_scorers(names: Sequence[str], settings: Settings) -> dict[str, Scorer]:
    """One client, one judge, one embedder, shared by every metric in the run."""
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    llm = llm_factory(settings.judge_model, client=client)
    embeddings = embedding_factory("openai", model=settings.embedding_model, client=client)
    return {name: METRICS[name](llm, embeddings) for name in names}


async def _score_one(
    sample: JudgeSample, scorers: Mapping[str, Scorer], gate: asyncio.Semaphore
) -> JudgeScores:
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}
    for name, scorer in scorers.items():
        async with gate:
            try:
                value = await scorer(sample)
            except Exception as error:  # noqa: BLE001 — one transient API error must
                # not cost a 38-question run; it is recorded as a row, never swallowed.
                errors[name] = f"{type(error).__name__}: {error}"
                continue
        if math.isnan(value):
            # ragas returns NaN when its own parse of a judge response fails.
            # Averaged in, one NaN poisons the arm and the row still looks like
            # a measurement.
            errors[name] = "nan"
            continue
        scores[name] = value
    return JudgeScores(question_id=sample.question_id, scores=scores, errors=errors)


async def _score_all(
    samples: Sequence[JudgeSample], scorers: Mapping[str, Scorer]
) -> list[JudgeScores]:
    gate = asyncio.Semaphore(CONCURRENCY)
    return list(await asyncio.gather(*(_score_one(s, scorers, gate) for s in samples)))


def judge(
    samples: Sequence[JudgeSample],
    *,
    metrics: Sequence[str] | None = None,
    settings: Settings | None = None,
    scorers: Mapping[str, Scorer] | None = None,
) -> list[JudgeScores]:
    """Score every sample on every metric. One row out per row in, in input order.

    ``metrics`` names entries in ``METRICS``; ``None`` runs all three.

    ``scorers`` is injectable so every unit test runs with no network, no key and
    no spend — the pattern ``answer_question``, ``search``, ``expand`` and
    ``compress`` already use for their retrievers, models and embedders.

    Synchronous on the outside and async underneath: everything that calls this
    is a script, and a sync boundary keeps ragas' event loop out of the rest of
    the project for the same reason its types stay in this module.
    """
    settings = settings or get_settings()
    names = list(METRICS) if metrics is None else list(metrics)
    unknown = [name for name in names if name not in METRICS]
    if unknown:
        raise ValueError(f"unknown metric(s) {unknown}; have {sorted(METRICS)}")
    if not samples:
        # Before _build_scorers, which reads the API key. An empty run must not
        # need one — that is what makes the "no questions selected" path testable.
        return []
    return asyncio.run(_score_all(samples, scorers or _build_scorers(names, settings)))


def mean_scores(results: Sequence[JudgeScores]) -> dict[str, float]:
    """Mean per metric over the rows that actually scored.

    A metric with no scored row is absent from the result rather than 0.0. Zero
    is a score; "not measured" is an em dash, and roadmap rule 1 is that the two
    never share a cell.
    """
    totals: dict[str, list[float]] = {}
    for result in results:
        for name, value in result.scores.items():
            totals.setdefault(name, []).append(value)
    return {name: sum(values) / len(values) for name, values in totals.items() if values}
