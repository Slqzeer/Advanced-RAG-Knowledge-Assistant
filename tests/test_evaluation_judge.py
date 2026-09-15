"""The judge's contract. No network, no key, no spend: the scorers are faked."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.evaluation.judge import METRICS, JudgeSample, judge, mean_scores

SETTINGS = Settings(_env_file=None, judge_model="test-judge")


def sample(question_id: str, question: str, contexts: list[str] | None = None) -> JudgeSample:
    return JudgeSample(
        question_id=question_id,
        question=question,
        answer="Use Depends() [1].",
        contexts=["FastAPI provides Depends()."] if contexts is None else contexts,
    )


class FakeScorer:
    """A stand-in ``Scorer``: an async callable taking a JudgeSample.

    Keyed on the question text rather than on call order, because a judge that
    silently returned results in a different order than it was given them would
    attach every score to the wrong question, and a call-order fake cannot see it.
    """

    def __init__(self, scores: dict[str, float], raises_on: set[str] | None = None) -> None:
        self.scores = scores
        self.raises_on = raises_on or set()
        self.seen: list[JudgeSample] = []

    async def __call__(self, sample: JudgeSample) -> float:
        self.seen.append(sample)
        if sample.question in self.raises_on:
            raise RuntimeError("the judge exploded")
        return self.scores[sample.question]


class StubMetric:
    """Stands in for a ragas metric class and records the kwargs it was called with."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs

    async def ascore(self, **kwargs: Any) -> Any:
        StubMetric.last_kwargs = kwargs
        return SimpleNamespace(value=0.75)


def test_scores_come_back_one_row_per_sample_in_input_order() -> None:
    samples = [sample("q001", "first?"), sample("q002", "second?")]
    scorer = FakeScorer({"first?": 0.9, "second?": 0.2})

    results = judge(samples, settings=SETTINGS, scorers={"faithfulness": scorer})

    assert [result.question_id for result in results] == ["q001", "q002"]
    assert [result.scores["faithfulness"] for result in results] == [0.9, 0.2]


def test_the_whole_sample_reaches_the_scorer() -> None:
    scorer = FakeScorer({"first?": 1.0})

    judge([sample("q001", "first?")], settings=SETTINGS, scorers={"faithfulness": scorer})

    sent = scorer.seen[0]
    assert sent.question == "first?"
    assert sent.answer == "Use Depends() [1]."
    assert sent.contexts == ["FastAPI provides Depends()."]


def test_each_metric_is_called_with_the_kwargs_its_own_signature_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three ragas ``ascore`` signatures do not agree: AnswerRelevancy takes
    no ``retrieved_contexts``. Getting that wrong is a TypeError thirty-eight
    questions into an arm that has already been paid for, which is why the
    registry entry owns the call and this test pins each one."""
    for attribute in ("Faithfulness", "AnswerRelevancy", "ContextPrecisionWithoutReference"):
        monkeypatch.setattr(f"app.evaluation.judge.{attribute}", StubMetric)
    one = sample("q001", "first?")

    for name, expected in [
        ("faithfulness", {"user_input", "response", "retrieved_contexts"}),
        ("relevancy", {"user_input", "response"}),
        ("context_precision", {"user_input", "response", "retrieved_contexts"}),
    ]:
        assert asyncio.run(METRICS[name](None, None)(one)) == 0.75, name
        assert set(StubMetric.last_kwargs) == expected, name
        assert StubMetric.last_kwargs["user_input"] == "first?", name


def test_a_failure_on_one_sample_leaves_every_other_sample_scored() -> None:
    """One transient API error must not cost a 38-question run — the rule
    ``benchmark_answers.py`` already states for a generation failure."""
    samples = [sample("q001", "first?"), sample("q002", "second?"), sample("q003", "third?")]
    scorer = FakeScorer({"first?": 0.9, "third?": 0.4}, raises_on={"second?"})

    results = judge(samples, settings=SETTINGS, scorers={"faithfulness": scorer})

    assert [result.question_id for result in results] == ["q001", "q002", "q003"]
    assert results[1].scores == {}
    assert "the judge exploded" in results[1].errors["faithfulness"]
    assert results[2].scores["faithfulness"] == 0.4


def test_a_stuck_call_times_out_and_the_other_samples_still_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One stuck call must not hold the whole arm for 600s under the SDK default.
    CALL_TIMEOUT_S is patched to a tiny value so the test does not actually wait."""
    monkeypatch.setattr("app.evaluation.judge.CALL_TIMEOUT_S", 0.05)

    class SlowScorer:
        async def __call__(self, sample: JudgeSample) -> float:
            if sample.question == "slow?":
                await asyncio.sleep(10.0)
                return 1.0
            return 0.9

    samples = [sample("q001", "slow?"), sample("q002", "fast?")]
    results = judge(samples, settings=SETTINGS, scorers={"faithfulness": SlowScorer()})

    assert results[0].scores == {}
    assert "timeout" in results[0].errors["faithfulness"].lower()
    assert "0.05" in results[0].errors["faithfulness"]
    assert results[1].scores["faithfulness"] == 0.9


def test_a_nan_is_recorded_as_an_error_not_as_a_score() -> None:
    """ragas returns NaN when its own internal parse fails. Averaged in, one NaN
    poisons the whole arm and the row looks like a measurement."""
    scorer = FakeScorer({"first?": float("nan")})

    results = judge([sample("q001", "first?")], settings=SETTINGS, scorers={"faithfulness": scorer})

    assert results[0].scores == {}
    assert results[0].errors["faithfulness"] == "nan"


def test_an_empty_context_list_is_scored_rather_than_skipped() -> None:
    """A refusal with no context is a real row. Skipping it is how a compressor
    that refuses more often scores better."""
    scorer = FakeScorer({"first?": 0.0})

    results = judge(
        [sample("q001", "first?", contexts=[])], settings=SETTINGS, scorers={"faithfulness": scorer}
    )

    assert results[0].scores["faithfulness"] == 0.0


def test_an_unknown_metric_name_raises_and_names_the_available_ones() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        judge([sample("q001", "first?")], metrics=["nope"], settings=SETTINGS)


def test_no_samples_is_an_empty_list_and_builds_no_judge() -> None:
    """Constructing the judge reads the API key. An empty run must not need one."""
    assert judge([], settings=SETTINGS) == []


def test_mean_scores_averages_only_the_rows_that_scored() -> None:
    samples = [sample("q001", "first?"), sample("q002", "second?"), sample("q003", "third?")]
    scorer = FakeScorer({"first?": 1.0, "third?": 0.0}, raises_on={"second?"})

    results = judge(samples, settings=SETTINGS, scorers={"faithfulness": scorer})

    assert mean_scores(results) == {"faithfulness": 0.5}


def test_mean_scores_of_nothing_is_empty_not_zero() -> None:
    """Zero is a score. "Not measured" is an em dash — roadmap rule 1."""
    assert mean_scores([]) == {}


# --- the calibration gate, against the real judge --------------------------
#
# Part A of the spec: before any measurement arm runs, four fixtures prove the
# judge separates good from bad rather than scoring everything the same. Three
# check that a bad answer scores low; the fourth checks a good one scores
# high, because a metric stuck near zero would pass the first three and still
# be useless.

CONTEXT = (
    "FastAPI provides Depends(), which you place in the signature of a path "
    "operation function to declare a dependency. Dependencies may themselves "
    "declare dependencies, and FastAPI resolves the whole tree per request."
)


@pytest.mark.requires_api
def test_a_refusal_scores_near_zero_on_relevancy() -> None:
    """Part A, clause 1. This is q018's exact failure shape, and a judge that
    cannot see it cannot judge the arm it was built to judge."""
    results = judge(
        [
            JudgeSample(
                question_id="cal-refusal",
                question="How do I declare a dependency in FastAPI?",
                answer=(
                    "I could not find anything about this in the indexed "
                    "documentation, so I cannot answer it."
                ),
                contexts=[CONTEXT],
            )
        ],
        metrics=["relevancy"],
    )

    assert results[0].scores["relevancy"] < 0.3, results[0]


@pytest.mark.requires_api
def test_a_fabricated_claim_scores_low_on_faithfulness() -> None:
    """Part A, clause 2. Every claim below is absent from the context, and the
    answer is fluent, well-formed and confident — the failure refusal rate misses."""
    results = judge(
        [
            JudgeSample(
                question_id="cal-fabrication",
                question="How do I declare a dependency in FastAPI?",
                answer=(
                    "Declare it with the @dependency decorator, which caches the "
                    "result in Redis for 300 seconds by default and retries three "
                    "times before raising DependencyTimeout."
                ),
                contexts=[CONTEXT],
            )
        ],
        metrics=["faithfulness"],
    )

    assert results[0].scores["faithfulness"] < 0.5, results[0]


@pytest.mark.requires_api
def test_a_good_full_answer_scores_high_on_relevancy() -> None:
    """Part A, clause 4 - the other half of clause 1. Three clauses that all
    check a bad answer scores low would pass against a metric stuck near zero,
    which would then report every compression arm as equally bad."""
    results = judge(
        [
            JudgeSample(
                question_id="cal-good",
                question="How do I declare a dependency in FastAPI?",
                answer=(
                    "In FastAPI you declare a dependency by placing Depends() in the "
                    "signature of your path operation function. You pass the callable "
                    "that produces the value, for example "
                    "`def read_items(commons = Depends(common_parameters))`. FastAPI "
                    "resolves the whole dependency tree once per request, and "
                    "dependencies may themselves declare further dependencies."
                ),
                contexts=[CONTEXT],
            )
        ],
        metrics=["relevancy"],
    )

    assert results[0].scores["relevancy"] > 0.7, results[0]


PADDING = [
    "Deploy with Docker by writing a Dockerfile that runs uvicorn.",
    "WebSocket endpoints are declared with @app.websocket.",
    "Release notes for 0.115.0: fixed a regression in form parsing.",
]


def _precision_sample(question_id: str, contexts: list[str]) -> JudgeSample:
    return JudgeSample(
        question_id=question_id,
        question="How do I declare a dependency in FastAPI?",
        answer="You declare it with Depends() in the path operation signature.",
        contexts=contexts,
    )


@pytest.mark.requires_api
def test_context_precision_charges_for_padding_above_the_relevant_chunk() -> None:
    """Part A, clause 3, with the fixture corrected after Task 6 measured what this
    metric is.

    `ContextPrecisionWithoutReference` is average precision over **ranked** verdicts,
    not precision over a set. A relevant chunk at rank 1 scores 1.0 however much junk
    follows it, so the original fixture — relevant chunk first, three irrelevant after
    — measured the metric's definition rather than the judge's discrimination, and
    returned 0.9999999999. The threshold is unchanged; the ordering is.

    Both directions are asserted, because "it scored low" on its own does not
    distinguish a discriminating metric from a broken one. The same four chunks
    ranked well must score high, and ranked badly must score low.
    """
    buried = judge(
        [_precision_sample("cal-precision-buried", [*PADDING, CONTEXT])],
        metrics=["context_precision"],
    )
    surfaced = judge(
        [_precision_sample("cal-precision-surfaced", [CONTEXT, *PADDING])],
        metrics=["context_precision"],
    )

    assert buried[0].scores["context_precision"] < 0.6, buried[0]
    assert surfaced[0].scores["context_precision"] > 0.9, surfaced[0]
