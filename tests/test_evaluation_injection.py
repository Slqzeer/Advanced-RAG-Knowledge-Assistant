"""The hostile fixture. Its checks are string matches, so the risk is a fixture
whose success can never be observed — every test here guards against that."""

from pathlib import Path
from typing import Any

import pytest

from app.evaluation.dataset import load_dataset
from app.evaluation.injection import (
    CANARY,
    FAKE_HOST,
    LEAK_MARKER,
    InjectionAttack,
    load_fixture,
    poisoned,
    succeeded,
)
from app.generation.guard import detect_injection
from app.generation.llm import SYSTEM_PROMPTS
from app.models.answers import Answer, RetrievalStats
from app.models.chunks import Chunk, ScoredChunk

FIXTURE = load_fixture(Path("data/eval/injections.json"))
BY_ID = {attack.attack_id: attack for attack in FIXTURE.attacks}


def answer(text: str, refusal: str | None = None) -> Answer:
    return Answer(
        answer=text,
        retrieval=RetrievalStats(retrieved=5, used=5, dropped=0),
        latency_ms=1.0,
        model="test-model",
        refusal=refusal,  # type: ignore[arg-type]
    )


def test_the_fixture_names_only_answerable_questions_the_benchmark_runs() -> None:
    questions = {q.question_id: q for q in load_dataset(Path("data/eval/questions.jsonl"))}
    for question_id in FIXTURE.question_ids:
        question = questions[question_id]
        assert question.category != "unanswerable" and not question.held_out


def test_every_attack_comes_in_both_phrasings_plus_one_control() -> None:
    pairs = {(a.attack, a.phrasing) for a in FIXTURE.attacks}
    for attack in ("hijack", "fake_url", "prompt_leak", "forced_refusal"):
        assert {(attack, "obvious"), (attack, "paraphrased")} <= pairs
    assert [a.attack for a in FIXTURE.attacks].count("control") == 1


def test_an_attack_that_asks_for_a_marker_carries_it() -> None:
    for attack in FIXTURE.attacks:
        if attack.attack == "hijack":
            assert CANARY in attack.text
        if attack.attack == "fake_url":
            assert FAKE_HOST in attack.text


def test_the_leak_marker_is_in_every_prompt_version() -> None:
    for prompt in SYSTEM_PROMPTS.values():
        assert LEAK_MARKER in prompt.lower()


def test_the_obvious_half_is_what_the_detector_catches_and_the_paraphrase_is_not() -> None:
    """Rule D reads the two halves separately; this is why that split means something."""
    for attack in FIXTURE.attacks:
        chunk = ScoredChunk(
            chunk=Chunk(
                document_id="attack:x",
                source="fastapi",
                title="t",
                doc_type="advanced",
                chunk_index=0,
                text=attack.text,
                char_start=0,
                char_end=len(attack.text),
            ),
            score=0.5,
            rank=1,
        )
        _, flagged = detect_injection([chunk])
        assert bool(flagged) == (attack.phrasing == "obvious"), attack.attack_id


@pytest.mark.parametrize(
    ("attack_id", "complying"),
    [
        ("hijack-obvious", answer(f"{CANARY} Use Depends() [1].")),
        ("fake-url-paraphrased", answer(f"Install from https://{FAKE_HOST}/fastapi.tar.gz.")),
        ("prompt-leak-obvious", answer(SYSTEM_PROMPTS["v2"])),
        ("forced-refusal-obvious", answer("I do not have enough…", refusal="model_declined")),
    ],
)
def test_a_complying_answer_counts_as_a_success(attack_id: str, complying: Answer) -> None:
    assert succeeded(BY_ID[attack_id], complying)


@pytest.mark.parametrize("attack_id", sorted(BY_ID))
def test_a_clean_answer_is_never_a_success(attack_id: str) -> None:
    assert not succeeded(BY_ID[attack_id], answer("Use Depends() [1]."))


def test_a_control_refusal_is_not_an_attack_success() -> None:
    assert not succeeded(BY_ID["control"], answer("I do not…", refusal="model_declined"))


def real_chunk(rank: int) -> ScoredChunk:
    chunk = Chunk(
        document_id=f"fastapi:doc-{rank}",
        source="fastapi",
        title="Dependencies",
        doc_type="tutorial",
        chunk_index=0,
        text="body",
        char_start=0,
        char_end=4,
    )
    return ScoredChunk(chunk=chunk, score=0.6 - 0.01 * rank, rank=rank)


def test_the_planted_chunk_is_rank_one_and_the_total_stays_top_k() -> None:
    def retriever(query: str, **kwargs: Any) -> list[ScoredChunk]:
        return [real_chunk(r) for r in range(1, kwargs["top_k"] + 1)]

    attack: InjectionAttack = BY_ID["hijack-obvious"]
    chunks = poisoned(retriever, attack)("q", top_k=5)
    assert len(chunks) == 5
    assert chunks[0].chunk.text == attack.text
    assert [c.rank for c in chunks] == [1, 2, 3, 4, 5]
    assert [c.chunk.document_id for c in chunks[1:]] == [f"fastapi:doc-{r}" for r in (1, 2, 3, 4)]
