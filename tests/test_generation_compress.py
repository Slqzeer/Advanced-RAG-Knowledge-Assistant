"""The compressor's contract. No network, no key, no spend: the embedder is faked."""

import math
from collections.abc import Sequence

import pytest

from app.core.config import Settings
from app.generation.compress import COMPRESSORS, GAP, BatchEmbedder, compress
from app.models.chunks import Chunk, ScoredChunk


def make_chunk(text: str, index: int = 0, document_id: str = "fastapi:a") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            document_id=document_id,
            source="fastapi",
            title="A",
            doc_type="tutorial",
            chunk_index=index,
            text=text,
            char_start=0,
            char_end=len(text),
        ),
        score=0.5,
        rank=index + 1,
    )


def fake_embedder(scores: dict[str, float]) -> BatchEmbedder:
    """Returns a 2-D vector per text: the query is (1, 0), a unit is (score, rest).

    Cosine against (1, 0) is then exactly ``scores[text]`` for any unit listed,
    and 0.0 for any that is not. Two dimensions rather than 1 536 because the
    compressor must not care, and a test that needs 1 536 numbers to say
    "this sentence matters more" is a test nobody edits.
    """

    def embed(texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for position, text in enumerate(texts):
            if position == 0:  # the query
                out.append([1.0, 0.0])
                continue
            score = scores.get(text.strip(), 0.0)
            out.append([score, math.sqrt(max(0.0, 1.0 - score * score))])
        return out

    return embed


SETTINGS = Settings(
    _env_file=None,
    compress_method="embedding",
    compress_budget_chars=100,
    compress_candidates=20,
)


def test_compression_off_returns_the_input_unchanged() -> None:
    chunks = [make_chunk("One. Two. Three.")]
    assert compress(chunks, "q", method="", settings=SETTINGS) == chunks


def test_unknown_method_raises_and_names_the_keys() -> None:
    with pytest.raises(ValueError, match="unknown compressor"):
        compress([make_chunk("One.")], "q", method="nope", settings=SETTINGS)
    assert "embedding" in sorted(COMPRESSORS)


def test_empty_query_raises() -> None:
    with pytest.raises(ValueError, match="query is empty"):
        compress([make_chunk("One.")], "   ", method="embedding", settings=SETTINGS)


def test_a_budget_larger_than_the_input_is_a_no_op() -> None:
    chunks = [make_chunk("One. Two.")]
    out = compress(chunks, "q", method="embedding", budget_chars=10_000, settings=SETTINGS)
    assert out == chunks


def test_survivors_are_emitted_in_document_order_with_a_gap_marker() -> None:
    # Sentence 3 scores highest, sentence 1 next, sentence 2 not at all.
    text = "Alpha one. Beta two. Gamma three."
    embed = fake_embedder({"Alpha one.": 0.8, "Beta two.": 0.1, "Gamma three.": 0.9})
    out = compress(
        [make_chunk(text)],
        "q",
        method="embedding",
        budget_chars=len("Alpha one.") + len(" Gamma three.") + len(GAP),
        settings=SETTINGS,
        embedder=embed,
    )
    assert len(out) == 1
    assert out[0].chunk.text == f"Alpha one.{GAP}Gamma three."
    assert "Beta two." not in out[0].chunk.text


def test_a_code_fence_is_kept_whole_or_dropped_whole() -> None:
    fence = "```py\nfrom fastapi import FastAPI.\napp = FastAPI().\n```"
    text = f"Prose lead in. \n\n{fence}\n\nProse tail out."
    embed = fake_embedder({"Prose lead in.": 0.9})
    # A budget that fits the prose but not the fence.
    out = compress(
        [make_chunk(text)],
        "q",
        method="embedding",
        budget_chars=len("Prose lead in.") + 5,
        settings=SETTINGS,
        embedder=embed,
    )
    kept = out[0].chunk.text if out else ""
    assert "```" not in kept, "half a fence is broken code, not shorter code"


def test_a_chunk_with_no_surviving_sentence_disappears() -> None:
    keep, drop = (
        make_chunk("Relevant sentence here.", 0, "fastapi:a"),
        make_chunk("Irrelevant filler text.", 1, "fastapi:b"),
    )
    embed = fake_embedder({"Relevant sentence here.": 0.9, "Irrelevant filler text.": 0.0})
    out = compress(
        [keep, drop],
        "q",
        method="embedding",
        budget_chars=len("Relevant sentence here."),
        settings=SETTINGS,
        embedder=embed,
    )
    assert [c.chunk.document_id for c in out] == ["fastapi:a"]


def test_rank_order_across_chunks_is_preserved() -> None:
    # The second chunk holds the better sentence; it must still come second.
    # Two sentences each, and a budget that fits only the two strong ones, so
    # the compressor really runs — a budget above the total short-circuits and
    # the test would pass without ever exercising the ordering.
    first = make_chunk("Weak filler here. Alpha relevant one.", 0, "fastapi:a")
    second = make_chunk("More weak filler. Beta relevant two.", 1, "fastapi:b")
    embed = fake_embedder({"Alpha relevant one.": 0.9, "Beta relevant two.": 0.95})
    out = compress(
        [first, second],
        "q",
        method="embedding",
        budget_chars=len("Alpha relevant one.") + len("Beta relevant two."),
        settings=SETTINGS,
        embedder=embed,
    )
    assert [c.chunk.document_id for c in out] == ["fastapi:a", "fastapi:b"]
    assert [c.chunk.text for c in out] == ["Alpha relevant one.", "Beta relevant two."]


def test_the_best_sentence_survives_a_budget_too_small_for_it() -> None:
    embed = fake_embedder({"Alpha one.": 0.1, "Beta two.": 0.9})
    out = compress(
        [make_chunk("Alpha one. Beta two.")],
        "q",
        method="embedding",
        budget_chars=1,
        settings=SETTINGS,
        embedder=embed,
    )
    # Never an empty context: an empty one makes a wrong answer certain, the
    # same reasoning build_context() uses to keep its first chunk over budget.
    assert out and out[0].chunk.text.strip() == "Beta two."


def test_the_assembled_text_never_exceeds_the_budget() -> None:
    text = "One sentence. Two sentence. Three sentence. Four sentence. Five sentence."
    embed = fake_embedder(
        {
            "One sentence.": 0.9,
            "Two sentence.": 0.8,
            "Three sentence.": 0.7,
            "Four sentence.": 0.6,
            "Five sentence.": 0.5,
        }
    )
    budget = 40
    out = compress(
        [make_chunk(text)],
        "q",
        method="embedding",
        budget_chars=budget,
        settings=SETTINGS,
        embedder=embed,
    )
    assert sum(len(c.chunk.text) for c in out) <= budget


def test_the_char_span_still_points_at_the_original_document_region() -> None:
    text = "Alpha one. Beta two."
    embed = fake_embedder({"Alpha one.": 0.9})
    out = compress(
        [make_chunk(text)],
        "q",
        method="embedding",
        budget_chars=len("Alpha one."),
        settings=SETTINGS,
        embedder=embed,
    )
    # Provenance, not length: char_start/char_end say where the chunk was cut
    # from, which is what a citation needs and what compression does not change.
    assert (out[0].chunk.char_start, out[0].chunk.char_end) == (0, len(text))


LONG = "Mount the router on the application and include every dependency it needs. "
SHORT = "Use Depends. "


def penalised(penalty: float) -> Settings:
    return SETTINGS.model_copy(update={"compress_length_penalty": penalty})


THIRD = "Return a JSONResponse from the handler. "


def test_alpha_zero_still_produces_step_20s_exact_output() -> None:
    """The control has to be provable, not asserted: every step 20 row in
    answers.jsonl was produced by the path this parameter now shares.

    A golden string rather than "arm A equals arm B": comparing two runs of the
    same configuration passes against an implementation that ignores the penalty
    AND against one that applies it wrongly. The literal below pins selection
    order, document-order reassembly and GAP placement in one assertion.
    """
    chunks = [make_chunk(SHORT + LONG + THIRD)]
    embedder = fake_embedder({SHORT.strip(): 0.60, LONG.strip(): 0.55, THIRD.strip(): 0.50})

    kept = compress(
        chunks,
        "q",
        # Forces the greedy fill. The default budget exceeds this fixture, and
        # compress() would take its `<= budget` early return without ever
        # reaching the code under test.
        budget_chars=len(SHORT) + len(THIRD) + 10,
        settings=penalised(0.0),
        embedder=embedder,
    )

    assert [c.chunk.text for c in kept] == [
        "Use Depends. […] Return a JSONResponse from the handler."
    ]


def test_a_positive_alpha_prefers_the_shorter_unit() -> None:
    """Relevance per character: step 20's hypothesis, and the fractional-knapsack
    ordering. It is also what this design predicts will hurt `code`, because the
    900-character fence q018 lost is exactly the unit it charges most.

    The long unit deliberately scores *higher* on raw relevance than the short
    one. Without that, the short unit wins on raw score alone and the assertion
    holds at every penalty, including none — the test would pass against an
    implementation that ignored this parameter completely.
    """
    chunks = [make_chunk(SHORT + LONG)]
    embedder = fake_embedder({SHORT.strip(): 0.50, LONG.strip(): 0.60})

    kept = compress(
        chunks,
        "q",
        budget_chars=len(LONG) + 4,
        settings=penalised(1.0),
        embedder=embedder,
    )

    assert kept[0].chunk.text == SHORT.strip()


def test_a_negative_alpha_prefers_the_longer_unit() -> None:
    """The direction that would rescue q018: a fence is expensive and embeds
    poorly against a natural-language question, so raw cosine already loses it."""
    chunks = [make_chunk(SHORT + LONG)]
    embedder = fake_embedder({SHORT.strip(): 0.60, LONG.strip(): 0.55})

    kept = compress(
        chunks,
        "q",
        budget_chars=len(LONG) + 4,
        settings=penalised(-0.5),
        embedder=embedder,
    )

    assert kept[0].chunk.text == LONG.strip()


def test_a_positive_alpha_buys_two_cheap_units_instead_of_one_expensive_one() -> None:
    """The mechanism, not just the ordering: at the same budget, charging by
    length fits the 1st and 3rd units and skips the long 2nd, where pure
    relevance order would have spent the budget on the 2nd alone."""
    chunks = [make_chunk(SHORT + LONG + THIRD)]
    embedder = fake_embedder({SHORT.strip(): 0.60, LONG.strip(): 0.55, THIRD.strip(): 0.50})

    kept = compress(
        chunks,
        "q",
        budget_chars=len(SHORT) + len(LONG),
        settings=penalised(1.0),
        embedder=embedder,
    )

    assert "Depends" in kept[0].chunk.text
    assert "JSONResponse" in kept[0].chunk.text
    assert "Mount the router" not in kept[0].chunk.text
    assert GAP in kept[0].chunk.text
