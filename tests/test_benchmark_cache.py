import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_cache as bench  # noqa: E402


def test_sweep_counts_a_paraphrase_nearer_the_wrong_anchor_as_false(monkeypatch):
    anchors = [
        SimpleNamespace(question_id="a", question="a"),
        SimpleNamespace(question_id="b", question="b"),
    ]
    probes = [
        SimpleNamespace(question_id="p", question="b", anchor_id="a", kind="paraphrase"),
        SimpleNamespace(question_id="n", question="n", anchor_id="a", kind="near_miss"),
    ]
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0], "n": [-1.0, 0.0]}
    monkeypatch.setattr(bench, "anchors", lambda: anchors)
    monkeypatch.setattr(bench, "pairs", lambda: probes)
    monkeypatch.setattr(bench, "default_embedder", lambda _: vectors.__getitem__)
    result = bench.sweep()
    assert result["candidate_threshold"] == pytest.approx(1.02)
    assert result["must_miss"] == 2
    assert result["paraphrase_hits"] == 0
    assert result["rule_s"] is False


def test_cosine_rejects_different_dimensions():
    with pytest.raises(ValueError):
        bench.cosine([1.0, 0.0], [1.0])
