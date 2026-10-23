import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_observability as bench  # noqa: E402


def test_rule_o_requires_every_line_and_sub_five_ms_p95() -> None:
    assert bench.rule_o(valid_lines=1000, requested=1000, p95_ms=4.999) is True
    assert bench.rule_o(valid_lines=999, requested=1000, p95_ms=1.0) is False
    assert bench.rule_o(valid_lines=1000, requested=1000, p95_ms=5.0) is False


def test_benchmark_writes_and_validates_the_requested_number(tmp_path) -> None:
    result = bench.benchmark(count=3, directory=tmp_path)
    assert result["writes"] == result["valid_lines"] == 3
    assert result["bytes_per_trace"] > 4_000
    assert result["p50_ms"] >= 0 and result["p95_ms"] >= result["p50_ms"]
