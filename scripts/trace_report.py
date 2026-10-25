"""Summarize local per-query traces without printing their content."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.observability.metrics import load_traces, summarize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    path = args.path or get_settings().trace_path
    if path is None:
        parser.error("provide PATH or configure TRACE_PATH")
    summary = summarize(load_traces(path))
    for name, value in summary.items():
        print(f"{name:<24} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
