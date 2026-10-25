"""Rebuild the offline benchmark dashboard snapshot from committed histories."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.dashboard import build_snapshot, write_data_script  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/eval"))
    parser.add_argument("--output", type=Path, default=Path("dashboard/data.js"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = build_snapshot(args.source)
    write_data_script(snapshot, args.output)
    unique = sum(len(snapshot[kind]) for kind in ("retrieval", "answers", "cache", "observability"))
    total = sum(snapshot["counts"].values())
    print(f"Wrote {args.output} with {unique} labelled views from {total} history rows")


if __name__ == "__main__":
    main()
