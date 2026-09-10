"""Clone documentation corpora into ``data/raw/<source>/``.

Run by a human, never by the test suite:

    uv run python scripts/fetch_corpus.py [--force]
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

RAW_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataclass(frozen=True)
class Source:
    name: str
    repo: str
    docs_subdir: str
    base_url: str


SOURCES = [
    Source(
        name="fastapi",
        repo="https://github.com/fastapi/fastapi.git",
        docs_subdir="docs/en/docs",
        base_url="https://fastapi.tiangolo.com",
    ),
]


def fetch(source: Source, force: bool) -> None:
    target = RAW_ROOT / source.name
    if target.exists():
        if not force:
            print(f"{source.name}: already present, skipping (use --force to refetch)")
            return
        shutil.rmtree(target)

    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / source.name
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                source.repo,
                str(clone),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(clone), "sparse-checkout", "set", source.docs_subdir],
            check=True,
        )
        docs = clone / source.docs_subdir
        if not docs.is_dir():
            raise SystemExit(f"{source.name}: {source.docs_subdir} missing after sparse checkout")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(docs, target)

    print(f"{source.name}: {sum(1 for _ in target.rglob('*.md'))} markdown files -> {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="refetch sources already on disk")
    args = parser.parse_args()
    for source in SOURCES:
        fetch(source, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
