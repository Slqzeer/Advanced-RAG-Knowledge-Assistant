"""Read a directory of Markdown files into validated :class:`RawDocument` objects."""

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

from app.models.documents import RawDocument

MARKDOWN_SUFFIXES = (".md", ".markdown")
HEADING = re.compile(r"^#\s+(.+)$")
# MkDocs attr-list anchors: "# Title { #slug }".
ATTR_LIST = re.compile(r"\s*\{[^}]*\}\s*$")
FRONT_MATTER = re.compile(r"\A---\r?\n.*?\r?\n---[ \t]*\r?\n", re.DOTALL)


def iter_markdown_files(root: Path) -> Iterator[Path]:
    """Yield every Markdown file under ``root``, skipping dot-directories."""
    for path in root.rglob("*"):
        if path.suffix.lower() not in MARKDOWN_SUFFIXES or not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        yield path


def _extract_title(text: str, path: Path) -> str:
    body = FRONT_MATTER.sub("", text)
    for line in body.splitlines():
        match = HEADING.match(line)
        if match:
            title = ATTR_LIST.sub("", match.group(1)).rstrip("#")
            return title.strip().replace("`", "")
    return path.stem.replace("-", " ").replace("_", " ").title()


def doc_type_of(relative_path: str) -> str:
    """The corpus section a path belongs to: its first directory, or ``root``.

    Derived rather than mapped, so a second source with a different layout gets
    its own facets for free instead of silently collapsing into ``other``.
    """
    head, _, tail = relative_path.partition("/")
    return head if tail else "root"


def load_document(path: Path, root: Path, source: str, base_url: str | None = None) -> RawDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    relative = path.relative_to(root).as_posix()
    stem = relative[: -len(path.suffix)]
    # MkDocs serves "a/index.md" at "/a/" and "index.md" at "/".
    slug = "" if stem == "index" else stem.removesuffix("/index")
    return RawDocument(
        document_id=f"{source}:{stem}",
        source=source,
        title=_extract_title(text, path),
        path=relative,
        doc_type=doc_type_of(relative),
        url=None if base_url is None else f"{base_url}/{slug}/" if slug else f"{base_url}/",
        text=text,
        content_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def load_documents(root: Path, source: str, base_url: str | None = None) -> list[RawDocument]:
    documents = [load_document(p, root, source, base_url) for p in iter_markdown_files(root)]
    return sorted(documents, key=lambda d: d.document_id)
