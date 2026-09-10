"""Turn raw MkDocs Markdown into text worth embedding.

The one rule that matters: code survives byte-for-byte. Exact strings like
``HTTPException(status_code=422)`` only exist inside code blocks, and they are
what keyword retrieval later has to match on. Every block is swapped for a
placeholder before any other rule runs, and restored last.
"""

import hashlib
import re
import unicodedata

from app.models.documents import RawDocument

MIN_PROSE_LENGTH = 200

PLACEHOLDER = "\x00CODE{}\x00"
PLACEHOLDER_RE = re.compile(r"\x00CODE(\d+)\x00")

FRONT_MATTER = re.compile(r"\A---\n.*?\n---[ \t]*\n", re.DOTALL)
# Fenced (``` or ~~~, any length >= 3), indented (4 spaces after a blank line),
# and inline spans. Inline spans are protected too: they hold identifiers, and
# the HTML and link rules below would happily eat `<div>` or `[a](b)`.
CODE = re.compile(
    r"(?P<fence>^(?P<f>```+|~~~+)(?P<info>[^\n]*)\n(?P<body>.*?)^(?P=f)[ \t]*$)"
    r"|(?P<indented>(?<=\n\n)(?:[ \t]*\n)*(?: {4}|\t)[^\n]*(?:\n(?:[ \t]*\n)*(?: {4}|\t)[^\n]*)*)"
    r"|(?P<inline>(?P<b>`+)(?!`).*?(?<!`)(?P=b)(?!`))",
    re.DOTALL | re.MULTILINE,
)
# Fence info strings: keep the language, drop `hl_lines=".."` and the rest of an
# attr list. The language may be a class inside it: "```{ .dockerfile .annotate }".
FENCE_LANGUAGE = re.compile(r"[ \t]*\{?[ \t]*\.?([A-Za-z0-9_+-]+)")
# MkDocs "termy" console blocks are ANSI colour rendered as HTML. Pure noise, and
# it breaks the exact strings keyword retrieval needs. Console fences only: the
# tags inside an html or jinja fence are the content.
TERMINAL_HTML = re.compile(r"</?(?:font|span|u|b)\b[^<>]*>")
LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])\s")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# MkDocs include flavours: pymdownx snippets, the old {!...!} and the current {*...*}.
INCLUDE = re.compile(r"^[ \t]*(?:--8<--[^\n]*|\{!.*?!\}|\{\*.*?\*\})[ \t]*$\n?", re.MULTILINE)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Not autolinks: "<https://x>" is content, "<div class=...>" is not.
HTML_TAG = re.compile(r"</?(?!https?:)[a-zA-Z][^<>]*>")
# Attribute lists: "# Title { #anchor }" or "text { .annotate }" at end of line.
ATTR_LIST = re.compile(r"[ \t]*\{[ \t]*[#.][^}\n]*\}[ \t]*$", re.MULTILINE)
BANG_ADMONITION = re.compile(r'^(?P<indent>[ \t]*)!!!\s+\S+(?:\s+"(?P<title>[^"]*)")?[ \t]*$')
SLASH_ADMONITION = re.compile(r"^/{3,}(?:\s+\S+(?:\s*\|\s*(?P<title>[^\n]*?))?)?[ \t]*$")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
BLANK_LINES = re.compile(r"\n{3,}")


def _continues_a_list(text: str, start: int) -> bool:
    """True when the indented run at ``start`` is list-item body, not a code block."""
    for line in reversed(text[:start].split("\n")):
        if line.strip():
            return bool(LIST_ITEM.match(line)) or line.startswith("  ")
    return False


def _flatten_admonitions(text: str) -> str:
    """Drop admonition markers, keep their titles, de-indent ``!!!`` bodies.

    Runs before code extraction on purpose: a ``!!!`` body is 4-space indented,
    so extracting indented code first would swallow the whole admonition.
    """
    out: list[str] = []
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        slash = SLASH_ADMONITION.match(line)
        if slash:
            title = slash.group("title")
            if title:
                out.append(title.strip())
            index += 1
            continue
        bang = BANG_ADMONITION.match(line)
        if not bang:
            out.append(line)
            index += 1
            continue
        title = bang.group("title")
        if title:
            out.append(title)
        index += 1
        body_indent = len(bang.group("indent")) + 4
        while index < len(lines) and (
            not lines[index].strip() or lines[index].startswith(" " * body_indent)
        ):
            out.append(lines[index][body_indent:] if lines[index].strip() else "")
            index += 1
    return "\n".join(out)


def clean_markdown(text: str) -> str:
    """Strip MkDocs machinery and markup noise, leaving prose and code intact."""
    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = text.replace("\u00a0", " ").replace("\x00", "")
    text = FRONT_MATTER.sub("", text)
    text = _flatten_admonitions(text)

    blocks: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        block = match.group(0)
        if match.group("fence") is not None:
            language = FENCE_LANGUAGE.match(match.group("info"))
            name = language.group(1) if language else ""
            body = match.group("body")
            if name.lower() == "console":
                body = TERMINAL_HTML.sub("", body)
            block = f"{match.group('f')}{name}\n{body}{match.group('f')}"
        elif match.group("indented") is not None and _continues_a_list(match.string, match.start()):
            return block  # indented list body, not code — let the prose rules run
        blocks.append(block)
        return PLACEHOLDER.format(len(blocks) - 1)

    # "\n\n" prefix so the indented-block lookbehind can fire on line 1.
    text = CODE.sub(_stash, "\n\n" + text)[2:]

    text = HTML_COMMENT.sub("", text)
    text = INCLUDE.sub("", text)
    text = IMAGE.sub("", text)
    text = HTML_TAG.sub("", text)
    text = ATTR_LIST.sub("", text)
    text = LINK.sub(r"\1", text)
    text = TRAILING_SPACE.sub("", text)
    text = BLANK_LINES.sub("\n\n", text)

    restored = PLACEHOLDER_RE.sub(lambda m: blocks[int(m.group(1))], text)
    if PLACEHOLDER_RE.search(restored) or "\x00" in restored:
        raise AssertionError("code placeholder was corrupted during cleaning")
    return restored.strip()


def prose_length(text: str) -> int:
    """Length of ``text`` with code blocks removed — code is not prose."""
    return len(CODE.sub("", "\n\n" + text).strip())


def clean_document(document: RawDocument) -> RawDocument | None:
    """Return ``document`` with cleaned text and a fresh hash, or ``None`` if it is a stub."""
    text = clean_markdown(document.text)
    if prose_length(text) < MIN_PROSE_LENGTH:
        return None
    return document.model_copy(
        update={"text": text, "content_hash": hashlib.sha256(text.encode()).hexdigest()}
    )
