import hashlib
from textwrap import dedent

import pytest

from app.ingestion.clean import MIN_PROSE_LENGTH, clean_document, clean_markdown
from app.models.documents import RawDocument

FENCED_CODE = dedent(
    """\
    Some prose.

    ```python
    @app.get("/items/{item_id}")
    async def read(item_id: int):

        if item_id > 10:
            raise HTTPException(status_code=422)
    ```

    More prose.
    """
)

TILDE_CODE = dedent(
    """\
    Some prose.

    ~~~python
    raise HTTPException(status_code=422)
    ~~~

    More prose.
    """
)

INDENTED_CODE = dedent(
    """\
    Some prose.

        raise HTTPException(status_code=422)
        # a [link](x.md) that must survive

    More prose.
    """
)

FRONT_MATTER = dedent(
    """\
    ---
    title: Settings
    tags: [a, b]
    ---

    # Settings

    Before.

    ---

    After.
    """
)

COMMENTS = "A <!-- one line --> B\n\n<!-- multi\nline\ncomment -->\n\nC\n"

INCLUDES = dedent(
    """\
    A

    --8<-- "docs/x.md"
    {!../../docs_src/templates/item.html!}
    {!> ../../docs_src/generate_clients/tutorial004.js!}
    {* ../../docs_src/behind_a_proxy/tutorial001.py hl[6] *}

    B
    """
)

BANG_ADMONITION = dedent(
    """\
    Before.

    !!! tip "Check this"
        Body line one.

        Body line two.

    After.
    """
)

SLASH_ADMONITION = dedent(
    """\
    Before.

    /// note | Technical Details

    Body line.

    ///

    //// tab | Python 3.10+

    Tab body.

    ////

    /// warning

    Warned.

    ///

    After.
    """
)

ATTR_LISTS = dedent(
    """\
    # Settings and Environment Variables { #settings-and-environment-variables }

    A paragraph. { .annotate }

    ## Sub {#sub}
    """
)

LINKS = dedent(
    """\
    See [Depends](../x.md) and [the docs](https://example.com "T").

    Bare <https://example.com/keep-me> stays.

    ![Swagger UI](https://fastapi.tiangolo.com/img/a.png) is gone.
    """
)

HTML = dedent(
    """\
    <div class="termy">

    An <abbr title="Hypertext">HTTP</abbr> request.

    </div>

    <img src="x.png" alt="nope">
    """
)

WHITESPACE = "A   \n\n\n\n\nB\t\n"

UNICODE = "caf\u00e9\u00a0latte \uff21\uff22"

FIXTURES = [
    FENCED_CODE,
    TILDE_CODE,
    INDENTED_CODE,
    FRONT_MATTER,
    COMMENTS,
    INCLUDES,
    BANG_ADMONITION,
    SLASH_ADMONITION,
    ATTR_LISTS,
    LINKS,
    HTML,
    WHITESPACE,
    UNICODE,
]


@pytest.mark.parametrize("text", [FENCED_CODE, TILDE_CODE, INDENTED_CODE])
def test_code_blocks_survive_verbatim(text: str) -> None:
    assert "raise HTTPException(status_code=422)" in clean_markdown(text)


def test_fenced_block_keeps_blank_lines_and_indentation() -> None:
    cleaned = clean_markdown(FENCED_CODE)
    assert '@app.get("/items/{item_id}")\nasync def read(item_id: int):\n\n    if' in cleaned


def test_indented_block_is_not_link_collapsed() -> None:
    assert "[link](x.md)" in clean_markdown(INDENTED_CODE)


def test_front_matter_goes_but_horizontal_rule_stays() -> None:
    cleaned = clean_markdown(FRONT_MATTER)
    assert "title: Settings" not in cleaned
    assert cleaned.startswith("# Settings")
    assert "\n---\n" in cleaned


def test_html_comments_are_removed() -> None:
    cleaned = clean_markdown(COMMENTS)
    assert "<!--" not in cleaned and "multi" not in cleaned
    assert "A" in cleaned and "B" in cleaned and "C" in cleaned


def test_mkdocs_includes_are_removed() -> None:
    cleaned = clean_markdown(INCLUDES)
    assert cleaned == "A\n\nB"


def test_bang_admonition_keeps_title_and_dedents_body() -> None:
    cleaned = clean_markdown(BANG_ADMONITION)
    assert "!!!" not in cleaned
    assert "Check this" in cleaned
    assert "\nBody line one.\n" in cleaned
    assert "\nBody line two.\n" in cleaned


def test_slash_admonition_markers_go_and_titles_stay() -> None:
    cleaned = clean_markdown(SLASH_ADMONITION)
    assert "///" not in cleaned
    assert "Technical Details" in cleaned
    assert "Python 3.10+" in cleaned
    assert "\nwarning\n" not in cleaned
    assert "Warned." in cleaned


def test_attribute_lists_are_stripped() -> None:
    cleaned = clean_markdown(ATTR_LISTS)
    assert "{" not in cleaned
    assert cleaned.startswith("# Settings and Environment Variables\n")
    assert "A paragraph.\n" in cleaned
    assert "## Sub" in cleaned


def test_links_collapse_and_images_vanish() -> None:
    cleaned = clean_markdown(LINKS)
    assert "See Depends and the docs." in cleaned
    assert "<https://example.com/keep-me>" in cleaned
    assert "Swagger UI" not in cleaned and "a.png" not in cleaned
    assert "is gone." in cleaned


def test_html_tags_are_stripped_but_text_survives() -> None:
    cleaned = clean_markdown(HTML)
    assert "<" not in cleaned and ">" not in cleaned
    assert "An HTTP request." in cleaned
    assert "nope" not in cleaned


def test_whitespace_is_normalised() -> None:
    assert clean_markdown(WHITESPACE) == "A\n\nB"


def test_unicode_is_nfkc_normalised() -> None:
    assert clean_markdown(UNICODE) == "café latte AB"


@pytest.mark.parametrize("text", FIXTURES)
def test_cleaning_is_idempotent(text: str) -> None:
    once = clean_markdown(text)
    assert clean_markdown(once) == once


def _document(text: str) -> RawDocument:
    return RawDocument(
        document_id="fastapi:docs/index",
        source="fastapi",
        title="FastAPI",
        path="docs/index.md",
        doc_type="docs",
        url=None,
        text=text,
        content_hash="0" * 64,
    )


def test_clean_document_rewrites_text_and_hash() -> None:
    raw = _document("# Title { #title }\n\n" + "word " * 60)
    cleaned = clean_document(raw)
    assert cleaned is not None
    assert cleaned.text == clean_markdown(raw.text)
    assert cleaned.content_hash == hashlib.sha256(cleaned.text.encode()).hexdigest()
    assert cleaned.document_id == raw.document_id


def test_clean_document_drops_stubs() -> None:
    assert clean_document(_document("# Redirect\n\nSee [elsewhere](x.md).\n")) is None


def test_clean_document_ignores_code_when_measuring_prose() -> None:
    long_code = "```python\n" + "x = 1\n" * 200 + "```"
    assert clean_document(_document(f"# Title\n\n{long_code}\n")) is None
    assert len(long_code) > MIN_PROSE_LENGTH


# --- edge cases found by running the cleaner over the real FastAPI corpus ---

FENCE_ATTRS = dedent(
    """\
    Text.

    ```{ .dockerfile .annotate hl_lines="10  13" }
    FROM python:3.14
    ```

    ```jinja hl_lines="7"
    {{ id }}
    ```
    """
)

TERMY = dedent(
    """\
    Text.

    ```console
    $ fastapi run
    <span style="color: green;">INFO</span>:     Uvicorn running on <u style="x">http://127.0.0.1:8000</u>
    ```
    """
)

HTML_FENCE = dedent(
    """\
    Text.

    ```html
    <span class="x">hi</span>
    ```
    """
)

LIST_CONTINUATION = dedent(
    """\
    Text.

    1. First annotation.

        A refresher on [`async` and `await`](../async.md#in-a-hurry) helps.

    2. Second annotation.
    """
)

FIXTURES.extend([FENCE_ATTRS, TERMY, HTML_FENCE, LIST_CONTINUATION])


def test_fence_attributes_go_but_language_and_code_stay() -> None:
    cleaned = clean_markdown(FENCE_ATTRS)
    assert "```dockerfile\nFROM python:3.14\n```" in cleaned
    assert "```jinja\n{{ id }}\n```" in cleaned
    assert "hl_lines" not in cleaned and ".annotate" not in cleaned


def test_console_blocks_lose_terminal_colour_html() -> None:
    cleaned = clean_markdown(TERMY)
    assert "<span" not in cleaned and "<u " not in cleaned
    assert "$ fastapi run" in cleaned
    assert "INFO:     Uvicorn running on http://127.0.0.1:8000" in cleaned


def test_html_inside_a_non_console_fence_is_untouched() -> None:
    assert '<span class="x">hi</span>' in clean_markdown(HTML_FENCE)


def test_indented_list_body_is_cleaned_not_treated_as_code() -> None:
    cleaned = clean_markdown(LIST_CONTINUATION)
    assert "(../async.md#in-a-hurry)" not in cleaned
    assert "A refresher on `async` and `await` helps." in cleaned
