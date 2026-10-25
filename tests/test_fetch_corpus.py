from pathlib import Path

from scripts import fetch_corpus


def test_fetch_checks_out_the_pinned_revision(monkeypatch, tmp_path: Path) -> None:
    revision = "50113da16fec53b66b80d75e80a89296de4fa5a5"
    source = fetch_corpus.Source(
        name="docs",
        repo="https://example.com/docs.git",
        docs_subdir="docs/en/docs",
        base_url="https://example.com",
        revision=revision,
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(command)
        if "sparse-checkout" in command:
            clone = Path(command[2])
            docs = clone / source.docs_subdir
            docs.mkdir(parents=True)
            (docs / "index.md").write_text("# Docs\n", encoding="utf-8")

    monkeypatch.setattr(fetch_corpus, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(fetch_corpus.subprocess, "run", fake_run)

    fetch_corpus.fetch(source, force=False)

    assert calls[1][-4:] == ["--depth", "1", "origin", revision]
    assert calls[2][-2:] == ["--detach", "FETCH_HEAD"]
    assert (tmp_path / "raw" / "docs" / "index.md").read_text(encoding="utf-8") == "# Docs\n"
