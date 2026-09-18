"""Drop retrieved chunks that carry instructions aimed at the model.

Pure: no I/O, no model. It runs after compression and before build_context, so
it scans exactly the text the model would receive.

Deliberately narrow. FastAPI's documentation is written in the imperative —
"run this", "ignore the previous section if…" — so every pattern here must also
survive a scan of all 1 484 corpus chunks with zero hits (step 22, Rule D step 1).
A paraphrased injection gets through by design; step 22 measures how often.
"""

import re
from collections.abc import Sequence

from app.models.chunks import ScoredChunk

# ponytail: three regexes, English only. A paraphrase with none of these words
# passes; step 22's fixture measures that half separately. Upgrade to a
# classifier only if a measured paraphrase rate justifies a call per chunk.
INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "override": re.compile(
        r"\b(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)"
        r"\s+(instructions|rules|prompts?)\b",
        re.IGNORECASE,
    ),
    "prompt_probe": re.compile(
        r"\b(reveal|print|repeat|show|output)\s+(your|the)\s+(full\s+)?"
        r"(system\s+prompt|instructions)\b",
        re.IGNORECASE,
    ),
    "role_marker": re.compile(r"^\s*(system|assistant)\s*:", re.IGNORECASE | re.MULTILINE),
}


def detect_injection(
    chunks: Sequence[ScoredChunk],
) -> tuple[list[ScoredChunk], list[str]]:
    """Kept chunks in their original order, and one warning per dropped chunk
    naming the first pattern it matched."""
    kept: list[ScoredChunk] = []
    warnings: list[str] = []
    for scored in chunks:
        hit = next(
            (
                name
                for name, pattern in INJECTION_PATTERNS.items()
                if pattern.search(scored.chunk.text)
            ),
            None,
        )
        if hit is None:
            kept.append(scored)
        else:
            warnings.append(f"injection_suspected: {scored.chunk.chunk_id} ({hit})")
    return kept, warnings
