# Step 21 — transcripts

## Task 1 — the dependency

```bash
$ uv add 'ragas>=0.4,<0.5'
```

`uv add` (uv 0.9.3) worked directly — no fallback to the hand-edit path was
needed. It resolved `ragas==0.4.3` and pulled in a large transitive tree,
dominated by the `langchain` v1 family (`langchain`, `langchain-core`,
`langchain-community`, `langchain-openai`, `langgraph` and friends),
`instructor`, `datasets`, `pandas`/`pyarrow`, and `scipy`/`scikit-network`.

Fixing Failure 1 below (the broken `import ragas`) required also pinning
`langchain-community>=0.3,<0.4` directly in `[project] dependencies` — see
that section for why. The tree below is the final, post-pin state:

```bash
$ uv tree --depth 1 --package ragas
Resolved 135 packages in 2ms
ragas v0.4.3
├── appdirs v1.4.4
├── datasets v5.0.1
├── diskcache v5.6.3
├── instructor v1.17.0
├── langchain v1.4.0
├── langchain-community v0.3.31
├── langchain-core v1.6.3
├── langchain-openai v1.6.2
├── nest-asyncio v1.6.0
├── networkx v3.6.1
├── numpy v2.5.3
├── openai v2.54.0
├── pillow v12.3.0
├── pydantic v2.13.5
├── rich v14.3.4
├── scikit-network v0.33.5
├── tiktoken v0.14.0
├── tqdm v4.70.1
└── typer v0.27.2
```

(The brief's exact invocation worked as written — resolution #2's fallback was
not needed. `langchain-community` moved from `0.4.2`, uv's first unconstrained
pick, to `0.3.31` once pinned; pinning it did **not** drag the rest of the
`langchain` v1 family — `langchain`, `langchain-core`, `langchain-openai` —
back to an older generation. uv resolved `langchain-community==0.3.31`
alongside `langchain-core==1.6.3` without conflict. `langchain-classic` and
`langchain-text-splitters` dropped out and `dataclasses-json`,
`marshmallow`, `typing-inspect` were added as `langchain-community`'s own
transitive dependencies changed shape.)

```bash
$ uv pip list | wc -l
134
```

That is 132 installed packages (2 header/separator lines in `uv pip list`'s
output). This is a much larger tree than the rest of the project's
dependencies combined (`cohere`, `fastapi`, `flashrank`, `openai`,
`pydantic-settings`, `qdrant-client`, `uvicorn` and their transitives). That is
a fact worth recording, not a reason to reopen the decision mid-step — per
resolution #2 of the task brief.

## Task 1 — the spike

### Failure 1 (not in the brief's table): `ModuleNotFoundError` on `import ragas`

The brief's Step 3 script, run verbatim, does not get as far as any of the
four failure modes the brief anticipated — it fails at the very first import:

```
$ uv run python scratch_ragas_spike.py
Traceback (most recent call last):
  File ".../scratch_ragas_spike.py", line 4, in <module>
    from ragas import SingleTurnSample
  File ".../ragas/__init__.py", line 5, in <module>
    from ragas.evaluation import aevaluate, evaluate
  File ".../ragas/evaluation.py", line 31, in <module>
    from ragas.llms import llm_factory
  File ".../ragas/llms/__init__.py", line 1, in <module>
    from ragas.llms.base import (
  File ".../ragas/llms/base.py", line 12, in <module>
    from langchain_community.chat_models.vertexai import ChatVertexAI
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

**What it means:** `ragas/llms/base.py` unconditionally does
`from langchain_community.chat_models.vertexai import ChatVertexAI` at module
scope, purely to offer a Vertex AI provider we never asked for. That submodule
was removed from `langchain-community` (moved to the separate
`langchain-google-vertexai` partner package) starting at `langchain-community`
0.4, and so is absent from the `langchain-community==0.4.2` that `ragas`'s own
(unconstrained) `Requires-Dist: langchain-community` resolves to by default.
This breaks `import ragas` — and therefore `llm_factory`, `embedding_factory`,
both metric APIs, everything — regardless of which of the two live APIs you
use. It is a confirmed upstream bug, not something specific to this project's
resolution: see ragas issues #2745, #2753, #2741, #2995 (a fix PR, #2837, was
open as of this writing but not released).

The controller independently verified that the bug is **not** ragas-version
specific — every `ragas` release from `0.3.6` through `0.4.2` fails the same
way — because the breakage lives in whichever `langchain-community` gets
resolved, not in `ragas`. Pinning `langchain-community` below `0.4` fixes
every one of those `ragas` versions cleanly, `ragas==0.4.3` included, and
without needing to downgrade `langchain`/`langchain-core`/`langchain-openai`
(see the dependency section above — they resolve fine against
`langchain-community==0.3.31`).

**The fix that works:** pin the transitive dependency in `pyproject.toml`,
not in application code:

```toml
# Never imported directly: pinned only because ragas 0.4.3 unconditionally
# imports the removed `langchain_community.chat_models.vertexai` module in
# ragas/llms/base.py, and langchain-community 0.4 dropped that module.
# langchain-community<0.4 still ships it, so `import ragas` works. See
# ragas issues #2745, #2753, #2995. Remove this pin once ragas stops
# importing that module (or drops the unconditional Vertex AI import).
"langchain-community>=0.3,<0.4",
```

followed by `uv lock && uv sync`. Verified with no stub or other workaround
in Python:

```
$ uv run python -c "import ragas; from ragas.metrics.collections import Faithfulness; print('ok', ragas.__version__)"
ok 0.4.3
```

### What did not get used, and why: a `sys.modules` stub

The first fix attempted here — before the controller's dependency-pin
correction — was to stub the missing submodule in `sys.modules` before the
first `ragas` import, since Vertex AI is never touched by this project:

```python
import sys
import types

_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _stub
```

This worked (it satisfies the import unconditionally), but it is the wrong
fix and **was rejected**: a `sys.modules` stub is application code that
fabricates a fake module at import time, must run before any `ragas` import
in every process that imports it (constraining import order in a file mypy
checks strictly), and would fail in a confusing way if `ragas` ever changes
what it imports elsewhere. A missing/incompatible transitive dependency is a
problem for the dependency file, not for `app/`. Do not resurrect this stub
in `app/evaluation/judge.py` or anywhere else — use the pin above instead.

### Failure 2 (not in the brief's table): legacy `ResponseRelevancy` rejects the modern embeddings object

Independently of Failure 1, the legacy `ragas.metrics` classes and
`single_turn_ascore` do not work uniformly across all three metrics. Each
legacy import emits a `DeprecationWarning` pointing at
`ragas.metrics.collections` (as resolution #5 anticipated), and legacy
`Faithfulness` scored fine via `single_turn_ascore` (`1.0`). The legacy
`ResponseRelevancy` (`ragas.metrics._answer_relevance`), however, fails:

```
AttributeError: 'OpenAIEmbeddings' object has no attribute 'embed_query'. Did you mean: 'embed_text'?
```

**What it means:** `embedding_factory("openai", model=..., client=client)`
auto-detects "modern" mode whenever a `client` is passed (see
`_is_legacy_embedding_call` in `ragas/embeddings/base.py`) and returns a
`BaseRagasEmbedding` (new interface: `embed_text`/`aembed_text`). The legacy
`ResponseRelevancy`/`AnswerRelevancy` metric class was never updated for that
interface — it still calls `.embed_query()`, which only the old
`BaseRagasEmbeddings` (plural) interface has. `Faithfulness` and
`LLMContextPrecisionWithoutReference` don't touch embeddings at all, so they
never hit this; only the embeddings-consuming metric does. There is no
`embedding_factory` call that produces an object both the legacy
`ResponseRelevancy` and the brief's `client=` construction are happy with
simultaneously — the legacy branch of `embedding_factory` ignores `client`
entirely and reads credentials from the environment instead, which is a
different (and less explicit) code path than the brief specifies.

**The fix that works:** exactly resolution #5's stated fallback — since the
legacy path does not work uniformly across all three metrics, all three move
to the `ragas.metrics.collections` API. Two of the three names differ from
what the brief (and much of ragas's own documentation) assumes:

- `LLMContextPrecisionWithoutReference` does not exist in
  `ragas.metrics.collections` — it is `ContextPrecisionWithoutReference`
  there (no `LLM` prefix).
- `ResponseRelevancy` does not exist in `ragas.metrics.collections` either —
  it is `AnswerRelevancy` there (the brief's table anticipated this rename in
  the other direction, for the legacy module; it turned out to apply to the
  collections module instead).
- `Faithfulness` keeps its name in both modules.

**The three collections classes do not share one call signature** — this is
the single most important thing this task found, because it changes Task 2's
`METRICS` shape from "construct three, call one uniform method" to "each
entry owns its own call":

- `Faithfulness.ascore` and `ContextPrecisionWithoutReference.ascore` both
  take `(user_input, response, retrieved_contexts)`.
- `AnswerRelevancy.ascore` takes only `(user_input, response)` — it has no
  context-relevance component, so it never asks for `retrieved_contexts`.

### The real run, collections API, three metrics, with the pin and no stub

```
$ uv run python scratch_ragas_spike.py
.../scratch_ragas_spike.py:23: DeprecationWarning: Importing embedding_factory from ragas.embeddings is deprecated. Import directly from ragas.embeddings.base or use modern providers: from ragas.embeddings import OpenAIEmbeddings, GoogleEmbeddings, HuggingFaceEmbeddings
  embeddings = embedding_factory("openai", model=settings.embedding_model, client=client)
faithfulness 1.0
relevancy 0.2783502074325034
context_precision 0.9999999999
```

Faithfulness and context precision land where the brief expected (both ~1.0
on this deliberately easy, fully-supported sample). Relevancy came in at
`0.278` — below the brief's "comfortably above 0.5" expectation, and
reproduced (`0.2784325473866333` in an earlier run, `0.2783502074325034`
here — stable to four significant figures across runs). See the next section
for why: this is real, documented metric behaviour, not an integration bug.

One more benign warning, also not in the brief's table: importing
`embedding_factory` from the `ragas.embeddings` package root (as the brief's
script and this spike both do) emits its own `DeprecationWarning`
recommending `ragas.embeddings.base` or a named provider class instead. It
does not affect behavior; recorded per resolution #5's spirit (uniform,
working, and noisy beats broken).

## Task 1 — relevancy is sensitive to answer completeness

The controller independently probed this with a four-way comparison — same
question and retrieved context as the spike above, judged by `gpt-4o`, only
the response varied:

```
good-terse     relevancy 0.278   faithfulness 1.000
good-full      relevancy 0.992   faithfulness 0.600
refusal        relevancy 0.000   faithfulness 0.000
fabrication    relevancy 0.513   faithfulness 0.000
```

Relevancy penalises terse answers: the metric works by having an LLM
regenerate candidate questions from the *response alone* and comparing their
embeddings to the original question, so a one-clause answer gives the
regenerator too little to reconstruct the original question from, and cosine
similarity suffers even when the answer is fully correct and faithful
(`good-terse`: relevancy 0.278, faithfulness 1.000). A fuller answer that
restates the question's subject scores far higher on relevancy even though
its faithfulness is lower (`good-full`: relevancy 0.992, faithfulness 0.600).
The metric still discriminates correctly at the extremes — a refusal scores
0.000 on both.

**Consequence for later steps:** relevancy moves with answer *length* as well
as answer *quality*. A step that changes how much context the model gets
(and therefore how much it writes) can move relevancy for that reason alone,
independent of any real quality change — a rise or fall in relevancy should
be read alongside answer length, not as a standalone quality signal.

## The four working lines — Task 2 copies these verbatim

The legacy `single_turn_ascore` path does **not** work uniformly across all
three metrics (Failure 2 above), so Task 2 must use the collections API for
all three, not the brief's original legacy assumption. No `sys.modules` stub
or other import-time workaround is needed — the `langchain-community` pin in
`pyproject.toml` (Failure 1 above) is what makes `import ragas` work at all.

```python
from ragas.metrics.collections import AnswerRelevancy as ResponseRelevancy
from ragas.metrics.collections import (
    ContextPrecisionWithoutReference as LLMContextPrecisionWithoutReference,
)
from ragas.metrics.collections import Faithfulness

llm = llm_factory("gpt-4o", client=client)
embeddings = embedding_factory("openai", model=settings.embedding_model, client=client)

faithfulness_result = await Faithfulness(llm=llm).ascore(
    user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
)
relevancy_result = await ResponseRelevancy(llm=llm, embeddings=embeddings).ascore(
    user_input=user_input, response=response
)
context_precision_result = await LLMContextPrecisionWithoutReference(llm=llm).ascore(
    user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
)
# each result is a MetricResult; the float score is result.value
```

`llm_factory` and `embedding_factory` keep exactly the signatures the brief
assumed — only the metric import path, class names, and the non-uniform
`ascore` calls differ from the brief's draft. Remember the signature split:
`Faithfulness` and `LLMContextPrecisionWithoutReference` both need
`retrieved_contexts`; `ResponseRelevancy` does not accept it.
