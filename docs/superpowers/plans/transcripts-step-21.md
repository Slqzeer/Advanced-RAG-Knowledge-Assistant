# Step 21 — transcripts

## Task 1 — the dependency

```bash
$ uv add 'ragas>=0.4,<0.5'
```

`uv add` (uv 0.9.3) worked directly — no fallback to the hand-edit path was
needed. It resolved `ragas==0.4.3` and pulled in a large transitive tree,
dominated by the `langchain` v1 family (`langchain`, `langchain-core`,
`langchain-community`, `langchain-openai`, `langchain-classic`, `langgraph`
and friends), `instructor`, `datasets`, `pandas`/`pyarrow`, and `scipy`/
`scikit-network`.

```bash
$ uv tree --depth 1 --package ragas
Resolved 134 packages in 1ms
ragas v0.4.3
├── appdirs v1.4.4
├── datasets v5.0.1
├── diskcache v5.6.3
├── instructor v1.17.0
├── langchain v1.4.0
├── langchain-community v0.4.2
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
not needed.)

```bash
$ uv pip list | wc -l
133
```

That is 131 installed packages (2 header/separator lines in `uv pip list`'s
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
`langchain-google-vertexai` partner package) and does not exist in the
`langchain-community==0.4.2` that `ragas`'s own (unconstrained)
`Requires-Dist: langchain-community` resolves to today. This breaks
`import ragas` — and therefore `llm_factory`, `embedding_factory`, both metric
APIs, everything — regardless of which of the two live APIs you use. It is a
confirmed, currently-unfixed upstream bug: see ragas issues #2745, #2753,
#2741, #2995 (a fix PR, #2837, was open as of this writing but not released;
`ragas==0.4.3` is still the latest on PyPI). The only workaround the upstream
thread names is downgrading to `ragas<0.4` (`ragas==0.3.9`), which throws away
the very API this step exists to confirm.

Downgrading `langchain-community` alone was tried and rejected: `ragas` also
depends on `langchain==1.4.0` / `langchain-core==1.6.3` (the new v1 langchain
generation), and any `langchain-community` version old enough to still ship
`chat_models/vertexai.py` predates that generation, so pinning it drags the
whole `langchain`/`langchain-core`/`langchain-openai`/`langgraph` stack back
with it — a much bigger, riskier change than the bug it fixes, for a class we
never use.

**The fix that works:** since Vertex AI is never touched by this project,
stub the missing submodule in `sys.modules` *before* the first `ragas` import.
This satisfies the import unconditionally and costs nothing at runtime:

```python
import sys
import types

_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _stub
```

This must run before `import ragas` (or any `from ragas...` import) anywhere
in the process — Task 2's `app/evaluation/judge.py` needs it at module top,
same as the spike.

### Failure 2 (not in the brief's table): legacy `ResponseRelevancy` rejects the modern embeddings object

With the stub above, `from ragas import SingleTurnSample` and the legacy
`ragas.metrics` imports succeed (each with a `DeprecationWarning` pointing at
`ragas.metrics.collections`, as resolution #5 anticipated). `Faithfulness`
scored fine via `single_turn_ascore` (`1.0`). The legacy `ResponseRelevancy`
(`ragas.metrics._answer_relevance`), however, fails:

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

The three collections classes also do not share one call signature:
`Faithfulness.ascore` and `ContextPrecisionWithoutReference.ascore` both take
`(user_input, response, retrieved_contexts)`; `AnswerRelevancy.ascore` takes
only `(user_input, response)` — it has no context-relevance component, so it
never asks for `retrieved_contexts`.

### The real run, collections API, three metrics

```
$ uv run python scratch_ragas_spike.py
.../scratch_ragas_spike.py:36: DeprecationWarning: Importing embedding_factory from ragas.embeddings is deprecated. Import directly from ragas.embeddings.base or use modern providers: from ragas.embeddings import OpenAIEmbeddings, GoogleEmbeddings, HuggingFaceEmbeddings
  embeddings = embedding_factory("openai", model=settings.embedding_model, client=client)
faithfulness 1.0
relevancy 0.2784325473866333
context_precision 0.9999999999
```

Faithfulness and context precision land where the brief expected (both ~1.0
on this deliberately easy, fully-supported sample). Relevancy came in at
`0.278` — below the brief's "comfortably above 0.5" expectation. This is not
an integration bug: faithfulness and context precision confirm the plumbing
works, and relevancy is a real, if unflattering, score for this response. The
answer never repeats the words "FastAPI" or "dependency" (it just says
"Depends() in the path operation signature"), and the metric works by having
an LLM regenerate candidate questions from the *response alone* and comparing
their embeddings to the original question — a terse, context-free answer
gives the regenerator less to reconstruct the original question from, and the
cosine similarity suffers accordingly. Task 2 should not assume real answers
will clear 0.5 on this metric without the response restating the topic; that
is a property of the metric, not a bug to fix.

One more benign warning, also not in the brief's table: importing
`embedding_factory` from the `ragas.embeddings` package root (as the brief's
script and this spike both do) emits its own `DeprecationWarning`
recommending `ragas.embeddings.base` or a named provider class instead. It
does not affect behavior; recorded per resolution #5's spirit (uniform,
working, and noisy beats broken).

## The four working lines — Task 2 copies these verbatim

The legacy `single_turn_ascore` path does **not** work uniformly across all
three metrics (Failure 2 above), so Task 2 must use the collections API for
all three, not the brief's original legacy assumption. The `sys.modules`
stub from Failure 1 must run before any `ragas` import, in the same module.

```python
import sys
import types

# Workaround: ragas 0.4.3's ragas/llms/base.py unconditionally imports
# ChatVertexAI from a langchain-community submodule that no longer exists.
# We never use Vertex AI; a stub module satisfies the import. Must run
# before any `ragas` import. See ragas issues #2745/#2753/#2995.
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _stub

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
assumed — only the metric import path, class names, scoring call, and the
pre-import stub differ from the brief's draft.
