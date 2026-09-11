"""The only module in the project that talks to a language model.

Kept to one function on purpose. Swapping OpenAI for Anthropic is editing
``complete``'s body — a different message shape behind the same signature — and
nothing else in the pipeline notices. That is the whole reason ``answer.py``
never sees a client.
"""

from typing import Any

from app.ingestion.embed import build_client

# v1 (step 08). Step 09 hardens the citation rules; keep the version marker so a
# prompt change and an evaluation number can be tied together.
SYSTEM_PROMPT = """\
You answer questions about technical documentation using only the provided context.
If the context does not contain the answer, say that you do not know.
Do not use prior knowledge. Do not invent APIs, flags or version numbers.
Cite the context entry for each factual statement using its bracketed number, e.g. [1].
Answer in the language of the question."""

# The corpus is English and the questions are often French. Without the last
# line above, every answer comes back in English and the system looks broken.

USER_TEMPLATE = """Context:
{context}

Question: {question}"""


def complete(
    system: str,
    user: str,
    *,
    model: str,
    client: Any | None = None,
    temperature: float = 0.0,
) -> tuple[str, dict[str, int]]:
    """Return the model's text and its token usage.

    ``temperature=0`` by default: a pipeline that answers differently on a second
    identical call cannot be evaluated, and steps 11 and 21 are evaluations.

    Usage is returned from the very first version because step 24 needs it and
    adding it later means touching every call site.
    """
    # Same factory as the embeddings: one credential, and one place the SDK's
    # retry policy (max_retries=5, exponential, 429 and 5xx) is configured.
    client = client or build_client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    # content is Optional in the SDK: a refusal or a length stop can return None,
    # and "" reaching the Answer model fails there loudly rather than here quietly.
    text = response.choices[0].message.content or ""
    usage = response.usage
    counts = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }
    return text, counts
