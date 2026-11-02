"""Synthetic training data (DA-11): a Bedrock model writes SFT examples for you.

Two modes, matching SageMaker's serverless data generation (whose API isn't public):

- **contextual**: pass ``documents``; each example's question is answerable from one chunk
  of them and its answer is grounded in that chunk (domain Q&A from manuals, policies, etc.).
- **non-contextual**: pass a ``task`` description and, optionally, a few seed ``examples``
  to copy the style of.

The generator answers with structured output (a JSON list of question/answer pairs per
call), questions are de-duplicated, and the result is cached in S3 by (model, inputs,
settings), so re-running doesn't pay twice. The cost bound (every call uses ``max_tokens``)
is checked against ``max_cost_usd`` before any call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from skyhammer.errors import CostLimitExceededError, SkyhammerError
from skyhammer.formats import message_text
from skyhammer.validation import estimate_tokens

if TYPE_CHECKING:
    from skyhammer.dataset import Dataset
    from skyhammer.llm import LLM

log = logging.getLogger("skyhammer")

CHUNK_TOKENS = 1500  # document text per contextual call
MAX_EXAMPLES = 20_000

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "examples",
        "schema": {
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                        },
                        "required": ["question", "answer"],
                    },
                }
            },
            "required": ["examples"],
        },
    },
}


def chunk(text: str, tokens: int = CHUNK_TOKENS) -> list[str]:
    """Split on paragraph breaks into pieces of about ``tokens`` tokens each."""
    out: list[str] = []
    current: list[str] = []
    size = 0
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        n = estimate_tokens(para)
        if current and size + n > tokens:
            out.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += n
    if current:
        out.append("\n\n".join(current))
    return out


def instructions(
    per_call: int,
    *,
    task: str | None,
    context: str | None,
    seeds: list[tuple[str, str]],
    batch_index: int,
) -> str:
    parts = [
        f"Write {per_call} new, varied training examples for fine-tuning an assistant. "
        "Each example is a realistic user question and the ideal assistant answer."
    ]
    if task:
        parts.append(f"Task: {task}")
    if context is not None:
        parts.append(
            "Every question must be answerable from the source text below, and every answer "
            "must rely only on it (no outside facts). Cover different parts of the text.\n"
            f"<source>\n{context}\n</source>"
        )
    if seeds:
        shown = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in seeds)
        parts.append(
            f"Match the style and length of these examples, without copying them:\n{shown}"
        )
    # Different batches over the same input would otherwise repeat each other.
    parts.append(f"This is batch {batch_index + 1}; vary topics, phrasing and difficulty.")
    return "\n\n".join(parts)


def _seed_pairs(examples: Dataset | None, limit: int = 5) -> list[tuple[str, str]]:
    if examples is None:
        return []
    pairs = []
    for _, r in examples._rows[:limit]:
        msgs = r.get("messages") or []
        user = next((m for m in msgs if m.get("role") == "user"), None)
        bot = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
        if user and bot:
            pairs.append((message_text(user.get("content")), message_text(bot.get("content"))))
    return pairs


def _documents(documents: Dataset | list[str] | str | None) -> list[str]:
    if documents is None:
        return []
    if isinstance(documents, str):
        docs = [documents]
    elif isinstance(documents, list):
        docs = documents
    else:
        docs = [str(r.get("text", "")) for _, r in documents._rows]
    return [c for d in docs for c in chunk(d)]


def _key(model_id: str, payload: Any) -> str:
    return hashlib.sha256(json.dumps([model_id, payload], sort_keys=True).encode()).hexdigest()


def plan_calls(n: int, per_call: int, chunks: list[str]) -> list[str | None]:
    """The context for each call (None when non-contextual). With more chunks than calls,
    the calls spread evenly over the documents; with fewer, chunks repeat round-robin."""
    calls = math.ceil(n / per_call)
    if not chunks:
        return [None] * calls
    if len(chunks) > calls:
        return [chunks[i * len(chunks) // calls] for i in range(calls)]
    return [chunks[i % len(chunks)] for i in range(calls)]


def estimate_cost(generator: LLM, prompts: list[str], max_tokens: int) -> float | None:
    """Upper bound: every call returns ``max_tokens``. None when the price is unknown."""
    from skyhammer.pricing import InferencePrices

    try:
        prices = InferencePrices.for_session(generator.session)
        name = generator.info.name
    except Exception:
        return None
    tokens_in = sum(estimate_tokens(p) for p in prompts)
    return prices.cost(name, tokens_in, len(prompts) * max_tokens)


def _normal(q: str) -> str:
    return " ".join(re.findall(r"\w+", q.lower()))


def synthesize(
    generator: LLM,
    *,
    n: int,
    task: str | None = None,
    documents: Dataset | list[str] | str | None = None,
    examples: Dataset | None = None,
    system: str | None = None,
    per_call: int = 5,
    max_tokens: int = 4096,
    temperature: float = 0.9,
    max_cost_usd: float | None = None,
    max_workers: int = 8,
    name: str = "synthetic",
) -> Dataset:
    """Generate about ``n`` SFT conversations (fewer if the model repeats itself)."""
    from skyhammer.dataset import Dataset

    if not 1 <= n <= MAX_EXAMPLES:
        raise SkyhammerError(f"n must be between 1 and {MAX_EXAMPLES:,}")
    if not 1 <= per_call <= 20:
        raise SkyhammerError("per_call must be between 1 and 20")
    chunks = _documents(documents)
    if not task and not chunks:
        raise SkyhammerError("Pass a task description, documents, or both")
    seeds = _seed_pairs(examples)
    contexts = plan_calls(n, per_call, chunks)
    prompts = [
        instructions(per_call, task=task, context=c, seeds=seeds, batch_index=i)
        for i, c in enumerate(contexts)
    ]
    bound = estimate_cost(generator, prompts, max_tokens)
    log.info(
        "Generating ~%d examples with %s in %d calls (at most %s)",
        n,
        generator.name,
        len(prompts),
        f"${bound:,.2f}" if bound is not None else "unknown cost",
    )
    if max_cost_usd is not None:
        if bound is None:
            raise SkyhammerError(f"No price for {generator.name}, so max_cost_usd can't be checked")
        if bound > max_cost_usd:
            raise CostLimitExceededError(
                bound, max_cost_usd, hint="Lower n or max_tokens, or raise max_cost_usd."
            )

    params = {"n": n, "per_call": per_call, "max_tokens": max_tokens, "t": temperature}
    key = _key(generator.model_id, [prompts, system, params])
    session = generator.session
    res = session.resources
    s3 = session.client("s3")
    cache = f"synthetic/{key[:32]}.jsonl"
    try:
        body = s3.get_object(Bucket=res.bucket, Key=cache)["Body"].read()
        log.info("Using cached synthetic data s3://%s/%s", res.bucket, cache)
        return Dataset.from_records([json.loads(line) for line in body.splitlines()], name)
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise

    def generate(prompt: str) -> list[dict[str, str]]:
        try:
            resp = generator.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                response_format=_SCHEMA,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            parsed = resp.choices[0].message.parsed()
        except (SkyhammerError, ValueError) as e:
            log.warning("generation call failed, skipping it: %s", e)
            return []
        items = parsed.get("examples", []) if isinstance(parsed, dict) else []
        return [i for i in items if isinstance(i, dict)]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        batches = list(pool.map(generate, prompts))

    seen: set[str] = {_normal(q) for q, _ in seeds}
    out: list[dict[str, Any]] = []
    for item in (i for b in batches for i in b):
        q, a = str(item.get("question", "")).strip(), str(item.get("answer", "")).strip()
        if not q or not a or _normal(q) in seen:
            continue
        seen.add(_normal(q))
        messages = [{"role": "system", "content": system}] if system else []
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        out.append({"messages": messages})
        if len(out) >= n:
            break
    if not out:
        raise SkyhammerError(f"{generator.name} produced no usable examples")
    if len(out) < n:
        log.warning(
            "Generated %d of %d examples (duplicates and failed calls dropped)", len(out), n
        )
    extra = (
        {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": res.kms_key_arn}
        if res.kms_key_arn
        else {}
    )
    s3.put_object(
        Bucket=res.bucket,
        Key=cache,
        Body=b"".join(json.dumps(r).encode() + b"\n" for r in out),
        **extra,
    )
    return Dataset.from_records(out, name)
