"""Distillation for students without Bedrock-native distillation (open-weight models).

Two steps: the teacher answers every prompt (live calls by default; a 50%-off batch job
from ``BATCH_FROM`` prompts, since batch jobs can queue for an hour or more), then the
student is fine-tuned (SFT) on those answers. Teacher
answers are cached in S3 by (teacher, prompts, settings), so re-running reattaches to the
same SFT job instead of paying for generation again.
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from skyhammer.batch import MIN_RECORDS
from skyhammer.dataset import Dataset
from skyhammer.errors import SkyhammerError
from skyhammer.formats import message_text
from skyhammer.validation import estimate_tokens

if TYPE_CHECKING:
    from skyhammer.llm import LLM

log = logging.getLogger("skyhammer")

# Batch saves 50% but can wait in Bedrock's queue for over an hour (seen live: 73+ min for
# 100 prompts); at small sizes the saving is cents, so stay live below this.
BATCH_FROM = 1000


def prompt_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The prompt part of a record: its messages without a trailing assistant turn."""
    messages = list(record.get("messages") or [])
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    if not messages or messages[-1].get("role") != "user":
        raise SkyhammerError("Each distillation prompt must end with a user message")
    return messages


def _cache_key(teacher_id: str, prompts: list[list[dict[str, Any]]], params: dict[str, Any]) -> str:
    blob = json.dumps([teacher_id, prompts, params], sort_keys=True, default=repr)
    return hashlib.sha256(blob.encode()).hexdigest()


def estimate_teacher_cost(
    teacher: LLM, prompts: list[list[dict[str, Any]]], max_tokens: int, *, batch: bool
) -> float | None:
    """Upper bound: every answer uses ``max_tokens``. None when the price is unknown."""
    from skyhammer.pricing import InferencePrices

    try:
        name = teacher.info.name
        prices = InferencePrices.for_session(teacher.session)
    except Exception:
        return None
    tokens_in = sum(estimate_tokens(json.dumps(p)) for p in prompts)
    return prices.cost(
        name, tokens_in, len(prompts) * max_tokens, tier="batch" if batch else "standard"
    )


def teacher_dataset(
    teacher: LLM,
    prompts: Dataset,
    *,
    max_tokens: int = 1024,
    temperature: float | None = None,
    use_batch: bool | None = None,
    max_workers: int = 8,
    poll_seconds: float = 60,
) -> Dataset:
    """Fill in the teacher's answer for every prompt. Records that fail are dropped."""
    session = teacher.session
    records = [r for _, r in prompts._rows]
    convs = [prompt_messages(r) for r in records]
    params: dict[str, Any] = {"max_tokens": max_tokens}
    if temperature is not None:
        params["temperature"] = temperature
    batch = len(convs) >= BATCH_FROM if use_batch is None else use_batch
    if batch and len(convs) < MIN_RECORDS:
        raise SkyhammerError(f"Batch needs at least {MIN_RECORDS} prompts; use live calls")
    key = _cache_key(teacher.model_id, convs, params)
    name = f"{prompts.name}-teacher"
    res = session.resources
    s3 = session.client("s3")
    cache = f"distill/{key[:32]}.jsonl"

    try:
        body = s3.get_object(Bucket=res.bucket, Key=cache)["Body"].read()
        log.info("Using cached teacher answers s3://%s/%s", res.bucket, cache)
        return Dataset.from_records([json.loads(line) for line in body.splitlines()], name)
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise

    if batch:
        log.info("Generating %d teacher answers with %s (batch)", len(convs), teacher.model_id)
        job = teacher.batch(
            [{"custom_id": f"{i:08d}", "messages": m} for i, m in enumerate(convs)],
            name=f"skyhammer-distill-{key[:12]}",
            **params,
        )
        job.wait(poll_seconds=poll_seconds)
        answers: list[str | None] = [None] * len(convs)
        for r in job.results():
            if r.completion:
                answers[int(r.custom_id)] = message_text(r.completion.choices[0].message.content)
    else:
        log.info("Generating %d teacher answers with %s (live)", len(convs), teacher.model_id)

        def ask(messages: list[dict[str, Any]]) -> str | None:
            try:
                resp = teacher.chat.completions.create(messages=messages, **params)
            except SkyhammerError as e:
                log.warning("teacher call failed, dropping prompt: %s", e)
                return None
            return message_text(resp.choices[0].message.content)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            answers = list(pool.map(ask, convs))

    out = [
        {**rec, "messages": [*conv, {"role": "assistant", "content": a.strip()}]}
        for rec, conv, a in zip(records, convs, answers, strict=True)
        if a and a.strip()
    ]
    if dropped := len(convs) - len(out):
        log.warning("Dropped %d of %d prompts the teacher didn't answer", dropped, len(convs))
    if not out:
        raise SkyhammerError("The teacher produced no answers; nothing to train on")
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
