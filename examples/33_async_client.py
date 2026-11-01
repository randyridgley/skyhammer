"""Async client: fan out many calls from asyncio code with a concurrency limit.

``AsyncLLM`` takes the same names as ``LLM`` (a base model, your fine-tune, or an
environment like ``support@prod``, splits included) and the same calls, awaited:
``chat.completions.create``, ``embeddings.create``, ``rerank`` and ``responses.create``.

``asyncio.gather`` over hundreds of prompts is safe: at most ``max_concurrency`` requests
are in flight at once and the rest wait their turn, so a burst doesn't trip Bedrock's
request quotas. Cost tracking, ``max_spend_usd``, guardrails and retries work as in ``LLM``.

Cost: under $0.01 (about 9 short Nova Micro calls).
Run: uv run python examples/33_async_client.py
"""

import asyncio
import time

from skyhammer import AsyncLLM

QUESTIONS = [f"What is {n} squared?" for n in range(2, 10)]
SYSTEM = {"role": "system", "content": "Reply with only the number."}


async def main() -> None:
    async with AsyncLLM("nova-micro", max_concurrency=4) as llm:
        started = time.monotonic()
        replies = await asyncio.gather(
            *(
                llm.chat.completions.create(
                    messages=[SYSTEM, {"role": "user", "content": q}], max_tokens=10
                )
                for q in QUESTIONS
            )
        )
        for q, r in zip(QUESTIONS, replies, strict=True):
            print(f"{q:22} {r.choices[0].message.content.strip()}")
        print(f"{len(replies)} calls in {time.monotonic() - started:.1f}s, 4 at a time")

        # Streaming: iterate with ``async for``.
        stream = await llm.chat.completions.create(
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=60,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices:
                print(chunk.choices[0].delta.content or "", end="", flush=True)
        print()


asyncio.run(main())
