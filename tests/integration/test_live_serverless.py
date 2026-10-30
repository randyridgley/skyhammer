"""M2 exit criteria, live: serverless calls across providers (fractions of a cent total)."""

from __future__ import annotations

import math

import pytest

from skyhammer import LLM, generate_api_key, openai_client
from skyhammer.session import Session

PROVIDERS = [
    "nova-micro",
    "claude-haiku-4-5",
    "llama3-3-70b-instruct",
    "qwen3-32b",
    "ministral-3-8b-instruct",
    "gpt-oss-20b",
]
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@pytest.mark.parametrize("name", PROVIDERS)
def test_chat_stream_tools_json(live_session: Session, name: str) -> None:
    llm = LLM(name, session=live_session)
    r = llm.chat.completions.create(
        messages=[{"role": "user", "content": "Say ok"}], max_tokens=300
    )
    assert (r.choices[0].message.content or "").strip()
    stream = llm.chat.completions.create(
        messages=[{"role": "user", "content": "Count 1 to 3"}], stream=True, max_tokens=300
    )
    chunks = list(stream)
    assert len(chunks) > 1 and stream.final().usage.total_tokens > 0
    t = llm.chat.completions.create(
        messages=[{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        tools=[WEATHER],
        max_tokens=300,
    )
    calls = t.choices[0].message.tool_calls
    assert calls and calls[0].function.name == "get_weather", t.choices[0].message.content
    j = llm.chat.completions.create(
        messages=[{"role": "user", "content": "Which city is the capital of France?"}],
        response_format=SCHEMA,
        max_tokens=400,
    )
    assert "paris" in str(j.choices[0].message.parsed().get("city", "")).lower()


def test_costs_and_spend(live_session: Session) -> None:
    before = live_session.spend.total_usd
    r = LLM("nova-micro", session=live_session).chat.completions.create(
        messages=[{"role": "user", "content": "Say ok"}], max_tokens=5
    )
    assert r.usage.cost_usd and r.usage.cost_usd > 0
    assert live_session.spend.total_usd > before


def test_embeddings_and_rerank(live_session: Session) -> None:
    docs = ["Berlin is the capital of Germany.", "Paris is the capital of France."]
    for name in ("titan-embed-text-v2", "embed-v4"):
        llm = LLM(name, session=live_session)
        d = llm.embeddings.create(input=docs, dimensions=256)
        q = (
            llm.embeddings.create(
                input="capital of France", input_type="search_query", dimensions=256
            )
            .data[0]
            .embedding
        )

        def cos(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b, strict=True)) / (
                math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
            )

        assert cos(q, d.data[1].embedding) > cos(q, d.data[0].embedding), name
    r = LLM("rerank-v3-5", session=live_session).rerank("capital of France", docs)
    assert r.results[0].index == 1


def test_api_key_and_openai_client(live_session: Session) -> None:
    client = openai_client(session=live_session, api_key=generate_api_key(live_session, hours=1))
    r = client.chat.completions.create(
        model="openai.gpt-oss-20b-1:0",
        max_tokens=200,
        messages=[{"role": "user", "content": "Say ok"}],
    )
    assert r.choices[0].message.content
