"""M2 exit criteria, live: serverless calls across providers (fractions of a cent total)."""

from __future__ import annotations

import base64
import math
import time
from collections.abc import Iterator
from pathlib import Path

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


@pytest.fixture
def deny_topic_guardrail(live_session: Session, unique_name: str) -> Iterator[str]:
    """A guardrail that denies investment advice; deleted afterwards."""
    bedrock = live_session.client("bedrock")
    gid: str = bedrock.create_guardrail(
        name=unique_name,
        topicPolicyConfig={
            "topicsConfig": [
                {
                    "name": "Investment advice",
                    "type": "DENY",
                    "definition": "Recommendations about which stocks or crypto to buy or sell.",
                    "examples": ["Which stocks should I buy?"],
                }
            ]
        },
        blockedInputMessaging="BLOCKED",
        blockedOutputsMessaging="BLOCKED",
        tags=[{"key": "skyhammer:name", "value": unique_name}],
    )["guardrailId"]
    try:
        while bedrock.get_guardrail(guardrailIdentifier=gid)["status"] != "READY":
            time.sleep(2)
        yield gid
    finally:
        bedrock.delete_guardrail(guardrailIdentifier=gid)


def test_guardrail_blocks_on_call_and_stream(
    live_session: Session, deny_topic_guardrail: str
) -> None:
    llm = LLM("nova-micro", session=live_session)
    guardrail = {"id": deny_topic_guardrail, "version": "DRAFT", "trace": True}
    ok = llm.chat.completions.create(
        messages=[{"role": "user", "content": "What is 2+2?"}], max_tokens=20, guardrail=guardrail
    )
    assert ok.choices[0].finish_reason != "content_filter"
    blocked = llm.chat.completions.create(
        messages=[{"role": "user", "content": "Which tech stocks should I buy?"}],
        guardrail=guardrail,
    )
    assert blocked.choices[0].finish_reason == "content_filter"
    assert blocked.choices[0].message.content == "BLOCKED"
    assert blocked.raw["trace"]["guardrail"]["inputAssessment"]
    stream = llm.chat.completions.create(
        messages=[{"role": "user", "content": "Should I buy bitcoin?"}],
        guardrail=guardrail,
        stream=True,
    )
    final = stream.final()
    assert final.choices[0].finish_reason == "content_filter"
    assert final.choices[0].message.content == "BLOCKED"


def test_video_inline_and_from_s3(live_session: Session, unique_name: str) -> None:
    clip = (Path(__file__).parents[2] / "examples" / "data" / "red.mp4").read_bytes()
    llm = LLM("nova-lite", session=live_session)
    ask = {"type": "text", "text": "What color fills this video? One word."}
    inline = {
        "type": "video_url",
        "video_url": "data:video/mp4;base64," + base64.b64encode(clip).decode(),
    }
    r = llm.chat.completions.create(
        messages=[{"role": "user", "content": [ask, inline]}], max_tokens=60
    )
    assert "red" in (r.choices[0].message.content or "").lower()
    bucket = live_session.resources_for(live_session.region).bucket
    key = f"integration/{unique_name}/red.mp4"
    s3 = live_session.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=clip)
    try:
        part = {"type": "video_url", "video_url": f"s3://{bucket}/{key}"}
        r = llm.chat.completions.create(
            messages=[{"role": "user", "content": [ask, part]}], max_tokens=60
        )
        assert "red" in (r.choices[0].message.content or "").lower()
    finally:
        s3.delete_object(Bucket=bucket, Key=key)
