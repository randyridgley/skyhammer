"""Tools, structured outputs, reasoning, service tiers and streaming over Converse."""

import base64
import json
from typing import Any

import boto3
import pytest
from botocore.validate import validate_parameters

from skyhammer.inference import (
    Stream,
    _ThinkSplitter,
    converse_stream_chunks,
    from_converse_response,
    split_reasoning,
    to_converse_request,
)

RT = boto3.client("bedrock-runtime", region_name="us-east-1")
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def valid(req: dict[str, Any], op: str = "Converse") -> dict[str, Any]:
    validate_parameters(req, RT.meta.service_model.operation_model(op).input_shape)
    return req


def test_tools_and_tool_choice_mapping() -> None:
    req = valid(
        to_converse_request(
            "m", [{"role": "user", "content": "Paris?"}], tools=[WEATHER], tool_choice="required"
        )
    )
    spec = req["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "get_weather" and spec["inputSchema"]["json"]["required"] == ["city"]
    assert req["toolConfig"]["toolChoice"] == {"any": {}}
    named = to_converse_request(
        "m",
        [{"role": "user", "content": "x"}],
        tools=[WEATHER],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert named["toolConfig"]["toolChoice"] == {"tool": {"name": "get_weather"}}
    assert "toolConfig" not in to_converse_request(
        "m", [{"role": "user", "content": "x"}], tools=[WEATHER], tool_choice="none"
    )


def test_tool_round_trip_history_is_alternating_converse() -> None:
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "Weather in Paris and Rome?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                },
                {
                    "id": "t2",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": '{"temp_c": 18}'},
        {"role": "tool", "tool_call_id": "t2", "content": "sunny, 24C"},
    ]
    req = valid(to_converse_request("m", messages, tools=[WEATHER]))
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["user", "assistant", "user"]  # two tool results merged into one turn
    assert req["messages"][1]["content"][0]["toolUse"]["input"] == {"city": "Paris"}
    results = req["messages"][2]["content"]
    assert results[0]["toolResult"]["content"] == [{"json": {"temp_c": 18}}]
    assert results[1]["toolResult"]["content"] == [{"text": "sunny, 24C"}]
    assert req["system"] == [{"text": "be brief"}]


def test_tool_call_response() -> None:
    resp = from_converse_response(
        "m",
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": "get_weather",
                                "input": {"city": "Paris"},
                            }
                        }
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
        },
    )
    msg = resp.choices[0].message
    assert resp.choices[0].finish_reason == "tool_calls" and msg.content is None
    assert msg.tool_calls and json.loads(msg.tool_calls[0].function.arguments) == {"city": "Paris"}


def test_structured_output_and_reasoning_effort() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    req = valid(
        to_converse_request(
            "m",
            [{"role": "user", "content": "2+2"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "math", "schema": schema},
            },
            reasoning_effort="low",
        )
    )
    fmt = req["outputConfig"]["textFormat"]
    assert fmt["type"] == "json_schema"
    assert json.loads(fmt["structure"]["jsonSchema"]["schema"]) == {
        **schema,
        "additionalProperties": False,
    }  # strict mode added
    assert req["outputConfig"]["effort"] == "low"
    obj = to_converse_request(
        "m", [{"role": "user", "content": "x"}], response_format={"type": "json_object"}
    )
    assert "JSON object" in obj["system"][-1]["text"]


def test_reasoning_content_and_think_tags() -> None:
    resp = from_converse_response(
        "m",
        {
            "output": {
                "message": {
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": "thinking..."}}},
                        {"text": "4"},
                    ]
                }
            },
            "stopReason": "end_turn",
            "usage": {},
        },
    )
    assert (resp.choices[0].message.content, resp.choices[0].message.reasoning_content) == (
        "4",
        "thinking...",
    )
    assert split_reasoning("<think>\nhmm\n</think>\n\nAnswer") == ("Answer", "hmm")
    assert split_reasoning("no tags") == ("no tags", None)


def test_service_tier_guardrail_latency_and_unknown_params() -> None:
    req = valid(
        to_converse_request(
            "m",
            [{"role": "user", "content": "x"}],
            service_tier="flex",
            guardrail={"id": "gr1", "version": 2},
            latency="optimized",
            extra_body={"top_k": 5},
        )
    )
    assert req["serviceTier"] == {"type": "flex"}
    assert req["guardrailConfig"] == {"guardrailIdentifier": "gr1", "guardrailVersion": "2"}
    assert req["performanceConfig"] == {"latency": "optimized"}
    assert req["additionalModelRequestFields"] == {"top_k": 5}
    assert to_converse_request("m", [], service_tier="standard")["serviceTier"]["type"] == "default"
    with pytest.raises(TypeError, match=r"Unsupported parameter.*seed"):
        to_converse_request("m", [], seed=1)
    with pytest.raises(ValueError, match="service_tier"):
        to_converse_request("m", [], service_tier="turbo")


def test_image_data_url_part() -> None:
    req = valid(
        to_converse_request(
            "m",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                }
            ],
        )
    )
    image = req["messages"][0]["content"][1]["image"]
    assert image["format"] == "png" and image["source"]["bytes"].startswith(b"\x89PNG")


EVENTS = [
    {"messageStart": {"role": "assistant"}},
    {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"reasoningContent": {"text": "Let me"}},
        }
    },
    {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "Checking "}}},
    {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "now."}}},
    {
        "contentBlockStart": {
            "contentBlockIndex": 2,
            "start": {"toolUse": {"toolUseId": "t1", "name": "get_weather"}},
        }
    },
    {"contentBlockDelta": {"contentBlockIndex": 2, "delta": {"toolUse": {"input": '{"city": '}}}},
    {"contentBlockDelta": {"contentBlockIndex": 2, "delta": {"toolUse": {"input": '"Paris"}'}}}},
    {"contentBlockStop": {"contentBlockIndex": 2}},
    {"messageStop": {"stopReason": "tool_use"}},
    {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 7, "totalTokens": 17}}},
]


def test_stream_chunks_and_final_assembly() -> None:
    stream = Stream("m", converse_stream_chunks("m", iter(EVENTS)))
    chunks = list(stream)
    assert chunks[0].choices[0].delta.role == "assistant"
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Checking now."
    final = stream.final()
    msg = final.choices[0].message
    assert msg.content == "Checking now." and msg.reasoning_content == "Let me"
    assert msg.tool_calls and msg.tool_calls[0].id == "t1"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"city": "Paris"}
    assert final.choices[0].finish_reason == "tool_calls" and final.usage.total_tokens == 17


@pytest.mark.parametrize(
    "pieces",
    [
        ["<think>a", "b</think>", "answer"],
        ["<thi", "nk>ab</th", "ink>ans", "wer"],
        ["  <think>ab</think>answer"],
    ],
)
def test_streamed_think_tags_split_across_chunks(pieces: list[str]) -> None:
    s = _ThinkSplitter()
    content = reasoning = ""
    for p in pieces:
        c, r = s.feed(p)
        content += c
        reasoning += r
    content += s.buf if s.state == "answer" else ""
    assert (content, reasoning) == ("answer", "ab")


def test_plain_streamed_text_passes_through() -> None:
    s = _ThinkSplitter()
    assert s.feed("Hello") == ("Hello", "")
    assert s.feed(" <think> not reasoning") == (" <think> not reasoning", "")


def _validation_error(feature: str) -> Exception:
    from botocore.exceptions import ClientError

    return ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": f"This model doesn't support the {feature} field. "
                "Remove it and try again.",
            }
        },
        "Converse",
    )


class Rejecting:
    """Fake Converse that rejects given features once each, like Bedrock does."""

    def __init__(self, *features: str) -> None:
        self.features = list(features)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **req: Any) -> dict[str, Any]:
        self.calls.append(req)
        for f in self.features:
            top, _, sub = f.partition(".")
            if top in req and (not sub or sub.split(".")[0] in req[top]):
                raise _validation_error(f)
        return {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {},
        }


def test_tool_choice_fallback_and_learning(monkeypatch: pytest.MonkeyPatch) -> None:
    from skyhammer import inference

    monkeypatch.setattr(inference, "_UNSUPPORTED", {})
    fake = Rejecting("toolConfig.toolChoice")
    req = to_converse_request(
        "llama", [{"role": "user", "content": "x"}], tools=[WEATHER], tool_choice="required"
    )
    inference.call_with_fallbacks("llama", req, fake)
    assert len(fake.calls) == 2 and "toolChoice" not in fake.calls[1]["toolConfig"]
    inference.call_with_fallbacks("llama", req, fake)  # learned: no failing attempt now
    assert len(fake.calls) == 3


def test_json_schema_emulated_with_forced_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    from skyhammer import inference

    monkeypatch.setattr(inference, "_UNSUPPORTED", {})
    schema = {"type": "object", "properties": {"sum": {"type": "integer"}}}
    req = to_converse_request(
        "nova",
        [{"role": "user", "content": "19+23"}],
        response_format={"type": "json_schema", "json_schema": {"name": "a", "schema": schema}},
    )
    seen: list[dict[str, Any]] = []

    def fake(**r: Any) -> dict[str, Any]:
        seen.append(r)
        if "outputConfig" in r:
            raise _validation_error("outputConfig")
        valid(r)
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "x",
                                "name": inference.JSON_TOOL,
                                "input": {"sum": 42},
                            }
                        }
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {},
        }

    resp = from_converse_response("nova", inference.call_with_fallbacks("nova", req, fake))
    assert seen[1]["toolConfig"]["toolChoice"] == {"tool": {"name": inference.JSON_TOOL}}
    assert resp.choices[0].message.parsed() == {"sum": 42}
    assert resp.choices[0].finish_reason == "stop" and resp.choices[0].message.tool_calls is None


def test_unrelated_validation_errors_are_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    from botocore.exceptions import ClientError

    from skyhammer import inference

    def fake(**r: Any) -> None:
        raise ClientError({"Error": {"Code": "ValidationException", "Message": "bad input"}}, "C")

    with pytest.raises(ClientError, match="bad input"):
        inference.call_with_fallbacks("m", {"modelId": "m", "messages": []}, fake)


def test_parsed_tolerates_leaked_text() -> None:
    from skyhammer.inference import Message, extract_json

    assert Message("assistant", '19 + 23{ "sum": 42}').parsed() == {"sum": 42}  # seen live
    assert extract_json("[1, 2]") == [1, 2]
    with pytest.raises(ValueError, match="No JSON"):
        extract_json("no json here")


@pytest.mark.parametrize(
    "message, feature",
    [
        (
            "This model doesn't support the effort field. Remove effort and try again.",
            "outputConfig.effort",
        ),
        (
            "This model doesn't support the toolConfig.toolChoice.any field.",
            "toolConfig.toolChoice",
        ),
        ("This model doesn't support the outputConfig field.", "outputConfig"),
        ("The provided service tier is not supported for this model.", "serviceTier"),
    ],
)
def test_rejection_messages_seen_live(message: str, feature: str) -> None:
    from botocore.exceptions import ClientError

    from skyhammer.inference import _rejected_feature

    err = ClientError({"Error": {"Code": "ValidationException", "Message": message}}, "Converse")
    assert _rejected_feature(err) == feature


def test_reasoning_effort_provider_mappings() -> None:
    from skyhammer.inference import degrade

    base = to_converse_request(
        "m",
        [{"role": "user", "content": "x"}],
        reasoning_effort="low",
        max_tokens=200,
        temperature=0.2,
    )
    claude = degrade(base, "outputConfig.effort", "us.anthropic.claude-haiku-4-5", quiet=True)
    assert claude and claude["additionalModelRequestFields"]["thinking"]["budget_tokens"] == 1024
    assert claude["inferenceConfig"]["maxTokens"] >= 2048  # budget + room for the answer
    assert "temperature" not in claude["inferenceConfig"]
    valid(claude)
    oss = degrade(base, "outputConfig.effort", "openai.gpt-oss-20b-1:0", quiet=True)
    assert oss and oss["additionalModelRequestFields"] == {"reasoning_effort": "low"}
    other = degrade(base, "outputConfig.effort", "qwen.qwen3-32b-v1:0", quiet=True)
    assert other and "outputConfig" not in other and "additionalModelRequestFields" not in other


def test_strict_schema_adds_additional_properties_recursively() -> None:
    from skyhammer.inference import strict_schema

    s = strict_schema(
        {
            "type": "object",
            "properties": {
                "a": {"type": "object", "properties": {"b": {"type": "string"}}},
                "c": {"type": "array", "items": {"type": "object", "properties": {}}},
                "d": {"type": "object", "additionalProperties": True},
            },
        }
    )
    assert s["additionalProperties"] is False
    assert s["properties"]["a"]["additionalProperties"] is False
    assert s["properties"]["c"]["items"]["additionalProperties"] is False
    assert s["properties"]["d"]["additionalProperties"] is True  # explicit choice kept


LLAMA_TEXT_CALL = json.dumps(
    {"type": "function", "name": "get_weather", "parameters": {"city": "Paris"}}
)


def test_text_tool_call_recovered_only_for_known_tools() -> None:
    resp = {
        "output": {"message": {"content": [{"text": LLAMA_TEXT_CALL}]}},
        "stopReason": "end_turn",
        "usage": {},
    }  # Llama 3.3, seen live
    r = from_converse_response("m", resp, tool_names={"get_weather"})
    tc = r.choices[0].message.tool_calls
    assert tc and json.loads(tc[0].function.arguments) == {"city": "Paris"}
    assert r.choices[0].finish_reason == "tool_calls" and not r.choices[0].message.content
    assert (
        from_converse_response("m", resp, tool_names={"other"}).choices[0].message.tool_calls
        is None
    )
    assert from_converse_response("m", resp).choices[0].message.tool_calls is None


def test_text_call_to_hidden_json_tool_becomes_content() -> None:
    from skyhammer.inference import JSON_TOOL

    text = json.dumps({"type": "function", "name": JSON_TOOL, "parameters": {"city": "Paris"}})
    resp = {
        "output": {"message": {"content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {},
    }
    msg = from_converse_response("m", resp).choices[0].message
    assert msg.parsed() == {"city": "Paris"} and msg.tool_calls is None


def test_prompt_cache_points() -> None:
    req = valid(
        to_converse_request(
            "m",
            [
                {"role": "system", "content": "long instructions"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "doc", "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": "question"},
                    ],
                },
            ],
            prompt_cache="system",
        )
    )
    assert req["system"] == [{"text": "long instructions"}, {"cachePoint": {"type": "default"}}]
    assert req["messages"][0]["content"][1] == {"cachePoint": {"type": "default"}}
    parts = valid(
        to_converse_request(
            "m",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "cache_point"},
                        {"type": "text", "text": "b"},
                    ],
                }
            ],
        )
    )
    assert parts["messages"][0]["content"][1] == {"cachePoint": {"type": "default"}}
    with pytest.raises(ValueError, match="prompt_cache"):
        to_converse_request("m", [{"role": "system", "content": "x"}], prompt_cache="all")


def test_text_completions_wrap_prompt() -> None:
    from skyhammer.inference import Chat

    class RT:
        def converse(self, **req: Any) -> dict[str, Any]:
            assert req["messages"] == [{"role": "user", "content": [{"text": "Once upon"}]}]
            return {
                "output": {"message": {"content": [{"text": " a time"}]}},
                "stopReason": "max_tokens",
                "usage": {},
            }

    c = Chat(RT(), "m").text_completions.create(prompt="Once upon", max_tokens=3)
    assert c.choices[0].text == " a time" and c.choices[0].finish_reason == "length"


def test_document_parts() -> None:
    pdf = base64.b64encode(b"%PDF-1.4 tiny").decode()
    req = valid(
        to_converse_request(
            "m",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize"},
                        {
                            "type": "file",
                            "file": {
                                "file_data": f"data:application/pdf;base64,{pdf}",
                                "filename": "Q3 report (final).pdf",
                            },
                        },
                    ],
                }
            ],
        )
    )
    doc = req["messages"][0]["content"][1]["document"]
    assert doc["format"] == "pdf" and doc["name"] == "Q3 report (final)"
    assert doc["source"]["bytes"].startswith(b"%PDF")
    with pytest.raises(ValueError, match="file parts need"):
        to_converse_request(
            "m",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "file": {"file_data": "data:application/zip;base64,AA=="}}
                    ],
                }
            ],
        )


def test_schema_keywords_rejected_live_are_relaxed(monkeypatch: pytest.MonkeyPatch) -> None:
    from botocore.exceptions import ClientError

    from skyhammer import inference

    monkeypatch.setattr(inference, "_UNSUPPORTED", {})
    schema = {
        "type": "object",
        "properties": {"urgency": {"type": "integer", "minimum": 1, "maximum": 5}},
        "required": ["urgency"],
    }
    req = to_converse_request(
        "claude",
        [{"role": "user", "content": "x"}],
        response_format={"type": "json_schema", "json_schema": {"name": "t", "schema": schema}},
    )
    seen: list[dict[str, Any]] = []

    def fake(**r: Any) -> dict[str, Any]:
        seen.append(r)
        sent = json.loads(r["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"])
        if "maximum" in sent["properties"]["urgency"]:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "The model returned the following errors: "
                        "output_config.format.schema: For 'integer' type, "
                        "properties maximum, minimum are not supported",
                    }
                },
                "Converse",
            )
        return {
            "output": {"message": {"content": [{"text": '{"urgency": 4}'}]}},
            "stopReason": "end_turn",
            "usage": {},
        }

    inference.call_with_fallbacks("claude", req, fake)
    relaxed = json.loads(seen[1]["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"])
    urgency = relaxed["properties"]["urgency"]
    assert "minimum" not in urgency and "maximum" not in urgency
    assert "minimum: 1" in urgency["description"] and "maximum: 5" in urgency["description"]
    inference.call_with_fallbacks("claude", req, fake)  # learned: relaxed on the first try
    assert len(seen) == 3
