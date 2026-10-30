import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from skyhammer import responses as R
from skyhammer.errors import SkyhammerError
from skyhammer.inference import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    ChunkChoice,
    Delta,
    FunctionCall,
    Message,
    Stream,
    ToolCall,
    Usage,
)
from skyhammer.mantle import MantleError
from skyhammer.responses import Response, Responses
from tests.conftest import FIXTURES

# Error bodies recorded live (2026-10-30).
RUNTIME_NOT_FOUND = json.dumps(
    {
        "error": {
            "message": "The model doesn't exist or doesn't support this API. Retry your "
            "request with a different model ID.",
            "type": "not_found_error",
            "param": None,
            "code": "model_not_found",
        }
    }
)
MANTLE_UNSUPPORTED = json.dumps(
    {
        "error": {
            "code": "validation_error",
            "message": "The model 'qwen.qwen3-32b' does not support the '/v1/responses' API",
            "param": None,
            "type": "invalid_request_error",
        }
    }
)


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "_ROUTES", {})
    monkeypatch.setattr(R, "LOCAL", R._LocalStore())


# ---- recorded native responses ----


def test_parses_recorded_runtime_and_mantle_responses() -> None:
    runtime = Response.from_json(
        json.loads((FIXTURES / "responses_runtime_gpt56.json").read_text())
    )
    assert runtime.output_text == "42"
    assert runtime.usage.input_tokens == 41 and runtime.usage.total_tokens == 46
    mantle = Response.from_json(json.loads((FIXTURES / "responses_mantle_gptoss.json").read_text()))
    assert mantle.output_text == "42"  # the reasoning item isn't part of the text
    assert mantle.output[0]["type"] == "reasoning"
    assert mantle.to_json() is mantle.raw


def test_unsupported_detection_uses_real_error_bodies() -> None:
    assert R._unsupported(MantleError(404, "POST", "/", RUNTIME_NOT_FOUND))
    assert R._unsupported(MantleError(400, "POST", "/", MANTLE_UNSUPPORTED))
    assert R._unsupported(MantleError(400, "POST", "/", "model `x` isn't supported on this route"))
    assert not R._unsupported(MantleError(400, "POST", "/", '{"error": {"code": "bad_param"}}'))
    assert not R._unsupported(MantleError(429, "POST", "/", "slow down"))


def test_mantle_model_ids() -> None:
    available = ["openai.gpt-oss-20b", "openai.gpt-oss-120b"]
    assert R.mantle_model_id("openai.gpt-oss-20b-1:0", available) == "openai.gpt-oss-20b"
    assert R.mantle_model_id("us.openai.gpt-oss-120b-1:0", available) == "openai.gpt-oss-120b"
    assert R.mantle_model_id("openai.gpt-oss-20b", available) == "openai.gpt-oss-20b"
    assert R.mantle_model_id("us.amazon.nova-micro-v1:0", available) is None


# ---- request mapping for emulated models ----


def test_input_items_become_chat_messages() -> None:
    assert R.input_to_messages("hi") == [{"role": "user", "content": "hi"}]
    msgs = R.input_to_messages(
        [
            {"role": "developer", "content": "Be brief."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Weather in "},
                    {"type": "input_text", "text": "Paris and Rome?"},
                ],
            },
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "w",
                "arguments": '{"city":"Paris"}',
            },
            {"type": "function_call", "call_id": "c2", "name": "w", "arguments": '{"city":"Rome"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "18C"},
            {"type": "function_call_output", "call_id": "c2", "output": {"temp_c": 21}},
            {"type": "reasoning", "summary": []},
        ]
    )
    assert msgs[0] == {"role": "system", "content": "Be brief."}
    assert msgs[1] == {"role": "user", "content": "Weather in Paris and Rome?"}
    assert [c["id"] for c in msgs[2]["tool_calls"]] == ["c1", "c2"]  # one assistant turn
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "18C"}
    assert msgs[4]["content"] == '{"temp_c": 21}'
    assert len(msgs) == 5
    image = R.input_to_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What?"},
                    {"type": "input_image", "image_url": "s3://b/k.png"},
                ],
            }
        ]
    )
    assert image[0]["content"][1] == {"type": "image_url", "image_url": {"url": "s3://b/k.png"}}
    with pytest.raises(SkyhammerError, match="native API"):
        R.input_to_messages([{"type": "web_search_call"}])


def test_params_map_to_chat_completions() -> None:
    out = R.to_chat_params(
        {
            "max_output_tokens": 50,
            "temperature": 0.2,
            "tools": [{"type": "function", "name": "w", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "w"},
            "text": {
                "format": {"type": "json_schema", "name": "out", "schema": {"type": "object"}}
            },
            "reasoning": {"effort": "low"},
            "store": True,
            "metadata": {"a": "b"},
        }
    )
    assert out == {
        "max_tokens": 50,
        "temperature": 0.2,
        "tools": [
            {"type": "function", "function": {"name": "w", "parameters": {"type": "object"}}}
        ],
        "tool_choice": {"type": "function", "function": {"name": "w"}},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "out", "schema": {"type": "object"}},
        },
        "reasoning_effort": "low",
    }
    with pytest.raises(SkyhammerError, match="background needs a model with the native"):
        R.to_chat_params({"background": True})
    with pytest.raises(SkyhammerError, match="server-side tools"):
        R.to_chat_params({"tools": [{"type": "web_search"}]})


# ---- fakes ----


class FakeChat:
    def __init__(self, replies: list[ChatCompletion]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []
        self.completions = self

    def create(self, *, messages: Any, stream: bool = False, **params: Any) -> Any:
        self.calls.append({"messages": [dict(m) for m in messages], **params})
        reply = self.replies.pop(0)
        if stream:
            text = reply.choices[0].message.content or ""
            chunks = [
                ChatCompletionChunk("c", "m", [ChunkChoice(0, Delta(content=text[:2]))]),
                ChatCompletionChunk(
                    "c", "m", [ChunkChoice(0, Delta(content=text[2:]), "stop")], usage=reply.usage
                ),
            ]
            return Stream("m", iter(chunks))
        return reply


def completion(
    text: str | None = None, calls: list[ToolCall] | None = None, finish: str = "stop"
) -> ChatCompletion:
    return ChatCompletion(
        "c",
        "m",
        [Choice(0, Message("assistant", text, calls), finish)],
        Usage(10, 3, 13, cost_usd=0.0001),
    )


def fake_llm(
    model_id: str = "us.amazon.nova-micro-v1:0",
    replies: list[ChatCompletion] | None = None,
    model_ref: Any = None,
) -> Any:
    spend = SimpleNamespace(checked=0, recorded=[])
    spend.check = lambda: setattr(spend, "checked", spend.checked + 1)
    spend.record = lambda mid, cost: spend.recorded.append((mid, cost))
    llm = SimpleNamespace(
        model_ref=model_ref,
        model_id=model_id,
        name="nova-micro",
        session=SimpleNamespace(region="us-east-1", spend=spend),
        chat=FakeChat(replies or []),
    )

    def record_usage(mid: str, usage: Usage, tier: Any) -> None:
        usage.cost_usd = 0.5
        spend.record(mid, usage.cost_usd)

    llm._record_usage = record_usage
    return llm


class FakeClient:
    """Scripted MantleClient: responses keyed by (endpoint, method, path)."""

    log: ClassVar[list[tuple[str, str, str, Any, Any]]] = []
    script: ClassVar[dict[tuple[str, str, str], Any]] = {}

    def __init__(self, session: Any, region: str, *, endpoint: str = "mantle", **_: Any) -> None:
        self.endpoint = endpoint

    def _answer(self, method: str, path: str, body: Any, headers: Any) -> Any:
        FakeClient.log.append((self.endpoint, method, path, body, headers))
        out = FakeClient.script[(self.endpoint, method, path)]
        if isinstance(out, Exception):
            raise out
        return out

    def request(
        self, method: str, path: str, *, json_body: Any = None, headers: Any = None, **_: Any
    ) -> Any:
        return self._answer(method, path, json_body, headers)

    def stream(
        self, method: str, path: str, *, json_body: Any, headers: Any = None
    ) -> Iterator[dict[str, Any]]:
        yield from self._answer(method, path, json_body, headers)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> type[FakeClient]:
    FakeClient.log = []
    FakeClient.script = {}
    monkeypatch.setattr(R, "MantleClient", FakeClient)
    return FakeClient


NATIVE = {
    "id": "resp_abc",
    "model": "m",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        }
    ],
    "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
}


# ---- routing ----


def test_native_runtime_first_then_cached(client: type[FakeClient]) -> None:
    client.script[("runtime", "POST", "/openai/v1/responses")] = NATIVE
    llm = fake_llm("us.openai.gpt-5.6-luna")
    r = Responses(llm).create(input="hi", max_output_tokens=10, store=False)
    assert isinstance(r, Response) and r.output_text == "hello" and not r.emulated
    assert client.log[0][3] == {
        "input": "hi",
        "max_output_tokens": 10,
        "store": False,
        "model": "us.openai.gpt-5.6-luna",
    }
    assert r.usage.cost_usd == 0.5 and llm.session.spend.checked == 1
    assert R._ROUTES[("us-east-1", "us.openai.gpt-5.6-luna")] == (
        "runtime",
        "us.openai.gpt-5.6-luna",
    )


def test_falls_back_to_mantle_with_its_model_id(client: type[FakeClient]) -> None:
    client.script[("runtime", "POST", "/openai/v1/responses")] = MantleError(
        404, "POST", "/", RUNTIME_NOT_FOUND
    )
    client.script[("mantle", "GET", "/v1/models")] = {"data": [{"id": "openai.gpt-oss-20b"}]}
    client.script[("mantle", "POST", "/v1/responses")] = NATIVE
    llm = fake_llm("openai.gpt-oss-20b-1:0")
    r = Responses(llm).create(input="hi")
    assert r.output_text == "hello"  # type: ignore[union-attr]
    assert client.log[-1][3]["model"] == "openai.gpt-oss-20b"
    assert llm.session.spend.recorded == [("openai.gpt-oss-20b", None)]
    client.log.clear()
    Responses(llm).create(input="again")
    assert [entry[0] for entry in client.log] == ["mantle"]  # remembered: no probing


def test_unsupported_everywhere_emulates_over_chat(client: type[FakeClient]) -> None:
    client.script[("runtime", "POST", "/openai/v1/responses")] = MantleError(
        404, "POST", "/", RUNTIME_NOT_FOUND
    )
    client.script[("mantle", "GET", "/v1/models")] = {"data": []}
    llm = fake_llm(replies=[completion("OK"), completion("Ada")])
    first = Responses(llm).create(input="My name is Ada.", instructions="Be brief.")
    assert first.emulated and first.id.startswith(R.EMULATED_PREFIX)  # type: ignore[union-attr]
    second = Responses(llm).create(input="My name?", previous_response_id=first.id)  # type: ignore[union-attr]
    assert second.output_text == "Ada"  # type: ignore[union-attr]
    sent = llm.chat.calls[1]["messages"]
    assert sent == [  # history kept, instructions not carried over (as in the OpenAI spec)
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "My name?"},
    ]
    assert Responses(llm).retrieve(first.id) is first  # type: ignore[union-attr]
    Responses(llm).delete(first.id)  # type: ignore[union-attr]
    with pytest.raises(SkyhammerError, match="No stored response"):
        Responses(llm).retrieve(first.id)  # type: ignore[union-attr]


def test_emulated_store_false_keeps_nothing_and_unknown_ids_fail(client: type[FakeClient]) -> None:
    R._ROUTES[("us-east-1", "us.amazon.nova-micro-v1:0")] = ("emulated", None)
    llm = fake_llm(replies=[completion("x")])
    r = Responses(llm).create(input="hi", store=False)
    with pytest.raises(SkyhammerError, match="in memory only"):
        Responses(llm).create(input="more", previous_response_id=r.id)  # type: ignore[union-attr]
    assert client.log == []


def test_emulated_tool_round_trip_and_length_limit(client: type[FakeClient]) -> None:
    R._ROUTES[("us-east-1", "us.amazon.nova-micro-v1:0")] = ("emulated", None)
    call = ToolCall("t1", FunctionCall("w", '{"city": "Paris"}'))
    llm = fake_llm(
        replies=[completion(None, [call], "tool_calls"), completion("18C", finish="length")]
    )
    tools = [{"type": "function", "name": "w", "parameters": {"type": "object"}}]
    r1 = Responses(llm).create(input="Weather?", tools=tools)
    assert r1.function_calls == [  # type: ignore[union-attr]
        {
            "type": "function_call",
            "id": r1.function_calls[0]["id"],
            "call_id": "t1",  # type: ignore[union-attr]
            "name": "w",
            "arguments": '{"city": "Paris"}',
            "status": "completed",
        }
    ]
    r2 = Responses(llm).create(
        input=[{"type": "function_call_output", "call_id": "t1", "output": "18C"}],
        previous_response_id=r1.id,
        tools=tools,  # type: ignore[union-attr]
    )
    assert r2.status == "incomplete"  # type: ignore[union-attr]
    assert r2.incomplete_details == {"reason": "max_output_tokens"}  # type: ignore[union-attr]
    sent = llm.chat.calls[1]["messages"]
    assert sent[1]["tool_calls"][0]["id"] == "t1"
    assert sent[2] == {"role": "tool", "tool_call_id": "t1", "content": "18C"}


def test_emulated_stream_events(client: type[FakeClient]) -> None:
    R._ROUTES[("us-east-1", "us.amazon.nova-micro-v1:0")] = ("emulated", None)
    llm = fake_llm(replies=[completion("hello")])
    s = Responses(llm).create(input="hi", stream=True)
    assert isinstance(s, R.ResponseStream)
    events = list(s)
    assert [e["type"] for e in events] == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ]
    assert "".join(e.get("delta", "") for e in events) == "hello"
    final = s.final()
    assert final.output_text == "hello" and final.emulated
    assert R.LOCAL.get(final.id) is not None


def test_native_stream_and_guardrail_headers(client: type[FakeClient]) -> None:
    completed = {"type": "response.completed", "response": NATIVE}
    client.script[("runtime", "POST", "/openai/v1/responses")] = [
        {"type": "response.created", "response": {"id": "resp_abc"}},
        {"type": "response.output_text.delta", "delta": "hello"},
        completed,
    ]
    llm = fake_llm("us.openai.gpt-5.6-luna")
    s = Responses(llm).create(input="hi", stream=True, guardrail={"id": "gr1", "version": 2})
    assert s.final().output_text == "hello"  # type: ignore[union-attr]
    _, _, _, body, headers = client.log[0]
    assert body["stream"] is True and "guardrail" not in body
    assert headers == {
        "X-Amzn-Bedrock-GuardrailIdentifier": "gr1",
        "X-Amzn-Bedrock-GuardrailVersion": "2",
    }
    assert llm.session.spend.recorded == [("us.openai.gpt-5.6-luna", 0.5)]


def test_mantle_fine_tunes_go_to_mantle_and_errors_surface(client: type[FakeClient]) -> None:
    ref = SimpleNamespace(
        api="mantle", model_arn="ft:openai.gpt-oss-20b:abc", region="us-west-2", name="support-oss"
    )
    client.script[("mantle", "POST", "/v1/responses")] = MantleError(500, "POST", "/", "boom")
    llm = fake_llm(model_ref=ref)
    with pytest.raises(MantleError, match="500"):
        Responses(llm).create(input="hi")
    with pytest.raises(SkyhammerError, match="Guardrails on bedrock-mantle"):
        Responses(llm)._native("mantle", "x", {"guardrail": {"id": "g"}}, False)


def test_retrieve_and_delete_native_ids(client: type[FakeClient]) -> None:
    client.script[("runtime", "GET", "/openai/v1/responses/resp_abc")] = MantleError(
        404, "GET", "/", "{}"
    )
    client.script[("mantle", "GET", "/v1/responses/resp_abc")] = NATIVE
    client.script[("runtime", "DELETE", "/openai/v1/responses/resp_gone")] = MantleError(
        404, "DELETE", "/", "{}"
    )
    client.script[("mantle", "DELETE", "/v1/responses/resp_gone")] = MantleError(
        404, "DELETE", "/", "{}"
    )
    llm = fake_llm("x")
    assert Responses(llm).retrieve("resp_abc").output_text == "hello"
    with pytest.raises(SkyhammerError, match="No stored response"):
        Responses(llm).delete("resp_gone")


def test_response_ids_cannot_reach_other_api_paths(client: type[FakeClient]) -> None:
    llm = fake_llm("x")
    for bad in ("../models/ft:x", "resp_abc/../../files", "resp_a?x=1", "resp_a#b", ""):
        for action in (Responses(llm).retrieve, Responses(llm).delete, Responses(llm).cancel):
            with pytest.raises(SkyhammerError):
                action(bad)
    assert client.log == []  # nothing was sent
