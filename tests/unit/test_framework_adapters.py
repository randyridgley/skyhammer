import asyncio
import json
from types import SimpleNamespace
from typing import Any

import litellm
import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import FunctionTool

from skyhammer import TrafficSplit
from skyhammer.inference import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    ChunkChoice,
    Delta,
    FunctionCall,
    FunctionCallDelta,
    Message,
    Stream,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from skyhammer.integrations import litellm as sh_litellm
from skyhammer.integrations.langchain import ChatSkyhammer
from skyhammer.integrations.llama_index import SkyhammerLLM, to_openai_messages

USAGE = Usage(7, 3, 10, cost_usd=0.0002)


class ScriptedLLM:
    """Asks for ``get_weather`` when offered tools and no result is in yet; otherwise
    echoes the last message. Records what each call sent."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, *, messages: list[dict[str, Any]], stream: bool = False, **params: Any) -> Any:
        self.calls.append({"messages": messages, "stream": stream, **params})
        wants_tool = params.get("tools") and messages[-1]["role"] != "tool"
        if stream:
            return Stream("m", iter(self._chunks(bool(wants_tool))))
        if wants_tool:
            call = ToolCall("call_1", FunctionCall("get_weather", '{"city": "Paris"}'))
            msg = Message("assistant", None, tool_calls=[call])
            return ChatCompletion("r1", "m", [Choice(0, msg, "tool_calls")], USAGE)
        text = f"echo: {messages[-1]['content']}"
        return ChatCompletion("r2", "m", [Choice(0, Message("assistant", text), "stop")], USAGE)

    @staticmethod
    def _chunks(wants_tool: bool) -> list[ChatCompletionChunk]:
        if wants_tool:
            deltas = [
                Delta(
                    tool_calls=[ToolCallDelta(0, "call_1", FunctionCallDelta("get_weather", ""))]
                ),
                Delta(tool_calls=[ToolCallDelta(0, None, FunctionCallDelta(None, '{"city": '))]),
                Delta(tool_calls=[ToolCallDelta(0, None, FunctionCallDelta(None, '"Paris"}'))]),
            ]
            finish = "tool_calls"
        else:
            deltas = [Delta(content="Hel"), Delta(content="lo")]
            finish = "stop"
        chunks = [ChatCompletionChunk("s", "m", [ChunkChoice(0, d)]) for d in deltas]
        chunks.append(ChatCompletionChunk("s", "m", [ChunkChoice(0, Delta(), finish)]))
        chunks.append(ChatCompletionChunk("s", "m", [], usage=USAGE))
        return chunks


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return json.dumps({"city": city, "temp_c": 18})


# ---- LangChain ----


def test_langchain_invoke_passes_params_and_reports_usage() -> None:
    fake = ScriptedLLM()
    chat = ChatSkyhammer(model=fake, max_tokens=50, model_kwargs={"service_tier": "flex"})
    out = chat.invoke([SystemMessage("be brief"), HumanMessage("hi")], stop=["\n"])
    assert out.content == "echo: hi"
    assert out.usage_metadata == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    assert out.response_metadata["cost_usd"] == 0.0002
    sent = fake.calls[0]
    assert sent["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert sent["max_tokens"] == 50 and sent["stop"] == ["\n"] and sent["service_tier"] == "flex"
    assert "temperature" not in sent  # unset parameters are never sent


def test_langchain_tool_round_trip() -> None:
    fake = ScriptedLLM()
    weather = tool(get_weather)
    chat = ChatSkyhammer(model=fake).bind_tools([weather], tool_choice="get_weather")
    question = HumanMessage("weather in Paris?")
    asked = chat.invoke([question])
    assert asked.tool_calls == [
        {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1", "type": "tool_call"}
    ]
    assert fake.calls[0]["tools"][0]["function"]["name"] == "get_weather"
    assert fake.calls[0]["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
    result = ToolMessage(weather.invoke(asked.tool_calls[0]["args"]), tool_call_id="call_1")
    answer = chat.invoke([question, asked, result])
    assert "temp_c" in str(answer.content)
    sent = fake.calls[1]["messages"]
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'
    assert sent[2] == {"role": "tool", "tool_call_id": "call_1", "content": result.content}


def test_langchain_streams_text_and_tool_calls() -> None:
    fake = ScriptedLLM()
    chat = ChatSkyhammer(model=fake)
    chunks = list(chat.stream("hi"))
    assert "".join(str(c.content) for c in chunks) == "Hello"
    total = chunks[0]
    for c in chunks[1:]:
        total += c
    assert total.usage_metadata["total_tokens"] == 10  # type: ignore[index]

    merged = None
    for c in chat.bind_tools([get_weather]).stream("weather?"):
        merged = c if merged is None else merged + c
    assert merged is not None
    assert merged.tool_calls[0]["args"] == {"city": "Paris"}  # type: ignore[attr-defined]


def test_langchain_async_and_split_variant() -> None:
    split = TrafficSplit({"only": 1}, llms={"only": ScriptedLLM()})  # type: ignore[dict-item]
    chat = ChatSkyhammer(model=split)
    out = asyncio.run(chat.ainvoke("hi"))
    assert out.response_metadata["variant"] == "only"
    assert chat._llm_type == "skyhammer"


# ---- LlamaIndex ----


def test_llama_index_chat_complete_and_stream() -> None:
    fake = ScriptedLLM()
    llm = SkyhammerLLM(model=fake, max_tokens=20)
    resp = llm.chat([ChatMessage(role="user", content="hi")])
    assert resp.message.content == "echo: hi"
    assert resp.additional_kwargs["total_tokens"] == 10
    assert llm.complete("ping").text == "echo: ping"
    streamed = list(llm.stream_complete("ping"))
    assert streamed[-1].text == "Hello" and streamed[0].delta == "Hel"
    assert asyncio.run(llm.acomplete("ping")).text == "echo: ping"

    async def astream() -> str:
        text = ""
        async for r in await llm.astream_chat([ChatMessage(role="user", content="x")]):
            text += r.delta or ""
        return text

    assert asyncio.run(astream()) == "Hello"
    assert llm.metadata.is_function_calling_model and llm.metadata.num_output == 20
    assert fake.calls[0]["max_tokens"] == 20


def test_llama_index_tool_calls() -> None:
    fake = ScriptedLLM()
    llm = SkyhammerLLM(model=fake)
    weather = FunctionTool.from_defaults(fn=get_weather)
    resp = llm.chat_with_tools([weather], user_msg="weather in Paris?", tool_required=True)
    (selection,) = llm.get_tool_calls_from_response(resp)
    assert (selection.tool_name, selection.tool_kwargs) == ("get_weather", {"city": "Paris"})
    assert fake.calls[0]["tool_choice"] == "required"
    assert fake.calls[0]["tools"][0]["function"]["name"] == "get_weather"

    history = [
        ChatMessage(role="user", content="weather in Paris?"),
        resp.message,
        ChatMessage(
            role="tool", content=get_weather("Paris"), additional_kwargs={"tool_call_id": "call_1"}
        ),
    ]
    sent = to_openai_messages(history)
    assert sent[1]["tool_calls"][0]["id"] == "call_1"
    assert json.loads(sent[1]["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
    assert sent[2]["role"] == "tool" and sent[2]["tool_call_id"] == "call_1"
    answer = llm.chat_with_tools([weather], chat_history=history)
    assert "temp_c" in (answer.message.content or "")

    streamed = list(llm.stream_chat_with_tools([weather], user_msg="weather?"))
    (selection,) = llm.get_tool_calls_from_response(streamed[-1])
    assert selection.tool_kwargs == {"city": "Paris"}
    with pytest.raises(ValueError, match="Expected a tool call"):
        llm.get_tool_calls_from_response(answer)


# ---- LiteLLM ----


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(litellm, "custom_provider_map", [])
    fake = ScriptedLLM()
    registered = sh_litellm.register()
    registered.target("support@prod")._resolved = fake  # skip resolving the name
    return SimpleNamespace(provider=registered, fake=fake)


def test_litellm_completion_and_tools(provider: Any) -> None:
    resp = litellm.completion(
        model="skyhammer/support@prod",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=30,
        temperature=0.2,
    )
    assert resp.choices[0].message.content == "echo: hi"
    assert resp.usage.total_tokens == 10
    assert resp._hidden_params["cost_usd"] == 0.0002
    sent = provider.fake.calls[0]
    assert sent["max_tokens"] == 30 and sent["temperature"] == 0.2

    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    resp = litellm.completion(
        model="skyhammer/support@prod",
        messages=[{"role": "user", "content": "weather?"}],
        tools=tools,
    )
    call = resp.choices[0].message.tool_calls[0]
    assert (call.id, call.function.name) == ("call_1", "get_weather")
    assert provider.fake.calls[1]["tools"] == tools
    assert sh_litellm.register() is not provider.provider  # re-registering replaces it
    assert len(litellm.custom_provider_map) == 1


def test_litellm_streaming_and_async(provider: Any) -> None:
    chunks = list(
        litellm.completion(
            model="skyhammer/support@prod",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    )
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Hello"

    async def run() -> tuple[str, str]:
        resp = await litellm.acompletion(
            model="skyhammer/support@prod", messages=[{"role": "user", "content": "a"}]
        )
        text = ""
        stream = await litellm.acompletion(
            model="skyhammer/support@prod",
            messages=[{"role": "user", "content": "b"}],
            stream=True,
        )
        async for c in stream:
            text += c.choices[0].delta.content or ""
        return resp.choices[0].message.content, text

    assert asyncio.run(run()) == ("echo: a", "Hello")
