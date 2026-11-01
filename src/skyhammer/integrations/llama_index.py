"""LlamaIndex LLM: ``SkyhammerLLM(model="support@prod")``.

A function-calling LLM, so it drives ``FunctionAgent`` / ``AgentWorkflow`` and query
engines alike: ``chat``, ``complete``, their streaming and async forms, and
``chat_with_tools`` / ``predict_and_call``. ``ChatResponse.additional_kwargs`` carries
token counts, ``cost_usd`` and, under a traffic split, the ``variant`` that answered.

Needs ``pip install "skyhammer[llama-index]"``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Sequence
from typing import TYPE_CHECKING, Any

from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from pydantic import Field, PrivateAttr

from skyhammer.inference import ChatCompletion
from skyhammer.integrations._openai import compact, metadata, tool_call_dicts, usage_dict
from skyhammer.target import Target

if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool


class SkyhammerLLM(FunctionCallingLLM):
    """A Skyhammer model as a LlamaIndex LLM.

    ``model`` is a name (``"nova-micro"``, a fine-tune, ``"support@prod"``), an ``LLM`` or a
    ``TrafficSplit``. ``model_kwargs`` passes other ``chat.completions.create`` arguments
    (``guardrail=``, ``reasoning_effort=``...)."""

    model: Any = Field(description="Skyhammer model name, LLM or TrafficSplit")
    max_tokens: int | None = None
    temperature: float | None = None
    context_window: int = Field(default=128_000, description="Used to size prompts")
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    session: Any = Field(default=None, exclude=True)
    _target: Target = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._target = Target(self.model, self.session)

    @classmethod
    def class_name(cls) -> str:
        return "SkyhammerLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens or 1024,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self._target.name,
        )

    def _params(self, **kwargs: Any) -> dict[str, Any]:
        return compact(
            {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **self.model_kwargs,
                **kwargs,
            }
        )

    # ---- chat ----

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        result: ChatCompletion = self._target.create(
            messages=to_openai_messages(messages), **self._params(**kwargs)
        )
        msg = result.choices[0].message
        calls = tool_call_dicts(msg.tool_calls)
        extra = {**usage_dict(result.usage), **metadata(result)}
        if msg.reasoning_content:
            extra["reasoning_content"] = msg.reasoning_content
        return ChatResponse(
            message=_assistant(msg.content or "", calls), raw=result.raw, additional_kwargs=extra
        )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        stream = self._target.create(
            messages=to_openai_messages(messages), stream=True, **self._params(**kwargs)
        )

        def gen() -> ChatResponseGen:
            text = ""
            calls: dict[int, dict[str, Any]] = {}
            for chunk in stream:
                delta = ""
                for choice in chunk.choices:
                    delta += choice.delta.content or ""
                    for d in choice.delta.tool_calls or []:
                        call = calls.setdefault(
                            d.index, {"id": None, "function": {"name": "", "arguments": ""}}
                        )
                        call["id"] = d.id or call["id"]
                        call["function"]["name"] += d.function.name or ""
                        call["function"]["arguments"] += d.function.arguments or ""
                text += delta
                extra: dict[str, Any] = {}
                if chunk.usage:
                    extra = {**usage_dict(chunk.usage), "cost_usd": chunk.usage.cost_usd}
                    if stream.variant:
                        extra["variant"] = stream.variant
                yield ChatResponse(
                    message=_assistant(text, [calls[i] for i in sorted(calls)], partial=True),
                    delta=delta,
                    additional_kwargs=extra,
                )

        return gen()

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        responses = await asyncio.to_thread(self.stream_chat, messages, **kwargs)

        async def gen() -> AsyncGenerator[ChatResponse, None]:
            while (item := await asyncio.to_thread(next, responses, None)) is not None:
                yield item

        return gen()

    # ---- completion: the prompt is sent as one user turn ----

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return chat_to_completion_decorator(self.chat)(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        return stream_chat_to_completion_decorator(self.stream_chat)(prompt, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return await achat_to_completion_decorator(self.achat)(prompt, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await astream_chat_to_completion_decorator(self.astream_chat)(prompt, **kwargs)

    # ---- tools ----

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | ChatMessage | None = None,
        chat_history: list[ChatMessage] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = list(chat_history or [])
        if user_msg is not None:
            if isinstance(user_msg, str):
                user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)
            messages.append(user_msg)
        out: dict[str, Any] = {"messages": messages, **kwargs}
        if tools:
            out["tools"] = [t.metadata.to_openai_tool() for t in tools]
            out["tool_choice"] = "required" if tool_required else "auto"
        return out

    def get_tool_calls_from_response(
        self, response: ChatResponse, error_on_no_tool_call: bool = True, **kwargs: Any
    ) -> list[ToolSelection]:
        calls = [
            ToolSelection(
                tool_id=b.tool_call_id or "",
                tool_name=b.tool_name,
                tool_kwargs=_kwargs(b.tool_kwargs),
            )
            for b in response.message.blocks
            if isinstance(b, ToolCallBlock)
        ]
        if not calls and error_on_no_tool_call:
            raise ValueError(f"Expected a tool call, got: {response.message.content!r}")
        return calls


def _kwargs(value: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}  # a partial stream; the finished response has the full arguments
    return parsed if isinstance(parsed, dict) else {}


def _assistant(text: str, calls: list[dict[str, Any]], *, partial: bool = False) -> ChatMessage:
    blocks: list[Any] = [TextBlock(text=text)] if text or not calls else []
    for c in calls:
        args = c["function"]["arguments"]
        blocks.append(
            ToolCallBlock(
                tool_call_id=c["id"],
                tool_name=c["function"]["name"],
                tool_kwargs=args if partial else _kwargs(args),
            )
        )
    return ChatMessage(role=MessageRole.ASSISTANT, blocks=blocks)


def to_openai_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """LlamaIndex messages as ``chat.completions`` messages, tool calls and results included."""
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.role.value
        if role in ("tool", "function"):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.additional_kwargs.get("tool_call_id", ""),
                    "content": m.content or "",
                }
            )
            continue
        msg: dict[str, Any] = {"role": "system" if role == "developer" else role}
        msg["content"] = m.content or ""
        calls = [
            {
                "id": b.tool_call_id,
                "type": "function",
                "function": {
                    "name": b.tool_name,
                    "arguments": b.tool_kwargs
                    if isinstance(b.tool_kwargs, str)
                    else json.dumps(b.tool_kwargs),
                },
            }
            for b in m.blocks
            if isinstance(b, ToolCallBlock)
        ] or m.additional_kwargs.get("tool_calls")
        if calls and role == "assistant":
            msg["tool_calls"] = calls
        out.append(msg)
    return out
