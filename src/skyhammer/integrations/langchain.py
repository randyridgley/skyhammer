"""LangChain chat model: ``ChatSkyhammer(model="support@prod")``.

Works anywhere LangChain takes a chat model: ``invoke``, ``stream``, ``batch``, their async
forms, ``bind_tools`` (tool-calling agents, LangGraph) and ``with_structured_output``.
``response_metadata`` carries the model that answered (``variant`` under a traffic split)
and ``cost_usd``; ``usage_metadata`` carries tokens.

Needs ``pip install "skyhammer[langchain]"``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import ToolCall as LCToolCall
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr

from skyhammer.inference import ChatCompletion, Usage
from skyhammer.integrations._openai import compact, metadata
from skyhammer.target import Target


class ChatSkyhammer(BaseChatModel):
    """A Skyhammer model as a LangChain chat model.

    ``model`` is a name (``"nova-micro"``, a fine-tune, ``"support@prod"``), an ``LLM`` or a
    ``TrafficSplit``. ``model_kwargs`` passes other ``chat.completions.create`` arguments
    (``guardrail=``, ``reasoning_effort=``, ``service_tier=``...)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    session: Any = Field(default=None, exclude=True)
    _target: Target = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._target = Target(self.model, self.session)

    @property
    def _llm_type(self) -> str:
        return "skyhammer"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self._target.name, **self._params(None)}

    def _params(self, stop: list[str] | None, **kwargs: Any) -> dict[str, Any]:
        return compact(
            {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stop": stop or self.stop,
                **self.model_kwargs,
                **kwargs,
            }
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result: ChatCompletion = self._target.create(
            messages=convert_to_openai_messages(messages), **self._params(stop, **kwargs)
        )
        msg = result.choices[0].message
        tool_calls: list[LCToolCall] = [
            {
                "name": c.function.name,
                "args": json.loads(c.function.arguments or "{}"),
                "id": c.id,
                "type": "tool_call",
            }
            for c in msg.tool_calls or []
        ]
        ai = AIMessage(
            content=msg.content or "",
            tool_calls=tool_calls,
            usage_metadata=_usage(result.usage),
            response_metadata=metadata(result),
            id=result.id,
        )
        if msg.reasoning_content:
            ai.additional_kwargs["reasoning_content"] = msg.reasoning_content
        return ChatResult(generations=[ChatGeneration(message=ai)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        stream = self._target.create(
            messages=convert_to_openai_messages(messages),
            stream=True,
            **self._params(stop, **kwargs),
        )
        for chunk in stream:
            meta: dict[str, Any] = {}
            text, calls = "", []
            for choice in chunk.choices:
                text += choice.delta.content or ""
                calls += [
                    ToolCallChunk(
                        name=d.function.name, args=d.function.arguments, id=d.id, index=d.index
                    )
                    for d in choice.delta.tool_calls or []
                ]
                if choice.finish_reason:
                    meta["finish_reason"] = choice.finish_reason
            if chunk.usage:
                meta["cost_usd"] = chunk.usage.cost_usd
                if stream.variant:
                    meta["variant"] = stream.variant
            message = AIMessageChunk(
                content=text,
                tool_call_chunks=calls,
                usage_metadata=_usage(chunk.usage) if chunk.usage else None,
                response_metadata=meta,
            )
            out = ChatGenerationChunk(message=message)
            if run_manager and text:
                run_manager.on_llm_new_token(text, chunk=out)
            yield out

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | dict[str, Any] | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Offer ``tools`` (LangChain tools, functions, pydantic models or OpenAI dicts).
        ``tool_choice`` is "auto", "any"/"required", "none" or a tool name."""
        bound: dict[str, Any] = {"tools": [convert_to_openai_tool(t) for t in tools]}
        if tool_choice is not None and tool_choice is not False:
            if tool_choice is True or tool_choice == "any":
                tool_choice = "required"
            elif isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            bound["tool_choice"] = tool_choice
        return super().bind(**bound, **kwargs)


def _usage(usage: Usage) -> UsageMetadata:
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
