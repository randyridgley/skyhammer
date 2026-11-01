"""LiteLLM provider: ``register()``, then ``litellm.completion(model="skyhammer/<name>")``.

``<name>`` is anything ``LLM`` takes: ``skyhammer/nova-micro``, ``skyhammer/support-qwen-v2``
or ``skyhammer/support@prod`` (traffic splits included). Sync, async and streaming calls,
tools and ``response_format`` work, so a LiteLLM Router or proxy can serve your
fine-tunes next to other providers. Each response's ``_hidden_params`` carries
``cost_usd`` and, under a split, the ``variant`` that answered.

Needs ``pip install "skyhammer[litellm]"``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

import litellm
from litellm import ModelResponse
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import GenericStreamingChunk

from skyhammer.inference import ChatCompletion
from skyhammer.integrations._openai import metadata, tool_call_dicts, usage_dict
from skyhammer.target import Target

if TYPE_CHECKING:
    from skyhammer.session import Session

PROVIDER = "skyhammer"
# OpenAI parameters LiteLLM may forward that ``chat.completions.create`` accepts.
_PASSED = {
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "tools",
    "tool_choice",
    "response_format",
    "reasoning_effort",
    "service_tier",
    "extra_body",
    "guardrail",
    "latency",
}


class SkyhammerProvider(CustomLLM):
    """Serves ``skyhammer/<name>`` models; one ``Target`` per name, shared by all calls."""

    def __init__(self, session: Session | None = None) -> None:
        super().__init__()
        self.session = session
        self._targets: dict[str, Target] = {}
        self._lock = threading.Lock()

    def target(self, model: str) -> Target:
        name = model.removeprefix(f"{PROVIDER}/")
        with self._lock:
            if name not in self._targets:
                self._targets[name] = Target(name, self.session)
            return self._targets[name]

    def _create(self, model: str, messages: list[Any], optional_params: dict[str, Any]) -> Any:
        params = {k: v for k, v in optional_params.items() if k in _PASSED and v is not None}
        if "max_completion_tokens" in params:
            params.setdefault("max_tokens", params.pop("max_completion_tokens"))
        stream = bool(optional_params.get("stream"))
        return self.target(model).create(messages=messages, stream=stream, **params)

    def completion(
        self, model: str, messages: list[Any], *args: Any, **kwargs: Any
    ) -> ModelResponse:
        result: ChatCompletion = self._create(model, messages, kwargs.get("optional_params", {}))
        return to_model_response(result, model)

    def streaming(
        self, model: str, messages: list[Any], *args: Any, **kwargs: Any
    ) -> Iterator[GenericStreamingChunk]:
        params = {**kwargs.get("optional_params", {}), "stream": True}
        stream = self._create(model, messages, params)
        for chunk in stream:
            text, finish, call = "", "", None
            for choice in chunk.choices:
                text += choice.delta.content or ""
                finish = choice.finish_reason or finish
                for d in choice.delta.tool_calls or []:
                    call = {
                        "id": d.id,
                        "type": "function",
                        "function": {"name": d.function.name, "arguments": d.function.arguments},
                        "index": d.index,
                    }
            yield {
                "text": text,
                "tool_use": call,  # type: ignore[typeddict-item]
                "is_finished": bool(finish),
                "finish_reason": finish,
                "usage": usage_dict(chunk.usage) if chunk.usage else None,  # type: ignore[typeddict-item]
                "index": 0,
            }

    async def acompletion(
        self, model: str, messages: list[Any], *args: Any, **kwargs: Any
    ) -> ModelResponse:
        return await asyncio.to_thread(self.completion, model, messages, *args, **kwargs)

    async def astreaming(  # type: ignore[override]  # an async generator, as LiteLLM's docs
        self, model: str, messages: list[Any], *args: Any, **kwargs: Any
    ) -> AsyncIterator[GenericStreamingChunk]:
        chunks = await asyncio.to_thread(self.streaming, model, messages, *args, **kwargs)
        while (item := await asyncio.to_thread(next, chunks, None)) is not None:
            yield item


def to_model_response(result: ChatCompletion, model: str) -> ModelResponse:
    choices = [
        {
            "index": c.index,
            "finish_reason": c.finish_reason,
            "message": {
                "role": "assistant",
                "content": c.message.content,
                **(
                    {"tool_calls": tool_call_dicts(c.message.tool_calls)}
                    if c.message.tool_calls
                    else {}
                ),
            },
        }
        for c in result.choices
    ]
    response = ModelResponse(
        id=result.id,
        created=result.created,
        model=model,
        choices=choices,
        usage=usage_dict(result.usage),
    )
    response._hidden_params.update(metadata(result))
    return response


def register(session: Session | None = None) -> SkyhammerProvider:
    """Add the ``skyhammer/`` provider to LiteLLM (once; calling again replaces it)."""
    provider = SkyhammerProvider(session)
    litellm.custom_provider_map = [
        p for p in litellm.custom_provider_map if p.get("provider") != PROVIDER
    ] + [{"provider": PROVIDER, "custom_handler": provider}]
    litellm.utils.custom_llm_setup()  # type: ignore[no-untyped-call]
    return provider
