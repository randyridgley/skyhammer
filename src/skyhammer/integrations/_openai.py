"""OpenAI-shaped dicts from Skyhammer results, shared by the framework adapters."""

from __future__ import annotations

from typing import Any

from skyhammer.inference import ChatCompletion, ToolCall, Usage


def tool_call_dicts(calls: list[ToolCall] | None) -> list[dict[str, Any]]:
    return [
        {
            "id": c.id,
            "type": "function",
            "function": {"name": c.function.name, "arguments": c.function.arguments},
        }
        for c in calls or []
    ]


def usage_dict(usage: Usage | None) -> dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def metadata(result: ChatCompletion) -> dict[str, Any]:
    """What the frameworks keep next to a response: model, finish reason, cost, variant."""
    out: dict[str, Any] = {
        "model": result.model,
        "finish_reason": result.choices[0].finish_reason if result.choices else None,
        "cost_usd": result.usage.cost_usd,
    }
    if result.variant:
        out["variant"] = result.variant
    return out


def compact(params: dict[str, Any]) -> dict[str, Any]:
    """Drop unset parameters: Skyhammer rejects unknown ones rather than ignoring them."""
    return {k: v for k, v in params.items() if v is not None}
