"""Langfuse: trace every model call, and send eval scores, to Langfuse.

- ``traced(model)`` wraps a model (a name, ``LLM``, environment such as ``support@prod``,
  or traffic split) so each ``chat.completions.create`` is recorded as a Langfuse
  generation: messages, answer, tokens, Skyhammer's cost (in Langfuse's cost column),
  latency, the variant that answered under a split, and errors. Streams are recorded
  when they finish, with time to first token.
- ``log_eval(result, records)`` records each eval sample as a trace with one score per
  grader (judge explanations as score comments), so Langfuse can filter and compare runs.

The client comes from ``Langfuse()`` (``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``,
``LANGFUSE_HOST``) unless you pass ``client=``. Needs ``pip install "skyhammer[langfuse]"``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langfuse import Langfuse, get_client

from skyhammer.inference import ChatCompletion, ChatCompletionChunk, Stream
from skyhammer.integrations._eval import eval_rows
from skyhammer.integrations._openai import metadata, tool_call_dicts
from skyhammer.target import Target

if TYPE_CHECKING:
    from skyhammer.environments import Environment
    from skyhammer.evaluation import EvalReport, EvalResult
    from skyhammer.llm import LLM
    from skyhammer.session import Session
    from skyhammer.traffic import TrafficSplit

# Call parameters worth showing next to a generation; the rest (tools, schemas) are large.
_PARAMS = ("max_tokens", "temperature", "top_p", "stop", "reasoning_effort", "service_tier")


def _client(client: Langfuse | None) -> Langfuse:
    return client if client is not None else get_client()


def usage_details(result: ChatCompletion) -> dict[str, int]:
    u = result.usage
    out = {"input": u.prompt_tokens, "output": u.completion_tokens, "total": u.total_tokens}
    if u.cache_read_tokens:
        out["cache_read_input_tokens"] = u.cache_read_tokens
    if u.cache_write_tokens:
        out["cache_creation_input_tokens"] = u.cache_write_tokens
    return out


def _output(result: ChatCompletion) -> dict[str, Any]:
    msg = result.choices[0].message if result.choices else None
    out: dict[str, Any] = {"role": "assistant", "content": msg.content if msg else None}
    if msg and msg.tool_calls:
        out["tool_calls"] = tool_call_dicts(msg.tool_calls)
    if msg and msg.reasoning_content:
        out["reasoning_content"] = msg.reasoning_content
    return out


def _finish(generation: Any, result: ChatCompletion) -> None:
    update: dict[str, Any] = {
        "output": _output(result),
        "model": result.variant or result.model,
        "usage_details": usage_details(result),
        "metadata": metadata(result),
    }
    if result.usage.cost_usd is not None:
        update["cost_details"] = {"total": result.usage.cost_usd}
    generation.update(**update)


class _Completions:
    def __init__(self, traced: traced) -> None:
        self._traced = traced

    def create(self, *, messages: list[dict[str, Any]], stream: bool = False, **params: Any) -> Any:
        """``chat.completions.create`` on the wrapped model, recorded as a generation."""
        t = self._traced
        generation = _client(t.client).start_observation(
            name=t.name,
            as_type="generation",
            input=messages,
            model=t.target.name,
            model_parameters={k: params[k] for k in _PARAMS if params.get(k) is not None},
            metadata=t.metadata or None,
        )
        try:
            result = t.target.create(messages=messages, stream=stream, **params)
        except BaseException as e:
            generation.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
            generation.end()
            raise
        if not stream:
            _finish(generation, result)
            generation.end()
            return result
        return _traced_stream(generation, result)


def _traced_stream(generation: Any, inner: Stream) -> Stream:
    def chunks() -> Iterator[ChatCompletionChunk]:
        first = True
        try:
            for chunk in inner:
                if first:
                    generation.update(completion_start_time=datetime.now(UTC))
                    first = False
                yield chunk
            done = inner.final()
            if inner.variant:
                done.variant = inner.variant
            _finish(generation, done)
        except Exception as e:
            generation.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
            raise
        finally:  # also when the caller stops reading early
            generation.end()

    out = Stream(inner.model_id, chunks())
    out.variant = inner.variant
    return out


class _Chat:
    def __init__(self, traced: traced) -> None:
        self.completions = _Completions(traced)


class traced:
    """``model`` with every ``chat.completions.create`` recorded in Langfuse.

    ``name`` labels the generations (default ``"skyhammer.chat"``); ``metadata`` is attached
    to each one. Other attributes pass through to the model, so a traced ``LLM`` still has
    ``model_id``, ``deploy()`` and so on."""

    def __init__(
        self,
        model: str | LLM | Environment | TrafficSplit,
        *,
        client: Langfuse | None = None,
        name: str = "skyhammer.chat",
        metadata: dict[str, Any] | None = None,
        session: Session | None = None,
    ) -> None:
        self.target = Target(model, session)
        self.client = client
        self.name = name
        self.metadata = dict(metadata or {})
        self.chat = _Chat(self)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.target.resolve(), attr)

    def __repr__(self) -> str:
        return f"traced({self.target.name!r})"


def log_eval(
    result: EvalResult | EvalReport,
    records: list[dict[str, Any]] | None = None,
    *,
    client: Langfuse | None = None,
    name: str = "skyhammer.eval",
    run: str | None = None,
) -> list[str]:
    """One trace per sample (prompt, output, reference) with a score per grader; returns
    the trace IDs. ``run`` tags every trace (for example "support-v2 vs base") so the
    run can be filtered in Langfuse. Pass ``records`` with a local result to include each
    prompt and reference answer. Call ``client.flush()`` before a short script exits."""
    lf = _client(client)
    model = getattr(result, "model", None)
    meta = {k: v for k, v in {"model": model, "run": run}.items() if v}
    ids = []
    for i, row in enumerate(eval_rows(result, records)):
        span = lf.start_observation(
            name=name,
            as_type="evaluator",
            input=row.prompt,
            output=row.output,
            metadata={
                **meta,
                "sample": i,
                **({"expected": row.expected} if row.expected is not None else {}),
                **({"category": row.category} if row.category else {}),
            },
        )
        for metric, value in row.scores.items():
            if value is None:
                continue
            comment = row.comments.get(metric)
            span.score_trace(name=metric, value=float(value), comment=comment)
        span.end()
        ids.append(span.trace_id)
    return ids
