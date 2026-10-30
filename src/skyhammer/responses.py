"""The OpenAI Responses API for every model: ``llm.responses.create(...)`` (SI-24).

Bedrock serves the Responses API natively for some models: OpenAI GPT models on
``bedrock-runtime`` (``/openai/v1/responses``) and GPT-OSS on ``bedrock-mantle``
(``/v1/responses``). For those, Skyhammer sends your request as-is, so stored
conversations (``store``, ``previous_response_id``), retrieval and deletion are Bedrock's:
stored turns are kept 30 days in the Region that served them. Pass ``store=False`` to keep
nothing.

Every other model (Nova, Claude, Llama, your fine-tunes and endpoints) gets the same API
emulated over chat completions: the same request and response shapes, function tools,
structured output and streaming events. ``previous_response_id`` works for emulated
models too, but the conversation is kept in this process's memory only (``Response.
emulated`` is True), so it doesn't survive a restart or reach another machine.

Which route a model takes is found on its first call (native on ``bedrock-runtime``, then
``bedrock-mantle``, then emulated) and remembered for the process.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from skyhammer.errors import SkyhammerError
from skyhammer.inference import ChatCompletion, Usage
from skyhammer.mantle import MantleClient, MantleError

if TYPE_CHECKING:
    from skyhammer.llm import LLM

EMULATED_PREFIX = "resp_skyhammer_"
MAX_LOCAL_CONVERSATIONS = 1000
_PATHS = {"runtime": "/openai/v1/responses", "mantle": "/v1/responses"}
_GEO = re.compile(r"^(us|eu|apac|jp|au|ca|us-gov|global)\.")
_VERSION = re.compile(r"(-\d+)?:\d+$")
_RESPONSE_ID = re.compile(r"resp_[A-Za-z0-9_-]{1,256}")

# Params the emulation understands; anything else needs a native model.
_EMULATED_PARAMS = {
    "instructions",
    "max_output_tokens",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "text",
    "reasoning",
    "metadata",
    "store",
    "previous_response_id",
    "parallel_tool_calls",
    "guardrail",
    "user",
}


@dataclass
class ResponseUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


@dataclass
class Response:
    """A Responses API result; ``output_text`` joins the assistant's text."""

    id: str
    model: str
    status: str
    output: list[dict[str, Any]]
    usage: ResponseUsage
    created_at: float = field(default_factory=time.time)
    incomplete_details: dict[str, Any] | None = None
    emulated: bool = False  # conversation kept in this process, not by Bedrock
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def output_text(self) -> str:
        return "".join(
            part.get("text", "")
            for item in self.output
            if item.get("type") == "message"
            for part in item.get("content") or []
            if part.get("type") == "output_text"
        )

    @property
    def function_calls(self) -> list[dict[str, Any]]:
        """``[{"call_id", "name", "arguments"}]`` the model wants you to run."""
        return [item for item in self.output if item.get("type") == "function_call"]

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Response:
        u = data.get("usage") or {}
        return cls(
            id=data["id"],
            model=data.get("model", ""),
            status=data.get("status", ""),
            output=list(data.get("output") or []),
            usage=ResponseUsage(
                u.get("input_tokens", 0), u.get("output_tokens", 0), u.get("total_tokens", 0)
            ),
            created_at=data.get("created_at") or time.time(),
            incomplete_details=data.get("incomplete_details"),
            raw=data,
        )

    def to_json(self) -> dict[str, Any]:
        if self.raw:
            return self.raw
        return {
            "id": self.id,
            "object": "response",
            "model": self.model,
            "status": self.status,
            "created_at": self.created_at,
            "output": self.output,
            "incomplete_details": self.incomplete_details,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


class ResponseStream:
    """Iterator of Responses API events (dicts with a ``type``, e.g.
    ``response.output_text.delta``); ``final()`` returns the completed ``Response``."""

    def __init__(self, events: Iterator[dict[str, Any]], *, emulated: bool = False) -> None:
        self._events = events
        self._emulated = emulated
        self._final: Response | None = None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for event in self._events:
            if event.get("type") in ("response.completed", "response.incomplete"):
                self._final = Response.from_json(event["response"])
                self._final.emulated = self._emulated
            elif event.get("type") in ("response.failed", "error"):
                raise SkyhammerError(f"Response failed: {json.dumps(event)[:500]}")
            yield event

    def __enter__(self) -> ResponseStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def final(self) -> Response:
        for _ in self:
            pass
        if self._final is None:
            raise SkyhammerError("The stream ended without a completed response")
        return self._final


# ---- conversations kept in memory for emulated models ----


class _LocalStore:
    def __init__(self, limit: int = MAX_LOCAL_CONVERSATIONS) -> None:
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.limit = limit

    def put(self, response_id: str, entry: dict[str, Any]) -> None:
        with self._lock:
            self._items[response_id] = entry
            self._items.move_to_end(response_id)
            while len(self._items) > self.limit:
                self._items.popitem(last=False)

    def get(self, response_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._items.get(response_id)

    def delete(self, response_id: str) -> bool:
        with self._lock:
            return self._items.pop(response_id, None) is not None


LOCAL = _LocalStore()
_ROUTES: dict[tuple[str, str], tuple[str, str | None]] = {}  # (region, model) -> route


# ---- Responses <-> chat completions ----


def _text_of(content: Any) -> Any:
    """Responses content -> chat content (a string, or parts when there are images)."""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content or []:
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": part.get("text", "")})
        elif kind == "input_image":
            url = part.get("image_url") or part.get("url")
            if not url:
                raise SkyhammerError("input_image needs an image_url (data: URL or s3://)")
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            raise SkyhammerError(f"Content type {kind!r} needs a model with the native API")
    if all(p["type"] == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def input_to_messages(items: Any) -> list[dict[str, Any]]:
    """Responses ``input`` (a string or a list of items) -> chat messages."""
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    messages: list[dict[str, Any]] = []
    for item in items:
        kind = item.get("type", "message")
        if kind == "message":
            role = "system" if item["role"] == "developer" else item["role"]
            messages.append({"role": role, "content": _text_of(item.get("content"))})
        elif kind == "function_call":
            call = {
                "id": item["call_id"],
                "type": "function",
                "function": {"name": item["name"], "arguments": item.get("arguments", "")},
            }
            last = messages[-1] if messages else None
            if last and last["role"] == "assistant" and last.get("tool_calls") is not None:
                last["tool_calls"].append(call)  # parallel calls share one assistant turn
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind == "function_call_output":
            output = item.get("output")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "content": output if isinstance(output, str) else json.dumps(output),
                }
            )
        elif kind == "reasoning":
            continue  # earlier reasoning isn't replayed to chat models
        else:
            raise SkyhammerError(f"Input item type {kind!r} needs a model with the native API")
    return messages


def _chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in tools:
        if t.get("type") != "function":
            raise SkyhammerError(
                f"Tool type {t.get('type')!r} (server-side tools) needs a bedrock-mantle model "
                "with the native Responses API; emulated models take function tools"
            )
        fn = {k: t[k] for k in ("name", "description", "parameters", "strict") if k in t}
        out.append({"type": "function", "function": fn})
    return out


def _chat_tool_choice(choice: Any) -> Any:
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"type": "function", "function": {"name": choice["name"]}}
    return choice


def to_chat_params(params: dict[str, Any]) -> dict[str, Any]:
    """Responses params -> chat completion params (for emulated models)."""
    unknown = set(params) - _EMULATED_PARAMS
    if unknown:
        raise SkyhammerError(
            f"{', '.join(sorted(unknown))} needs a model with the native Responses API"
        )
    out: dict[str, Any] = {}
    if params.get("max_output_tokens") is not None:
        out["max_tokens"] = params["max_output_tokens"]
    for key in ("temperature", "top_p", "guardrail"):
        if params.get(key) is not None:
            out[key] = params[key]
    if params.get("tools"):
        out["tools"] = _chat_tools(params["tools"])
    if params.get("tool_choice") is not None:
        out["tool_choice"] = _chat_tool_choice(params["tool_choice"])
    fmt = (params.get("text") or {}).get("format")
    if fmt and fmt.get("type") == "json_schema":
        schema = {k: fmt[k] for k in ("name", "schema", "strict", "description") if k in fmt}
        out["response_format"] = {"type": "json_schema", "json_schema": schema}
    elif fmt and fmt.get("type") == "json_object":
        out["response_format"] = {"type": "json_object"}
    effort = (params.get("reasoning") or {}).get("effort")
    if effort:
        out["reasoning_effort"] = effort
    return out


def output_items(result: ChatCompletion) -> list[dict[str, Any]]:
    msg = result.choices[0].message
    items: list[dict[str, Any]] = []
    if msg.reasoning_content:
        items.append(
            {
                "type": "reasoning",
                "id": f"rs_{uuid.uuid4().hex}",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": msg.reasoning_content}],
            }
        )
    if msg.content:
        items.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": msg.content, "annotations": []}],
            }
        )
    for tc in msg.tool_calls or []:
        items.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex}",
                "call_id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
                "status": "completed",
            }
        )
    return items


def _assistant_message(result: ChatCompletion) -> dict[str, Any]:
    msg = result.choices[0].message
    out: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return out


def _emulated_response(response_id: str, model: str, result: ChatCompletion) -> Response:
    finish = result.choices[0].finish_reason
    u = result.usage
    return Response(
        id=response_id,
        model=model,
        status="incomplete" if finish == "length" else "completed",
        output=output_items(result),
        usage=ResponseUsage(u.prompt_tokens, u.completion_tokens, u.total_tokens, u.cost_usd),
        incomplete_details={"reason": "max_output_tokens"} if finish == "length" else None,
        emulated=True,
    )


# ---- routing ----


def _unsupported(e: MantleError) -> bool:
    """The endpoint doesn't serve this model with the Responses API (seen live: 404
    ``model_not_found`` on bedrock-runtime, 400 "does not support the '/v1/responses'
    API" or "isn't supported on this route" on bedrock-mantle)."""
    if e.status == 404 and e.code == "model_not_found":
        return True
    text = e.body.lower()
    return e.status == 400 and ("does not support" in text or "isn't supported on" in text)


def mantle_model_id(model_id: str, available: list[str]) -> str | None:
    """The bedrock-mantle ID for a runtime model ID (``openai.gpt-oss-20b-1:0`` ->
    ``openai.gpt-oss-20b``), if mantle lists it."""
    if model_id in available:
        return model_id
    bare = _VERSION.sub("", _GEO.sub("", model_id))
    return bare if bare in available else None


class Responses:
    """``llm.responses``: ``create``, ``retrieve``, ``delete`` and ``cancel``."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    # -- plumbing --

    @property
    def _region(self) -> str:
        ref = self._llm.model_ref
        return ref.region if ref else self._llm.session.region

    def _client(self, endpoint: str) -> MantleClient:
        return MantleClient(self._llm.session, self._region, endpoint=endpoint)

    def _candidates(self) -> list[tuple[str, str | None]]:
        """Routes to try, in order, for this model."""
        llm = self._llm
        ref = llm.model_ref
        if ref and ref.api == "mantle":
            return [("mantle", ref.model_arn), ("emulated", None)]
        if ref:
            return [("emulated", None)]  # custom, imported and SageMaker-served fine-tunes
        model_id = llm.model_id
        return [("runtime", model_id), ("mantle", model_id), ("emulated", None)]

    def _mantle_id(self, model_id: str) -> str | None:
        try:
            listing = self._client("mantle").request("GET", "/v1/models")
        except (MantleError, OSError):
            return None  # no bedrock-mantle in this Region
        return mantle_model_id(model_id, [m["id"] for m in listing.get("data", [])])

    def _native(
        self, endpoint: str, model_id: str, body: dict[str, Any], stream: bool
    ) -> Response | ResponseStream:
        headers: dict[str, str] = {}
        guardrail = body.pop("guardrail", None)
        if guardrail:
            if endpoint != "runtime":
                raise SkyhammerError("Guardrails on bedrock-mantle Responses aren't supported")
            headers["X-Amzn-Bedrock-GuardrailIdentifier"] = guardrail["id"]
            headers["X-Amzn-Bedrock-GuardrailVersion"] = str(guardrail.get("version", "DRAFT"))
        body = {**body, "model": model_id}
        client = self._client(endpoint)
        if stream:
            events = client.stream(
                "POST", _PATHS[endpoint], json_body={**body, "stream": True}, headers=headers
            )
            first = next(events)  # surface errors (and unsupported models) before returning
            return ResponseStream(self._usage_events(model_id, _chain(first, events)))
        raw = client.request("POST", _PATHS[endpoint], json_body=body, headers=headers)
        resp = Response.from_json(raw)
        self._record(model_id, resp.usage, native=endpoint)
        return resp

    def _usage_events(
        self, model_id: str, events: Iterator[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        for event in events:
            if event.get("type") in ("response.completed", "response.incomplete"):
                self._record(model_id, Response.from_json(event["response"]).usage, "runtime")
            yield event

    def _record(self, model_id: str, usage: ResponseUsage, native: str | None) -> None:
        if native == "mantle":
            self._llm.session.spend.record(model_id, None)  # not in the Price List
            return
        u = Usage(usage.input_tokens, usage.output_tokens, usage.total_tokens)
        self._llm._record_usage(model_id, u, None)
        usage.cost_usd = u.cost_usd

    # -- emulation --

    def _emulate(self, body: dict[str, Any], stream: bool) -> Response | ResponseStream:
        params = {k: v for k, v in body.items() if k != "input"}
        chat_params = to_chat_params(params)
        history: list[dict[str, Any]] = []
        prev = params.get("previous_response_id")
        if prev:
            entry = LOCAL.get(prev)
            if entry is None:
                raise SkyhammerError(
                    f"No conversation {prev!r} in this process: emulated models keep "
                    "previous_response_id state in memory only"
                )
            history = list(entry["messages"])
        turn = input_to_messages(body.get("input") or [])
        system = (
            [{"role": "system", "content": params["instructions"]}]
            if params.get("instructions")
            else []
        )
        messages = system + history + turn
        response_id = f"{EMULATED_PREFIX}{uuid.uuid4().hex}"
        model = self._llm.model_ref.name if self._llm.model_ref else self._llm.name
        store = params.get("store", True)

        def finish(result: ChatCompletion) -> Response:
            response = _emulated_response(response_id, model, result)
            if store:
                messages_after = history + turn + [_assistant_message(result)]
                LOCAL.put(response_id, {"messages": messages_after, "response": response})
            return response

        chat = self._llm.chat.completions
        if not stream:
            return finish(chat.create(messages=messages, **chat_params))

        def events() -> Iterator[dict[str, Any]]:
            yield {
                "type": "response.created",
                "response": {"id": response_id, "model": model, "status": "in_progress"},
            }
            s = chat.create(messages=messages, stream=True, **chat_params)
            for chunk in s:
                for ch in chunk.choices:
                    if ch.delta.content:
                        yield {"type": "response.output_text.delta", "delta": ch.delta.content}
            final = finish(s.final())
            yield {"type": f"response.{final.status}", "response": final.to_json()}

        return ResponseStream(events(), emulated=True)

    # -- public API --

    def create(
        self, *, input: Any = None, stream: bool = False, **params: Any
    ) -> Response | ResponseStream:
        """Create a response. ``input`` is a string or a list of input items; params follow
        the OpenAI Responses API (``instructions``, ``previous_response_id``, ``store``,
        ``tools``, ``text``, ``reasoning``, ``max_output_tokens``...). ``guardrail=
        {"id", "version"}`` applies a Bedrock guardrail. ``stream=True`` returns a
        ``ResponseStream`` of events."""
        session = self._llm.session
        session.spend.check()
        body = {"input": input, **params}
        if body["input"] is None:
            body.pop("input")
        prev = params.get("previous_response_id")
        if prev and str(prev).startswith(EMULATED_PREFIX):
            return self._emulate(body, stream)
        key = (
            self._region,
            self._llm.model_ref.model_arn if self._llm.model_ref else self._llm.model_id,
        )
        candidates = [_ROUTES[key]] if key in _ROUTES else self._candidates()
        for endpoint, model_id in candidates:
            if endpoint == "emulated":
                _ROUTES[key] = (endpoint, None)
                return self._emulate(body, stream)
            assert model_id is not None
            target = model_id
            if endpoint == "mantle" and not self._llm.model_ref and key not in _ROUTES:
                found = self._mantle_id(model_id)
                if found is None:
                    continue
                target = found
            try:
                result = self._native(endpoint, target, dict(body), stream)
            except MantleError as e:
                if key not in _ROUTES and _unsupported(e):
                    continue
                raise
            _ROUTES[key] = (endpoint, target)
            return result
        raise SkyhammerError(f"No route serves the Responses API for {key[1]}")  # unreachable

    def _native_route(self) -> tuple[str, str | None] | None:
        key = (
            self._region,
            self._llm.model_ref.model_arn if self._llm.model_ref else self._llm.model_id,
        )
        route = _ROUTES.get(key)
        return route if route and route[0] != "emulated" else None

    def _by_id(self, method: str, response_id: str, suffix: str = "") -> dict[str, Any]:
        if not _RESPONSE_ID.fullmatch(response_id):  # it becomes part of a signed URL path
            raise SkyhammerError(f"Not a response ID: {response_id[:80]!r}")
        route = self._native_route()
        endpoints = [route[0]] if route else ["runtime", "mantle"]
        last: MantleError | None = None
        for endpoint in endpoints:
            try:
                return self._client(endpoint).request(
                    method, f"{_PATHS[endpoint]}/{response_id}{suffix}"
                )
            except MantleError as e:
                if e.status != 404:
                    raise
                last = e
        raise SkyhammerError(f"No stored response {response_id!r}") from last

    def retrieve(self, response_id: str) -> Response:
        """A stored response (native: kept by Bedrock 30 days; emulated: this process)."""
        if response_id.startswith(EMULATED_PREFIX):
            entry = LOCAL.get(response_id)
            if entry is None:
                raise SkyhammerError(f"No stored response {response_id!r} in this process")
            response: Response = entry["response"]
            return response
        return Response.from_json(self._by_id("GET", response_id))

    def delete(self, response_id: str) -> None:
        if response_id.startswith(EMULATED_PREFIX):
            if not LOCAL.delete(response_id):
                raise SkyhammerError(f"No conversation {response_id!r} in this process")
            return
        self._by_id("DELETE", response_id)

    def cancel(self, response_id: str) -> Response:
        """Cancel a background response (bedrock-mantle ``background=True``)."""
        if response_id.startswith(EMULATED_PREFIX):
            raise SkyhammerError("Emulated responses are synchronous; nothing to cancel")
        return Response.from_json(self._by_id("POST", response_id, "/cancel"))


def _chain(first: dict[str, Any], rest: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    yield first
    yield from rest
