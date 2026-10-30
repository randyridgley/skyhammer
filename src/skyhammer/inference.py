"""OpenAI-compatible chat over Bedrock Converse (any Bedrock model) and InvokeModel
(imported models, which accept OpenAI ChatCompletion bodies).

``chat.completions.create`` accepts the OpenAI parameters below and maps them onto Bedrock
natively; anything else raises instead of being silently dropped (use ``extra_body`` to pass
model-specific fields through ``additionalModelRequestFields``).

    messages, max_tokens, temperature, top_p, stop, stream,
    tools, tool_choice, response_format, reasoning_effort, service_tier,
    guardrail={"id", "version", "trace"?}, latency="optimized", extra_body

Responses are OpenAI-shaped. ``message.reasoning_content`` carries model reasoning (Bedrock
``reasoningContent`` blocks, or ``<think>`` text split out of the answer). ``.raw`` holds the
original Bedrock response.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import re
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from botocore.exceptions import ClientError

from skyhammer.formats import message_text

log = logging.getLogger(__name__)

_STOP_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
}

_SERVICE_TIERS = {
    "priority": "priority",
    "default": "default",
    "standard": "default",
    "flex": "flex",
    "reserved": "reserved",
}

_PARAMS = {
    "max_tokens",
    "temperature",
    "top_p",
    "stop",
    "tools",
    "tool_choice",
    "response_format",
    "reasoning_effort",
    "service_tier",
    "guardrail",
    "latency",
    "extra_body",
    "max_completion_tokens",
    "prompt_cache",
}

_THINK = re.compile(r"^\s*<(think|reasoning)>(.*?)</\1>\s*", re.S)


# ---------------------------------------------------------------- response types


@dataclass
class FunctionCall:
    name: str
    arguments: str  # JSON string, as in the OpenAI API


@dataclass
class ToolCall:
    id: str
    function: FunctionCall
    type: str = "function"


@dataclass
class Message:
    role: str
    content: str | None
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None

    def parsed(self) -> Any:
        """The JSON value in ``content`` (with ``response_format``). Tolerates text around the
        JSON, which some models emit even under constrained decoding."""
        return extract_json(self.content or "")


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: str


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None


@dataclass
class ChatCompletion:
    id: str
    model: str
    choices: list[Choice]
    usage: Usage
    created: int = field(default_factory=lambda: int(time.time()))
    object: str = "chat.completion"
    service_tier: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class FunctionCallDelta:
    name: str | None = None
    arguments: str | None = None


@dataclass
class ToolCallDelta:
    index: int
    id: str | None = None
    function: FunctionCallDelta = field(default_factory=FunctionCallDelta)
    type: str | None = None


@dataclass
class Delta:
    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None
    reasoning_content: str | None = None


@dataclass
class ChunkChoice:
    index: int
    delta: Delta
    finish_reason: str | None = None


@dataclass
class ChatCompletionChunk:
    id: str
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None
    created: int = field(default_factory=lambda: int(time.time()))
    object: str = "chat.completion.chunk"


def extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text[i:])[0]
            except ValueError:
                continue
    raise ValueError(f"No JSON value found in model output: {text[:200]!r}")


# ---------------------------------------------------------------- request mapping

# Hidden tool used to emulate json_schema output on models without outputConfig support.
JSON_TOOL = "skyhammer_json_response"


CACHE_POINT = {"cachePoint": {"type": "default"}}


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    """OpenAI content (string or parts) -> Converse content blocks. Prompt caching:
    ``{"type": "cache_point"}`` parts, or Anthropic-style ``cache_control`` on a text part."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content:
        kind = part.get("type", "text")
        if kind == "text":
            blocks.append({"text": part.get("text", "")})
            if part.get("cache_control"):
                blocks.append(dict(CACHE_POINT))
        elif kind == "cache_point":
            blocks.append(dict(CACHE_POINT))
        elif kind == "image_url":
            url = (
                part["image_url"]["url"]
                if isinstance(part["image_url"], dict)
                else part["image_url"]
            )
            m = re.match(r"data:image/(\w+);base64,(.+)", url, re.S)
            if not m:
                raise ValueError("image_url must be a base64 data URL (data:image/png;base64,...)")
            fmt = "jpeg" if m.group(1) == "jpg" else m.group(1)
            blocks.append(
                {"image": {"format": fmt, "source": {"bytes": base64.b64decode(m.group(2))}}}
            )
        elif kind == "file":
            blocks.append(_document_block(part["file"]))
        else:
            raise ValueError(f"Unsupported content part type {kind!r}")
    return blocks


_DOC_FORMATS = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/html": "html",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
}


def _document_block(f: dict[str, Any]) -> dict[str, Any]:
    """OpenAI ``{"type": "file", "file": {"file_data": "data:<mime>;base64,...", "filename"}}``
    -> Converse document block."""
    m = re.match(r"data:([\w/.+-]+);base64,(.+)", f.get("file_data", ""), re.S)
    if not m or m.group(1) not in _DOC_FORMATS:
        raise ValueError(
            "file parts need a base64 data URL of a PDF, DOCX, XLSX, CSV, TXT, MD or HTML document"
        )
    stem = re.sub(
        r"[^A-Za-z0-9 \-()\[\]]", "-", (f.get("filename") or "document").rsplit(".", 1)[0]
    )
    return {
        "document": {
            "format": _DOC_FORMATS[m.group(1)],
            "name": stem[:200] or "document",
            "source": {"bytes": base64.b64decode(m.group(2))},
        }
    }


def _tool_result_block(m: dict[str, Any]) -> dict[str, Any]:
    raw = m.get("content")
    text = message_text(raw) if not isinstance(raw, str) else raw
    try:
        value = json.loads(text)
        content = [{"json": value}] if isinstance(value, dict) else [{"text": text}]
    except (TypeError, ValueError):
        content = [{"text": text}]
    return {"toolResult": {"toolUseId": m["tool_call_id"], "content": content}}


def _converse_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "tool":
            role, blocks = "user", [_tool_result_block(m)]
        elif role == "assistant":
            blocks = _content_blocks(m.get("content"))
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                args = fn.get("arguments") or "{}"
                blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": tc["id"],
                            "name": fn["name"],
                            "input": json.loads(args) if isinstance(args, str) else args,
                        }
                    }
                )
        else:
            blocks = _content_blocks(m.get("content"))
        if out and out[-1]["role"] == role:  # Converse requires alternating roles
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})
    return out


def strict_schema(schema: Any) -> Any:
    """Set ``additionalProperties: false`` on object schemas that don't specify it (OpenAI
    strict-mode semantics; Claude's structured outputs reject schemas without it)."""
    if isinstance(schema, list):
        return [strict_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: strict_schema(v) for k, v in schema.items()}
    if out.get("type") == "object" or "properties" in out:
        out.setdefault("additionalProperties", False)
    return out


_KEYWORDS_REJECTED = re.compile(r"properties ([\w, ]+?) (?:are|is) not supported")


def relax_schema(schema: Any, keywords: set[str]) -> Any:
    """Remove JSON-schema keywords a model rejects (e.g. Claude: minimum/maximum on integers),
    keeping each constraint as text in the property's description so the model still sees it."""
    if isinstance(schema, list):
        return [relax_schema(x, keywords) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: relax_schema(v, keywords) for k, v in schema.items() if k not in keywords}
    dropped = {k: schema[k] for k in keywords if k in schema}
    if dropped:
        note = "; ".join(f"{k}: {json.dumps(v)}" for k, v in dropped.items())
        out["description"] = f"{out.get('description', '')} (constraints: {note})".strip()
    return out


def recover_text_tool_call(text: str, tool_names: set[str]) -> ToolCall | None:
    """Some models (Llama 3.x) emit a tool call as JSON text instead of a toolUse block.
    Accept it only if it is exactly one JSON object naming a provided tool."""
    stripped = text.strip()
    if not tool_names or not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except ValueError:
        return None
    if not isinstance(obj, dict) or obj.get("name") not in tool_names:
        return None
    args = obj.get("parameters", obj.get("arguments", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return ToolCall(f"call_{uuid.uuid4().hex[:12]}", FunctionCall(obj["name"], json.dumps(args)))


def _tool_config(
    tools: list[dict[str, Any]] | None, tool_choice: Any, messages: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not tools:
        return None
    history_has_tools = any(m.get("role") == "tool" or m.get("tool_calls") for m in messages)
    if tool_choice == "none" and not history_has_tools:
        return None  # Converse has no "none"; omitting tools is equivalent
    specs = []
    for t in tools:
        fn = t["function"] if t.get("type", "function") == "function" else t
        specs.append(
            {
                "toolSpec": {
                    "name": fn["name"],
                    "description": fn.get("description") or fn["name"],
                    "inputSchema": {
                        "json": fn.get("parameters") or {"type": "object", "properties": {}}
                    },
                }
            }
        )
    cfg: dict[str, Any] = {"tools": specs}
    if tool_choice in (None, "auto", "none"):
        pass
    elif tool_choice == "required":
        cfg["toolChoice"] = {"any": {}}
    elif isinstance(tool_choice, dict):
        cfg["toolChoice"] = {"tool": {"name": tool_choice["function"]["name"]}}
    else:
        raise ValueError(f"Unsupported tool_choice {tool_choice!r}")
    return cfg


def to_converse_request(
    model_id: str, messages: list[dict[str, Any]], **params: Any
) -> dict[str, Any]:
    unknown = set(params) - _PARAMS
    if unknown:
        raise TypeError(
            f"Unsupported parameter(s): {', '.join(sorted(unknown))}. "
            "Pass model-specific fields with extra_body={...}."
        )
    req: dict[str, Any] = {"modelId": model_id, "messages": _converse_messages(messages)}
    system: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "system":
            system += (
                _content_blocks(m["content"])
                if isinstance(m["content"], list)
                else [{"text": message_text(m["content"])}]
            )

    response_format = params.get("response_format")
    output: dict[str, Any] = {}
    if response_format:
        rtype = response_format.get("type")
        if rtype == "json_schema":
            js = response_format["json_schema"]
            schema: dict[str, Any] = {
                "schema": json.dumps(strict_schema(js["schema"])),
                "name": js.get("name", "response"),
            }
            if js.get("description"):
                schema["description"] = js["description"]
            output["textFormat"] = {"type": "json_schema", "structure": {"jsonSchema": schema}}
        elif rtype == "json_object":
            system.append({"text": "Respond with a single valid JSON object and nothing else."})
        elif rtype != "text":
            raise ValueError(f"Unsupported response_format type {rtype!r}")
    if params.get("reasoning_effort"):
        output["effort"] = params["reasoning_effort"]
    if output:
        req["outputConfig"] = output
    if params.get("prompt_cache") == "system" and system:
        system.append(dict(CACHE_POINT))  # cache everything up to the end of the system prompt
    elif params.get("prompt_cache") not in (None, "system"):
        raise ValueError("prompt_cache must be 'system' (or use cache_point content parts)")
    if system:
        req["system"] = system

    cfg: dict[str, Any] = {}
    max_tokens = params.get("max_tokens") or params.get("max_completion_tokens")
    if max_tokens is not None:
        cfg["maxTokens"] = max_tokens
    if params.get("temperature") is not None:
        cfg["temperature"] = params["temperature"]
    if params.get("top_p") is not None:
        cfg["topP"] = params["top_p"]
    stop = params.get("stop")
    if stop is not None:
        cfg["stopSequences"] = [stop] if isinstance(stop, str) else stop
    if cfg:
        req["inferenceConfig"] = cfg

    tool_config = _tool_config(params.get("tools"), params.get("tool_choice"), messages)
    if tool_config:
        req["toolConfig"] = tool_config
    tier = params.get("service_tier")
    if tier and tier != "auto":
        if tier not in _SERVICE_TIERS:
            raise ValueError(f"service_tier must be one of {sorted(_SERVICE_TIERS)} or 'auto'")
        req["serviceTier"] = {"type": _SERVICE_TIERS[tier]}
    if params.get("latency"):
        req["performanceConfig"] = {"latency": params["latency"]}
    if params.get("guardrail"):
        g = params["guardrail"]
        req["guardrailConfig"] = {
            "guardrailIdentifier": g["id"],
            "guardrailVersion": str(g.get("version", "DRAFT")),
        }
        if g.get("trace"):
            req["guardrailConfig"]["trace"] = "enabled"
    if params.get("extra_body"):
        req["additionalModelRequestFields"] = params["extra_body"]
    return req


# ---------------------------------------------------------------- response mapping


def compact(value: Any) -> Any:
    """Drop ``None`` values recursively (Bedrock batch output includes every union member of
    a content block, set to null)."""
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def split_reasoning(text: str) -> tuple[str, str | None]:
    """Move a leading ``<think>``/``<reasoning>`` block (Qwen3, DeepSeek-R1, GPT-OSS over the
    OpenAI endpoint) out of the answer."""
    m = _THINK.match(text)
    if not m:
        return text, None
    return text[m.end() :], m.group(2).strip()


def from_converse_response(
    model_id: str, resp: dict[str, Any], tool_names: set[str] | None = None
) -> ChatCompletion:
    resp = compact(resp)  # batch output spells absent fields as explicit nulls
    blocks = resp.get("output", {}).get("message", {}).get("content", [])
    texts, reasoning, calls = [], [], []
    for b in blocks:
        if "toolUse" in b and b["toolUse"].get("name") == JSON_TOOL:
            texts = [json.dumps(b["toolUse"].get("input", {}))]  # emulated json_schema output
            resp = {**resp, "stopReason": "end_turn"}
            break
        if "text" in b:
            texts.append(b["text"])
        elif "reasoningContent" in b:
            r = b["reasoningContent"].get("reasoningText", {}).get("text")
            if r:
                reasoning.append(r)
        elif "toolUse" in b:
            tu = b["toolUse"]
            calls.append(
                ToolCall(tu["toolUseId"], FunctionCall(tu["name"], json.dumps(tu.get("input", {}))))
            )
    text, think = split_reasoning("".join(texts))
    if think:
        reasoning.insert(0, think)
    if not calls:
        # Models without forced tool choice may write the call as text, including a call
        # to the hidden json_schema tool (Llama 3.x, seen live).
        recovered = recover_text_tool_call(text, (tool_names or set()) | {JSON_TOOL})
        if recovered and recovered.function.name == JSON_TOOL:
            text = recovered.function.arguments
        elif recovered:
            calls, text = [recovered], ""
            resp = {**resp, "stopReason": "tool_use"}
    usage = resp.get("usage", {})
    return ChatCompletion(
        id=resp.get("ResponseMetadata", {}).get("RequestId") or f"chatcmpl-{uuid.uuid4().hex}",
        model=model_id,
        choices=[
            Choice(
                0,
                Message(
                    "assistant",
                    text if (text or not calls) else None,
                    calls or None,
                    "\n".join(reasoning) or None,
                ),
                _STOP_REASON.get(resp.get("stopReason", ""), "stop"),
            )
        ],
        usage=Usage(
            usage.get("inputTokens", 0),
            usage.get("outputTokens", 0),
            usage.get("totalTokens", 0),
            usage.get("cacheReadInputTokens", 0),
            usage.get("cacheWriteInputTokens", 0),
        ),
        service_tier=(resp.get("serviceTier") or {}).get("type"),
        raw=resp,
    )


def to_openai_chat_body(messages: list[dict[str, Any]], **params: Any) -> dict[str, Any]:
    """Imported models take OpenAI ChatCompletion bodies via InvokeModel."""
    unknown = set(params) - _PARAMS
    if unknown:
        raise TypeError(f"Unsupported parameter(s): {', '.join(sorted(unknown))}")
    body: dict[str, Any] = {
        "messages": [
            {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id")}
            for m in messages
        ]
    }
    for key in (
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "tools",
        "tool_choice",
        "response_format",
    ):
        if params.get(key) is not None:
            body[key] = params[key]
    if params.get("extra_body"):
        body.update(params["extra_body"])
    return body


def from_openai_chat_response(model_id: str, resp: dict[str, Any]) -> ChatCompletion:
    usage = resp.get("usage", {})
    choices = []
    for i, c in enumerate(resp.get("choices", [])):
        msg = c.get("message", {})
        text, think = split_reasoning(message_text(msg.get("content")))
        calls = [
            ToolCall(
                tc.get("id", f"call_{j}"),
                FunctionCall(tc["function"]["name"], tc["function"].get("arguments", "")),
            )
            for j, tc in enumerate(msg.get("tool_calls") or [])
        ]
        choices.append(
            Choice(
                c.get("index", i),
                Message(
                    msg.get("role", "assistant"),
                    text,
                    calls or None,
                    msg.get("reasoning_content") or think,
                ),
                c.get("finish_reason") or "stop",
            )
        )
    return ChatCompletion(
        id=resp.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        model=model_id,
        choices=choices,
        usage=Usage(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
        ),
        raw=resp,
    )


# ---------------------------------------------------------------- streaming


class _ThinkSplitter:
    """Routes streamed text inside a leading <think>…</think> to reasoning_content."""

    def __init__(self) -> None:
        self.state = "start"  # start -> thinking -> answer
        self.buf = ""

    def feed(self, text: str) -> tuple[str, str]:
        """Returns (content, reasoning) to emit for this piece of text."""
        if self.state == "answer":
            return text, ""
        self.buf += text
        if self.state == "start":
            stripped = self.buf.lstrip()
            if "<think>".startswith(stripped) and len(stripped) < len("<think>"):
                return "", ""  # could still become <think>
            if not stripped.startswith("<think>"):
                self.state, out, self.buf = "answer", self.buf, ""
                return out, ""
            self.state, self.buf = "thinking", stripped[len("<think>") :]
        if "</think>" in self.buf:
            reasoning, _, rest = self.buf.partition("</think>")
            self.state, self.buf = "answer", ""
            return rest.lstrip(), reasoning
        keep = len("</think>") - 1  # hold back a possible partial closing tag
        out, self.buf = self.buf[:-keep], self.buf[-keep:]
        return "", out


class Stream:
    """Iterator of ``ChatCompletionChunk``. ``final()`` returns the assembled completion."""

    def __init__(self, model_id: str, chunks: Iterator[ChatCompletionChunk]) -> None:
        self.model_id = model_id
        self._chunks = chunks
        self._seen: list[ChatCompletionChunk] = []

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        for chunk in self._chunks:
            self._seen.append(chunk)
            yield chunk

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def final(self) -> ChatCompletion:
        for _ in self:  # drain anything not yet consumed
            pass
        text, reasoning, finish = [], [], "stop"
        calls: dict[int, ToolCall] = {}
        usage = Usage(0, 0, 0)
        for c in self._seen:
            if c.usage:
                usage = c.usage
            for ch in c.choices:
                d = ch.delta
                text.append(d.content or "")
                reasoning.append(d.reasoning_content or "")
                for tc in d.tool_calls or []:
                    cur = calls.setdefault(tc.index, ToolCall(tc.id or "", FunctionCall("", "")))
                    if tc.id:
                        cur.id = tc.id
                    cur.function.name += tc.function.name or ""
                    cur.function.arguments += tc.function.arguments or ""
                if ch.finish_reason:
                    finish = ch.finish_reason
        content = "".join(text)
        return ChatCompletion(
            id=self._seen[0].id if self._seen else f"chatcmpl-{uuid.uuid4().hex}",
            model=self.model_id,
            choices=[
                Choice(
                    0,
                    Message(
                        "assistant",
                        content if content or not calls else None,
                        [calls[i] for i in sorted(calls)] or None,
                        "".join(reasoning) or None,
                    ),
                    finish,
                )
            ],
            usage=usage,
        )


def converse_stream_chunks(model_id: str, events: Any) -> Iterator[ChatCompletionChunk]:
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    tool_index: dict[int, int] = {}
    json_block: int | None = None
    splitter = _ThinkSplitter()

    def chunk(delta: Delta, finish: str | None = None) -> ChatCompletionChunk:
        return ChatCompletionChunk(cid, model_id, [ChunkChoice(0, delta, finish)])

    for event in events:
        if "messageStart" in event:
            yield chunk(Delta(role=event["messageStart"].get("role", "assistant")))
        elif "contentBlockStart" in event:
            start = event["contentBlockStart"].get("start", {})
            if "toolUse" in start and start["toolUse"]["name"] == JSON_TOOL:
                json_block = event["contentBlockStart"]["contentBlockIndex"]
            elif "toolUse" in start:
                idx = tool_index.setdefault(
                    event["contentBlockStart"]["contentBlockIndex"], len(tool_index)
                )
                tu = start["toolUse"]
                yield chunk(
                    Delta(
                        tool_calls=[
                            ToolCallDelta(
                                idx, tu["toolUseId"], FunctionCallDelta(tu["name"], ""), "function"
                            )
                        ]
                    )
                )
        elif "contentBlockDelta" in event:
            d = event["contentBlockDelta"]["delta"]
            if "text" in d:
                content, reasoning = splitter.feed(d["text"])
                if content or reasoning:
                    yield chunk(Delta(content=content or None, reasoning_content=reasoning or None))
            elif "reasoningContent" in d and d["reasoningContent"].get("text"):
                yield chunk(Delta(reasoning_content=d["reasoningContent"]["text"]))
            elif "toolUse" in d and event["contentBlockDelta"]["contentBlockIndex"] == json_block:
                yield chunk(Delta(content=d["toolUse"].get("input", "")))
            elif "toolUse" in d:
                idx = tool_index.setdefault(
                    event["contentBlockDelta"]["contentBlockIndex"], len(tool_index)
                )
                yield chunk(
                    Delta(
                        tool_calls=[
                            ToolCallDelta(
                                idx,
                                function=FunctionCallDelta(arguments=d["toolUse"].get("input", "")),
                            )
                        ]
                    )
                )
        elif "messageStop" in event:
            stop = event["messageStop"].get("stopReason", "")
            if splitter.buf:  # flush held-back text
                content, reasoning = (
                    (splitter.buf, "") if splitter.state != "thinking" else ("", splitter.buf)
                )
                splitter.buf = ""
                yield chunk(Delta(content=content or None, reasoning_content=reasoning or None))
            yield chunk(
                Delta(),
                _STOP_REASON.get(
                    stop if json_block is None or stop != "tool_use" else "end_turn", "stop"
                ),
            )
        elif "metadata" in event:
            u = event["metadata"].get("usage", {})
            yield ChatCompletionChunk(
                cid,
                model_id,
                [],
                Usage(
                    u.get("inputTokens", 0),
                    u.get("outputTokens", 0),
                    u.get("totalTokens", 0),
                    u.get("cacheReadInputTokens", 0),
                    u.get("cacheWriteInputTokens", 0),
                ),
            )
        else:
            for key in (
                "internalServerException",
                "modelStreamErrorException",
                "validationException",
                "throttlingException",
                "serviceUnavailableException",
            ):
                if key in event:
                    raise RuntimeError(f"stream error {key}: {event[key].get('message')}")


def openai_stream_chunks(model_id: str, body: Any) -> Iterator[ChatCompletionChunk]:
    """InvokeModelWithResponseStream for imported models: each event carries an OpenAI chunk.
    Leading ``<think>`` text in streamed content goes to ``reasoning_content``."""
    splitter = _ThinkSplitter()
    for event in body:
        payload = event.get("chunk", {}).get("bytes")
        if not payload:
            continue
        data = json.loads(payload)
        u = data.get("usage")
        choices = []
        for c in data.get("choices", []):
            delta = c.get("delta") or {}
            content, reasoning = delta.get("content"), delta.get("reasoning_content")
            if content:
                content, think = splitter.feed(content)
                reasoning = (reasoning or "") + think or None
                content = content or None
            if c.get("finish_reason") and splitter.buf:
                rest = splitter.buf
                splitter.buf = ""
                if splitter.state == "thinking":
                    reasoning = (reasoning or "") + rest
                else:
                    content = (content or "") + rest
            choices.append(
                ChunkChoice(
                    c.get("index", 0),
                    Delta(role=delta.get("role"), content=content, reasoning_content=reasoning),
                    c.get("finish_reason"),
                )
            )
        yield ChatCompletionChunk(
            data.get("id", ""),
            model_id,
            choices,
            Usage(
                u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("total_tokens", 0)
            )
            if u
            else None,
        )


# ---------------------------------------------------------------- capability fallbacks

# model_id -> request features Bedrock rejected for that model. Learned per process so later
# calls skip the failing attempt.
_UNSUPPORTED: dict[str, set[str]] = {}
_FEATURES = (
    "toolConfig.toolChoice",
    "outputConfig.effort",
    "outputConfig",
    "serviceTier",
    "performanceConfig",
)


# Bedrock's wording varies by field ("the effort field", "service tier is not supported").
_REJECTION_PATTERNS = (
    ("toolChoice", "toolConfig.toolChoice"),
    ("effort", "outputConfig.effort"),
    ("outputConfig", "outputConfig"),
    ("service tier", "serviceTier"),
    ("serviceTier", "serviceTier"),
    ("performanceConfig", "performanceConfig"),
    ("latency", "performanceConfig"),
)

# Reasoning effort for models that reject Converse ``outputConfig.effort`` but expose
# reasoning through provider-specific fields. Claude needs a thinking budget in tokens.
_CLAUDE_BUDGET = {"low": 1024, "medium": 4096, "high": 16000, "xhigh": 32000, "max": 64000}


def _effort_fields(model_id: str, effort: str) -> dict[str, Any] | None:
    if "anthropic." in model_id:
        return {"thinking": {"type": "enabled", "budget_tokens": _CLAUDE_BUDGET.get(effort, 4096)}}
    if "openai.gpt-oss" in model_id:
        return {"reasoning_effort": effort if effort in ("low", "medium", "high") else "high"}
    return None


def _rejected_feature(error: ClientError) -> str | None:
    if error.response.get("Error", {}).get("Code") != "ValidationException":
        return None
    msg = str(error)
    m = _KEYWORDS_REJECTED.search(msg)
    if m and "schema" in msg:  # e.g. "For 'integer' type, properties maximum, minimum ..."
        return "schema:" + ",".join(sorted(k.strip() for k in m.group(1).split(",")))
    if not any(k in msg for k in ("doesn't support", "does not support", "not supported")):
        return None
    return next((feature for needle, feature in _REJECTION_PATTERNS if needle in msg), None)


def degrade(
    req: dict[str, Any], feature: str, model_id: str, *, quiet: bool = False
) -> dict[str, Any] | None:
    """A copy of ``req`` without ``feature`` (emulated where possible); None if unused."""
    req = copy.deepcopy(req)
    warn = (lambda *a: None) if quiet else log.warning
    if feature.startswith("schema:"):
        fmt = req.get("outputConfig", {}).get("textFormat")
        if not fmt:
            return None
        keywords = set(feature.removeprefix("schema:").split(","))
        js = fmt["structure"]["jsonSchema"]
        js["schema"] = json.dumps(relax_schema(json.loads(js["schema"]), keywords))
        warn(
            "%s doesn't support schema keywords %s; moved them into descriptions",
            model_id,
            sorted(keywords),
        )
        return req
    if feature == "toolConfig.toolChoice":
        if "toolChoice" not in req.get("toolConfig", {}):
            return None
        req["toolConfig"].pop("toolChoice")
        log.warning(
            "%s doesn't support tool_choice; using auto (the model may not call a tool)", model_id
        )
        return req
    if feature == "outputConfig.effort":
        if "effort" not in req.get("outputConfig", {}):
            return None
        effort = req["outputConfig"].pop("effort")
        if not req["outputConfig"]:
            req.pop("outputConfig")
        fields = _effort_fields(model_id, effort)
        if fields:
            req.setdefault("additionalModelRequestFields", {}).update(fields)
            if "thinking" in fields:  # Claude: budget must fit and sampling must be default
                cfg = req.setdefault("inferenceConfig", {})
                budget = fields["thinking"]["budget_tokens"]
                cfg["maxTokens"] = max(cfg.get("maxTokens", 0), budget + 1024)
                cfg.pop("temperature", None)
                cfg.pop("topP", None)
            warn("%s: mapping reasoning_effort=%s to %s", model_id, effort, list(fields))
        else:
            warn("%s doesn't support reasoning_effort; ignoring it", model_id)
        return req
    if feature == "outputConfig":
        out = req.pop("outputConfig", None)
        if out is None:
            return None
        fmt = out.get("textFormat")
        if fmt:  # emulate json_schema with a forced tool whose input schema is the schema
            js = fmt["structure"]["jsonSchema"]
            tools = req.setdefault("toolConfig", {}).setdefault("tools", [])
            tools.append(
                {
                    "toolSpec": {
                        "name": JSON_TOOL,
                        "description": js.get("description") or "Return the final answer as JSON.",
                        "inputSchema": {"json": json.loads(js["schema"])},
                    }
                }
            )
            req["toolConfig"]["toolChoice"] = {"tool": {"name": JSON_TOOL}}
        warn(
            "%s doesn't support outputConfig; %s",
            model_id,
            "emulating json_schema via tool use" if fmt else "ignoring reasoning_effort",
        )
        return req
    if feature in ("serviceTier", "performanceConfig"):
        if feature not in req:
            return None
        req.pop(feature)
        warn("%s doesn't support %s; using the default", model_id, feature)
        return req
    return None


def call_with_fallbacks(model_id: str, req: dict[str, Any], call: Callable[..., Any]) -> Any:
    for learned in _UNSUPPORTED.get(model_id, set()):  # apply what we already learned
        req = degrade(req, learned, model_id, quiet=True) or req
    for _ in range(len(_FEATURES)):
        try:
            return call(**req)
        except ClientError as e:
            feature = _rejected_feature(e)
            new = degrade(req, feature, model_id) if feature else None
            if feature is None or new is None:
                raise
            _UNSUPPORTED.setdefault(model_id, set()).add(feature)
            req = new
    return call(**req)


def sse_stream_chunks(model_id: str, body: Any) -> Iterator[ChatCompletionChunk]:
    """SageMaker response streams: LMI sends OpenAI chunks as JSON Lines
    (``application/jsonlines``, seen live); other servers use SSE ``data: {...}``. Lines may
    be split across PayloadParts, so buffer until a newline."""
    buf = ""

    def events() -> Iterator[dict[str, Any]]:
        nonlocal buf
        for part in body:
            buf += part.get("PayloadPart", {}).get("Bytes", b"").decode()
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line and line != "[DONE]":
                    yield {"chunk": {"bytes": line.encode()}}
        tail = buf.strip().removeprefix("data:").strip()
        if tail and tail != "[DONE]":
            yield {"chunk": {"bytes": tail.encode()}}

    yield from openai_stream_chunks(model_id, events())


_CAPACITY = re.compile(
    r"no capacity|nocapacity|insufficient capacity|scaling up|"
    r"no (?:available )?(?:copies|instances)",
    re.I,
)


def is_capacity_cold_start(e: ClientError) -> bool:
    """A SageMaker inference component scaled to zero rejects requests until a copy starts."""
    code = e.response.get("Error", {}).get("Code", "")
    return code in (
        "ValidationError",
        "ModelError",
        "ServiceUnavailable",
        "ModelNotReadyException",
    ) and bool(_CAPACITY.search(str(e)))


# ---------------------------------------------------------------- client

# Raised while an idle imported model is being restored (scale-from-zero on Bedrock).
_COLD_START_CODES = {"ModelNotReadyException"}
# Surfaced as ThrottledError (with guidance) once botocore's own retries are exhausted.
_THROTTLE_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceQuotaExceededException",
}


class _Completions:
    def __init__(self, chat: Chat) -> None:
        self._chat = chat

    def create(self, *, messages: list[dict[str, Any]], stream: bool = False, **params: Any) -> Any:
        """Returns ``ChatCompletion``, or a ``Stream`` of chunks when ``stream=True``."""
        if self._chat._before_call:
            self._chat._before_call()  # e.g. spend limit
        if stream:
            return self._chat._with_cold_start(lambda: self._stream(messages, params))
        result: ChatCompletion = self._chat._with_cold_start(lambda: self._once(messages, params))
        self._chat._after(result.usage, result.service_tier)
        return result

    def _once(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> ChatCompletion:
        chat = self._chat
        if chat.api == "sagemaker":
            resp = chat.runtime.invoke_endpoint(
                EndpointName=chat.model_id,
                InferenceComponentName=chat.component,
                Body=json.dumps(to_openai_chat_body(messages, **params)),
                ContentType="application/json",
                Accept="application/json",
            )
            return from_openai_chat_response(chat.model_id, json.loads(resp["Body"].read()))
        if chat.api == "openai_chat":
            resp = chat.runtime.invoke_model(
                modelId=chat.model_id,
                body=json.dumps(to_openai_chat_body(messages, **params)),
                contentType="application/json",
                accept="application/json",
            )
            return from_openai_chat_response(chat.model_id, json.loads(resp["body"].read()))
        req = to_converse_request(chat.model_id, messages, **params)
        resp = call_with_fallbacks(chat.model_id, req, chat.runtime.converse)
        names = {(t.get("function") or t)["name"] for t in params.get("tools") or []}
        return from_converse_response(chat.model_id, resp, tool_names=names)

    def _stream(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> Stream:
        chat = self._chat
        if chat.api == "sagemaker":
            body = to_openai_chat_body(messages, **params) | {"stream": True}
            resp = chat.runtime.invoke_endpoint_with_response_stream(
                EndpointName=chat.model_id,
                InferenceComponentName=chat.component,
                Body=json.dumps(body),
                ContentType="application/json",
            )
            chunks = sse_stream_chunks(chat.model_id, resp["Body"])
        elif chat.api == "openai_chat":
            body = to_openai_chat_body(messages, **params) | {"stream": True}
            resp = chat.runtime.invoke_model_with_response_stream(
                modelId=chat.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            chunks = openai_stream_chunks(chat.model_id, resp["body"])
        else:
            req = to_converse_request(chat.model_id, messages, **params)
            resp = call_with_fallbacks(chat.model_id, req, chat.runtime.converse_stream)
            chunks = converse_stream_chunks(chat.model_id, resp["stream"])

        def tracked() -> Iterator[ChatCompletionChunk]:
            for c in chunks:
                if c.usage:
                    chat._after(c.usage, params.get("service_tier"))
                yield c

        return Stream(chat.model_id, tracked())


@dataclass
class CompletionChoice:
    index: int
    text: str
    finish_reason: str


@dataclass
class Completion:
    id: str
    model: str
    choices: list[CompletionChoice]
    usage: Usage
    object: str = "text_completion"


class _TextCompletions:
    """OpenAI ``completions.create(prompt=...)``: the prompt is sent as one user turn."""

    def __init__(self, chat: Chat) -> None:
        self._chat = chat

    def create(self, *, prompt: str, **params: Any) -> Completion:
        r = self._chat.completions.create(messages=[{"role": "user", "content": prompt}], **params)
        return Completion(
            r.id,
            r.model,
            [
                CompletionChoice(c.index, c.message.content or "", c.finish_reason)
                for c in r.choices
            ],
            r.usage,
        )


class Chat:
    """OpenAI-style ``chat.completions.create``. ``api`` is "converse" (Bedrock-native custom
    and base models) or "openai_chat" (imported models, via InvokeModel)."""

    def __init__(
        self,
        runtime: Any,
        model_id: str,
        *,
        api: str = "converse",
        cold_start_timeout_s: float = 900,
        component: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        on_usage: Callable[[Usage, str | None], None] | None = None,
        before_call: Callable[[], None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.model_id = model_id
        self.api = api
        self.component = component  # SageMaker inference component name (api="sagemaker")
        self.cold_start_timeout_s = cold_start_timeout_s
        self._sleep = sleep
        self._on_usage = on_usage
        self._before_call = before_call
        self.completions = _Completions(self)
        self.text_completions = _TextCompletions(self)

    def _after(self, usage: Usage, tier: str | None) -> None:
        if self._on_usage:
            self._on_usage(usage, tier)

    def _with_cold_start(self, call: Callable[[], Any]) -> Any:
        """Retry while the model is restoring from idle, so scale-to-zero feels serverless."""
        waited, delay, warned = 0.0, 5.0, False
        while True:
            try:
                return call()
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if is_capacity_cold_start(e):
                    code = "ModelNotReadyException"  # endpoint scaled to zero: same handling
                if code in _THROTTLE_CODES:
                    from skyhammer.errors import ThrottledError

                    message = e.response.get("Error", {}).get("Message", code)
                    raise ThrottledError(self.model_id, message) from e
                if code not in _COLD_START_CODES:
                    raise
                if waited >= self.cold_start_timeout_s:
                    raise
                if not warned:
                    log.warning(
                        "%s is warming up after being idle; retrying for up to %ds",
                        self.model_id,
                        self.cold_start_timeout_s,
                    )
                    warned = True
                self._sleep(delay)
                waited += delay
                delay = min(delay * 2, 30.0)
