"""Client-side traffic splitting between models for A/B tests and canaries (SV-11).

``TrafficSplit({"support-qwen-v1": 90, "support-qwen-v2": 10})`` sends each call to one
model, picked by weight. Pass ``sticky_key=`` (a user or session ID) to keep someone on
the same model across calls. Every response says which model served it
(``response.variant``), and ``split.stats`` compares calls, errors, latency, tokens and
cost per model.

Splitting happens in the client because the models usually live on different backends
(a Bedrock custom model against a SageMaker endpoint, or a fine-tune against its base);
SageMaker production-variant weights can't route between inference components.
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skyhammer.errors import SkyhammerError

if TYPE_CHECKING:
    from skyhammer.inference import ChatCompletionChunk, Stream, Usage
    from skyhammer.llm import LLM
    from skyhammer.session import Session


@dataclass
class VariantStats:
    calls: int = 0
    errors: int = 0
    latency_s: float = 0.0  # total, for calls that returned (streams: time to first chunk)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def mean_latency_s(self) -> float | None:
        ok = self.calls - self.errors
        return self.latency_s / ok if ok else None


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Weights as fractions summing to 1; zero-weight models are kept (no traffic)."""
    if not weights:
        raise SkyhammerError("A traffic split needs at least one model")
    if any(w < 0 for w in weights.values()):
        raise SkyhammerError(f"Traffic weights can't be negative: {dict(weights)}")
    total = float(sum(weights.values()))
    if total <= 0:
        raise SkyhammerError(f"At least one model needs a positive weight: {dict(weights)}")
    return {name: w / total for name, w in weights.items()}


def pick(weights: Mapping[str, float], point: float) -> str:
    """The model whose cumulative-weight bucket contains ``point`` (0 <= point < 1)."""
    cumulative = 0.0
    chosen = ""
    for name, w in weights.items():
        if w <= 0:
            continue
        chosen = name
        cumulative += w
        if point < cumulative:
            return name
    return chosen  # rounding: the last model with traffic


def sticky_point(key: str, salt: str = "") -> float:
    """A stable point in [0, 1) for ``key``: the same user lands on the same model."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class TrafficSplit:
    """Weighted routing across models; ``.chat.completions.create`` works like ``LLM.chat``.

    Models are names of your fine-tuned models or any Bedrock model alias or ID, or ready
    ``LLM`` objects. ``salt`` changes which bucket a sticky key lands in (use one per
    experiment so the same users aren't always the canary)."""

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        session: Session | None = None,
        llms: Mapping[str, LLM] | None = None,
        salt: str = "",
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.weights = normalize_weights(weights)
        self._session = session
        self._llms: dict[str, LLM] = dict(llms or {})
        self.salt = salt
        self._rng = rng
        self.stats: dict[str, VariantStats] = {name: VariantStats() for name in self.weights}
        self._lock = threading.Lock()  # stats and the model cache are shared across threads
        self.chat = _SplitChat(self)

    def __repr__(self) -> str:
        parts = ", ".join(f"{n}: {w:.0%}" for n, w in self.weights.items())
        return f"TrafficSplit({{{parts}}})"

    def llm(self, name: str) -> LLM:
        with self._lock:
            if name not in self._llms:
                from skyhammer.environments import resolve_llm

                self._llms[name] = resolve_llm(name, session=self._session)
            return self._llms[name]

    def choose(self, sticky_key: str | None = None) -> str:
        point = sticky_point(sticky_key, self.salt) if sticky_key is not None else self._rng()
        return pick(self.weights, point)

    def summary(self) -> list[dict[str, Any]]:
        """One row per model: weight, calls, error rate, mean latency, tokens and cost."""
        rows = []
        for name, s in self.stats.items():
            rows.append(
                {
                    "model": name,
                    "weight": self.weights[name],
                    "calls": s.calls,
                    "error_rate": s.errors / s.calls if s.calls else None,
                    "mean_latency_s": s.mean_latency_s,
                    "prompt_tokens": s.prompt_tokens,
                    "completion_tokens": s.completion_tokens,
                    "cost_usd": s.cost_usd,
                }
            )
        return rows


class _SplitCompletions:
    def __init__(self, split: TrafficSplit) -> None:
        self._split = split

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        stream: bool = False,
        sticky_key: str | None = None,
        **params: Any,
    ) -> Any:
        """Same as ``LLM.chat.completions.create``; the result's ``variant`` names the model
        that answered. ``sticky_key`` pins a user to one model."""
        split = self._split
        name = split.choose(sticky_key)
        stats, lock = split.stats[name], split._lock
        with lock:
            stats.calls += 1
        started = time.monotonic()
        try:
            result = split.llm(name).chat.completions.create(
                messages=messages, stream=stream, **params
            )
        except Exception:
            with lock:
                stats.errors += 1
            raise
        if stream:
            return _tracked_stream(result, name, stats, started, lock)
        with lock:
            stats.latency_s += time.monotonic() - started
            _add_usage(stats, result.usage)
        result.variant = name
        return result


def _add_usage(stats: VariantStats, usage: Usage) -> None:
    stats.prompt_tokens += usage.prompt_tokens
    stats.completion_tokens += usage.completion_tokens
    stats.cost_usd += usage.cost_usd or 0.0


def _tracked_stream(
    stream: Stream, name: str, stats: VariantStats, started: float, lock: threading.Lock
) -> Stream:
    stream.variant = name
    inner = stream._chunks

    def chunks() -> Iterator[ChatCompletionChunk]:
        first = True
        for chunk in inner:
            with lock:
                if first:
                    stats.latency_s += time.monotonic() - started
                    first = False
                if chunk.usage:
                    _add_usage(stats, chunk.usage)
            yield chunk

    stream._chunks = chunks()
    return stream


class _SplitChat:
    def __init__(self, split: TrafficSplit) -> None:
        self.completions = _SplitCompletions(split)
