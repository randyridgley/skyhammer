import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

from skyhammer import AsyncLLM, Environment, TrafficSplit
from skyhammer.errors import SkyhammerError
from skyhammer.inference import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    ChunkChoice,
    Delta,
    Message,
    Stream,
    Usage,
)
from skyhammer.pricing import SpendTracker
from tests.unit.test_environments import FakeLLM
from tests.unit.test_registry_cli import s3_session  # noqa: F401  (fixture)

MESSAGES = [{"role": "user", "content": "hi"}]


class SlowLLM:
    """Records how many calls overlap and which threads they ran on."""

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.threads: set[str] = set()
        self.lock = threading.Lock()
        self.chat_built = 0
        self.responses = SimpleNamespace(
            create=lambda **p: self._run(("response", p)),
            retrieve=lambda rid: ("retrieve", rid),
            delete=lambda rid: None,
            cancel=lambda rid: ("cancel", rid),
        )
        self.embeddings = SimpleNamespace(create=lambda **p: self._run(("embed", p)))
        self.completions = SimpleNamespace(create=lambda **p: self._run(("text", p)))

    @property
    def chat(self) -> Any:
        self.chat_built += 1
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def rerank(self, query: str, documents: list[str], **kw: Any) -> Any:
        return self._run(("rerank", query, documents, kw))

    def _run(self, result: Any) -> Any:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.threads.add(threading.current_thread().name)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        return result

    def _create(self, *, messages: Any, stream: bool = False, **params: Any) -> Any:
        if stream:
            chunks = [
                ChatCompletionChunk("c", "m", [ChunkChoice(0, Delta(content=t))]) for t in "abc"
            ]
            return Stream("m", iter(chunks))
        text = messages[0]["content"]
        return self._run(
            ChatCompletion(
                "c", "m", [Choice(0, Message("assistant", text), "stop")], Usage(1, 1, 2)
            )
        )


def test_gather_runs_concurrently_up_to_the_limit() -> None:
    slow = SlowLLM()
    client = AsyncLLM(slow, max_concurrency=3)  # type: ignore[arg-type]

    async def main() -> list[Any]:
        return await asyncio.gather(
            *(
                client.chat.completions.create(messages=[{"role": "user", "content": str(i)}])
                for i in range(12)
            )
        )

    started = time.monotonic()
    results = asyncio.run(main())
    assert [r.choices[0].message.content for r in results] == [str(i) for i in range(12)]
    assert slow.peak == 3
    assert time.monotonic() - started < 12 * slow.delay  # really overlapped
    assert all(t.startswith("skyhammer") for t in slow.threads)
    assert slow.chat_built == 1  # one Chat (and boto3 client) shared by every call


def test_streams_hold_a_slot_until_read_or_closed() -> None:
    client = AsyncLLM(SlowLLM(), max_concurrency=2)  # type: ignore[arg-type]

    async def main() -> None:
        s = await client.chat.completions.create(messages=MESSAGES, stream=True)
        assert client._slots._value == 1
        assert [c.choices[0].delta.content async for c in s] == ["a", "b", "c"]
        assert client._slots._value == 2

        s = await client.chat.completions.create(messages=MESSAGES, stream=True)
        assert (await s.final()).choices[0].message.content == "abc"
        assert client._slots._value == 2

        async with await client.chat.completions.create(messages=MESSAGES, stream=True) as s:
            await anext(s)
            assert client._slots._value == 1
        assert client._slots._value == 2
        await s.aclose()  # closing twice gives back only one slot
        assert client._slots._value == 2

    asyncio.run(main())


def test_errors_propagate_and_free_the_slot() -> None:
    client = AsyncLLM(TrafficSplit({"bad": 1}, llms={"bad": FakeLLM("bad", fail=True)}))

    async def main() -> None:
        for stream in (False, True):
            with pytest.raises(RuntimeError, match="boom"):
                await client.chat.completions.create(messages=MESSAGES, stream=stream)
        assert client._slots._value == client.max_concurrency

    asyncio.run(main())


def test_traffic_splits_keep_variants_and_counts_under_concurrency() -> None:
    split = TrafficSplit({"v1": 50, "v2": 50}, llms={"v1": FakeLLM("v1"), "v2": FakeLLM("v2")})
    client = AsyncLLM(split, max_concurrency=8)

    async def main() -> list[Any]:
        calls = [client.chat.completions.create(messages=MESSAGES) for _ in range(200)]
        return await asyncio.gather(*calls)

    results = asyncio.run(main())
    assert {r.variant for r in results} == {"v1", "v2"}
    assert sum(s.calls for s in split.stats.values()) == 200
    assert sum(s.prompt_tokens for s in split.stats.values()) == 2000

    async def stream() -> str:
        s = await client.chat.completions.create(messages=MESSAGES, stream=True)
        assert s.variant in ("v1", "v2")  # attributes pass through
        return (await s.final()).choices[0].message.content  # type: ignore[no-any-return]

    assert asyncio.run(stream()) == "ok"

    async def embed() -> None:
        await client.embeddings.create(input="x")

    with pytest.raises(SkyhammerError, match="splits traffic"):
        asyncio.run(embed())


def test_other_calls_pass_through() -> None:
    client = AsyncLLM(SlowLLM(delay=0))  # type: ignore[arg-type]

    async def main() -> None:
        assert await client.embeddings.create(input="x") == ("embed", {"input": "x"})
        assert await client.completions.create(prompt="p") == ("text", {"prompt": "p"})
        assert await client.rerank("q", ["d"], top_n=1) == (
            "rerank",
            "q",
            ["d"],
            {"top_n": 1, "return_documents": False},
        )
        assert await client.responses.create(input="hi") == (
            "response",
            {"stream": False, "input": "hi"},
        )
        assert await client.responses.retrieve("resp_1") == ("retrieve", "resp_1")
        assert await client.responses.cancel("resp_1") == ("cancel", "resp_1")
        assert await client.responses.delete("resp_1") is None
        async with client:
            pass

    asyncio.run(main())


def test_names_resolve_lazily_off_the_event_loop(
    s3_session: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _ = s3_session
    resolved_on: list[str] = []

    def fake_resolve(name: str, session: Any = None) -> Any:
        resolved_on.append(threading.current_thread().name)
        return SlowLLM(delay=0)

    monkeypatch.setattr("skyhammer.environments.resolve_llm", fake_resolve)
    client = AsyncLLM("nova-micro", session=session)
    assert resolved_on == []  # constructing doesn't touch AWS

    async def main() -> None:
        await client.chat.completions.create(messages=MESSAGES)

    asyncio.run(main())
    assert len(resolved_on) == 1 and resolved_on[0].startswith("skyhammer")

    Environment("support@prod", session=session).split({"nova-micro": 90, "nova-lite": 10})
    assert isinstance(AsyncLLM("support@prod", session=session).sync, TrafficSplit)
    assert "support@prod" in repr(AsyncLLM("support@prod"))
    with pytest.raises(SkyhammerError, match="at least 1"):
        AsyncLLM("nova-micro", max_concurrency=0)


def test_spend_tracking_is_thread_safe() -> None:
    spend = SpendTracker()
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda _: spend.record("m", 0.001), range(5000)))
    assert spend.by_model["m"] == pytest.approx(5.0)
    assert spend.total_usd == pytest.approx(5.0)
