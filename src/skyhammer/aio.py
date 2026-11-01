"""Async client (DX-8): ``AsyncLLM`` runs Skyhammer calls on worker threads for asyncio code.

``await AsyncLLM("nova-micro").chat.completions.create(messages=...)`` returns the same
``ChatCompletion`` as ``LLM``; ``stream=True`` returns an async iterator. Fan out with
``asyncio.gather``: at most ``max_concurrency`` requests are in flight at once (a stream
counts until it is read to the end or closed), and the rest wait their turn, so a burst
of tasks doesn't trip Bedrock's request quotas or exhaust the connection pool.

Each call runs the synchronous client on a thread (boto3 has no native asyncio support),
so everything ``LLM`` does (cost tracking, the spend limit, guardrails, fallbacks,
cold-start retries) works the same. Cancelling a task stops waiting for its result, but
a request already sent finishes in the background.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, TypeVar

from skyhammer.errors import SkyhammerError

if TYPE_CHECKING:
    from skyhammer.embeddings import EmbeddingResponse, RerankResponse
    from skyhammer.environments import Environment
    from skyhammer.inference import ChatCompletion, Completion
    from skyhammer.llm import LLM
    from skyhammer.responses import Response
    from skyhammer.session import Session
    from skyhammer.traffic import TrafficSplit

T = TypeVar("T")
DEFAULT_MAX_CONCURRENCY = 16
_DONE = object()


class AsyncStream(AsyncIterator[Any]):
    """Async iterator over a stream's chunks (or Responses API events). ``await final()``
    returns the assembled result; attributes such as ``variant`` pass through. It holds a
    concurrency slot until read to the end or closed (``async with`` closes it)."""

    def __init__(self, client: AsyncLLM, stream: Any) -> None:
        self._client = client
        self._stream = stream
        self._iter: Iterator[Any] = iter(stream)
        self._open = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __aiter__(self) -> AsyncStream:
        return self

    async def __anext__(self) -> Any:
        if not self._open:
            raise StopAsyncIteration
        try:
            item = await self._client._offload(next, self._iter, _DONE)
        except BaseException:
            self._release()
            raise
        if item is _DONE:
            self._release()
            raise StopAsyncIteration
        return item

    async def final(self) -> Any:
        """Read the rest of the stream and return the assembled completion or response."""
        try:
            return await self._client._offload(self._stream.final)
        finally:
            self._release()

    async def aclose(self) -> None:
        """Stop reading and give the concurrency slot back."""
        self._release()

    async def __aenter__(self) -> AsyncStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._release()

    def _release(self) -> None:
        if self._open:
            self._open = False
            self._client._slots.release()

    def __del__(self) -> None:  # an abandoned stream must not keep its slot forever
        if getattr(self, "_open", False):
            self._release()


class AsyncLLM:
    """Async front for an ``LLM``, an environment (``"support@prod"``, splits included) or a
    ``TrafficSplit``. Same calls as the sync client, awaited:

    ``chat.completions.create``, ``completions.create``, ``embeddings.create``, ``rerank``
    and ``responses.create`` / ``retrieve`` / ``delete`` / ``cancel``.

    ``model`` is resolved on first use, on a worker thread, so constructing one never
    blocks the event loop. ``sync`` is the underlying client."""

    def __init__(
        self,
        model: str | LLM | Environment | TrafficSplit,
        *,
        session: Session | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        if max_concurrency < 1:
            raise SkyhammerError("max_concurrency must be at least 1")
        self._model = model
        self._session = session
        self._target: Any = None
        self._chat: Any = None
        self._resolve_lock = threading.Lock()
        self.max_concurrency = max_concurrency
        self._slots = asyncio.Semaphore(max_concurrency)
        self._pool = ThreadPoolExecutor(max_concurrency, thread_name_prefix="skyhammer")
        self.chat = _AsyncChat(self)
        self.completions = _AsyncTextCompletions(self)
        self.embeddings = _AsyncEmbeddings(self)
        self.responses = _AsyncResponses(self)

    def __repr__(self) -> str:
        name = self._model if isinstance(self._model, str) else repr(self._model)
        return f"AsyncLLM({name}, max_concurrency={self.max_concurrency})"

    @property
    def sync(self) -> Any:
        """The ``LLM`` or ``TrafficSplit`` that serves the calls (resolved now if needed)."""
        return self._resolve()

    def _resolve(self) -> Any:
        with self._resolve_lock:
            if self._target is None:
                self._target = _resolve(self._model, self._session)
            return self._target

    def _chat_client(self) -> Any:
        """One ``Chat`` per model, reused so every call shares a client and its connections.
        A split picks its model per call, so it is asked each time."""
        from skyhammer.traffic import TrafficSplit

        target = self._resolve()
        if isinstance(target, TrafficSplit):
            return target.chat
        with self._resolve_lock:
            if self._chat is None:
                self._chat = target.chat
            return self._chat

    def _llm(self, feature: str) -> LLM:
        from skyhammer.traffic import TrafficSplit

        target = self._resolve()
        if isinstance(target, TrafficSplit):
            raise SkyhammerError(
                f"{feature} needs one model, but {self._model} splits traffic; "
                "use chat.completions or pick a model with .sync.llm(name)"
            )
        llm: LLM = target
        return llm

    async def _offload(self, fn: Callable[..., T], *args: Any) -> T:
        """Run ``fn`` on the pool without taking a slot (for work inside a held slot)."""
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    async def _call(self, fn: Callable[[], T]) -> T:
        async with self._slots:
            return await self._offload(fn)

    async def _open_stream(self, fn: Callable[[], Any]) -> AsyncStream:
        await self._slots.acquire()
        try:
            stream = await self._offload(fn)
        except BaseException:
            self._slots.release()
            raise
        return AsyncStream(self, stream)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        return_documents: bool = False,
    ) -> RerankResponse:
        """Order ``documents`` by relevance to ``query`` (see ``LLM.rerank``)."""
        return await self._call(
            lambda: self._llm("rerank").rerank(
                query, documents, top_n=top_n, return_documents=return_documents
            )
        )

    async def aclose(self) -> None:
        """Shut down the worker threads (waits for running calls)."""
        await asyncio.get_running_loop().run_in_executor(None, self._pool.shutdown)

    async def __aenter__(self) -> AsyncLLM:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def _resolve(model: Any, session: Session | None) -> Any:
    from skyhammer.environments import Environment, is_ref, resolve_llm

    if isinstance(model, Environment):
        return model.traffic()
    if isinstance(model, str):
        if is_ref(model):
            return Environment(model, session=session).traffic()
        return resolve_llm(model, session=session)
    return model  # an LLM or a TrafficSplit


class _AsyncCompletions:
    def __init__(self, client: AsyncLLM) -> None:
        self._client = client

    async def create(
        self, *, messages: list[dict[str, Any]], stream: bool = False, **params: Any
    ) -> Any:
        """Same arguments as ``LLM.chat.completions.create``. Returns a ``ChatCompletion``,
        or an ``AsyncStream`` of chunks with ``stream=True``."""
        client = self._client

        def call() -> Any:
            return client._chat_client().completions.create(
                messages=messages, stream=stream, **params
            )

        if stream:
            return await client._open_stream(call)
        result: ChatCompletion = await client._call(call)
        return result


class _AsyncChat:
    def __init__(self, client: AsyncLLM) -> None:
        self.completions = _AsyncCompletions(client)


class _AsyncTextCompletions:
    def __init__(self, client: AsyncLLM) -> None:
        self._client = client

    async def create(self, *, prompt: str, **params: Any) -> Completion:
        """``completions.create(prompt=...)`` (the prompt is sent as one user turn)."""
        client = self._client
        result: Completion = await client._call(
            lambda: client._llm("completions").completions.create(prompt=prompt, **params)
        )
        return result


class _AsyncEmbeddings:
    def __init__(self, client: AsyncLLM) -> None:
        self._client = client

    async def create(self, **params: Any) -> EmbeddingResponse:
        """Same arguments as ``LLM.embeddings.create`` (``input=``, ``dimensions=``...)."""
        client = self._client
        result: EmbeddingResponse = await client._call(
            lambda: client._llm("embeddings").embeddings.create(**params)
        )
        return result


class _AsyncResponses:
    def __init__(self, client: AsyncLLM) -> None:
        self._client = client

    async def create(self, *, stream: bool = False, **params: Any) -> Any:
        """Same arguments as ``LLM.responses.create``. Returns a ``Response``, or an
        ``AsyncStream`` of events with ``stream=True``."""
        client = self._client

        def call() -> Any:
            return client._llm("responses").responses.create(stream=stream, **params)

        if stream:
            return await client._open_stream(call)
        return await client._call(call)

    async def retrieve(self, response_id: str) -> Response:
        client = self._client
        result: Response = await client._call(
            lambda: client._llm("responses").responses.retrieve(response_id)
        )
        return result

    async def delete(self, response_id: str) -> None:
        client = self._client
        await client._call(lambda: client._llm("responses").responses.delete(response_id))

    async def cancel(self, response_id: str) -> Response:
        client = self._client
        result: Response = await client._call(
            lambda: client._llm("responses").responses.cancel(response_id)
        )
        return result
