"""What a wrapper (``AsyncLLM``, the framework adapters) calls: a model name, an ``LLM``,
an environment or a ``TrafficSplit``, resolved on first use and safe to share across threads.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from skyhammer.errors import SkyhammerError

if TYPE_CHECKING:
    from skyhammer.environments import Environment
    from skyhammer.llm import LLM
    from skyhammer.session import Session
    from skyhammer.traffic import TrafficSplit


class Target:
    """``model`` as a name (``"nova-micro"``, ``"support-qwen-v2"``, ``"support@prod"``), an
    ``LLM``, an ``Environment`` or a ``TrafficSplit``. Environments resolve to their traffic
    split, so canaries keep working behind any wrapper."""

    def __init__(
        self, model: str | LLM | Environment | TrafficSplit, session: Session | None = None
    ) -> None:
        self.model = model
        self.session = session
        self._resolved: Any = None
        self._chat: Any = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        if isinstance(self.model, str):
            return self.model
        return str(getattr(self.model, "ref", None) or getattr(self.model, "name", self.model))

    def resolve(self) -> LLM | TrafficSplit:
        with self._lock:
            if self._resolved is None:
                self._resolved = _resolve(self.model, self.session)
            resolved: LLM | TrafficSplit = self._resolved
            return resolved

    def chat(self) -> Any:
        """``Chat`` for the model, built once so every call shares a client and its
        connections. A split picks its model per call, so it is returned as is."""
        from skyhammer.traffic import TrafficSplit

        target = self.resolve()
        if isinstance(target, TrafficSplit):
            return target.chat
        with self._lock:
            if self._chat is None:
                self._chat = target.chat
            return self._chat

    def create(self, *, messages: list[dict[str, Any]], stream: bool = False, **params: Any) -> Any:
        """``chat.completions.create`` on the model (or the split)."""
        return self.chat().completions.create(messages=messages, stream=stream, **params)

    def llm(self, feature: str) -> LLM:
        """The single model behind the target; ``feature`` names the call for the error."""
        from skyhammer.traffic import TrafficSplit

        target = self.resolve()
        if isinstance(target, TrafficSplit):
            raise SkyhammerError(
                f"{feature} needs one model, but {self.name} splits traffic; "
                "use chat.completions or pick a model with .sync.llm(name)"
            )
        return target


def _resolve(model: Any, session: Session | None) -> Any:
    from skyhammer.environments import Environment, is_ref, resolve_llm

    if isinstance(model, Environment):
        return model.traffic()
    if isinstance(model, str):
        if is_ref(model):
            return Environment(model, session=session).traffic()
        return resolve_llm(model, session=session)
    return model  # an LLM or a TrafficSplit
