"""Environments: stable names like ``support-qwen@prod`` that point at a model version (SV-12).

Code calls ``LLM.get("support-qwen@prod")`` and never changes; you promote a new version
to ``prod`` when it passes evaluation, send a canary share of traffic to it first, and
roll back in one call if it misbehaves:

    env = Environment("support-qwen@prod")
    env.split({"support-qwen-v1": 90, "support-qwen-v2": 10})   # canary
    env.promote("support-qwen-v2")                               # all traffic
    env.rollback()                                               # back to the split

Environments live in the manifest store next to model manifests
(``manifests/envs/<app>.json``), with every change kept as history. Like the rest of
the store, the last writer wins.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from skyhammer import registry
from skyhammer import tags as tagging
from skyhammer.errors import BootstrapMissingError, SkyhammerError
from skyhammer.traffic import TrafficSplit, normalize_weights

if TYPE_CHECKING:
    from skyhammer.llm import LLM
    from skyhammer.session import Session

_ENVS = "envs"
_APP = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9])*$")
_ENV = re.compile(r"^[a-z0-9](-*[a-z0-9])*$")
MAX_HISTORY = 20


def parse(ref: str) -> tuple[str, str]:
    """``"support-qwen@prod"`` -> ``("support-qwen", "prod")``."""
    app, sep, env = ref.partition("@")
    if not sep or not _APP.match(app) or len(app) > 54 or not _ENV.match(env) or len(env) > 32:
        raise SkyhammerError(
            f"Invalid environment {ref!r}: use <app>@<env>, for example support-qwen@prod "
            "(letters, digits and hyphens; the environment in lowercase)"
        )
    return app, env


def is_ref(name: str) -> bool:
    return "@" in name


def _session(session: Session | None) -> Session:
    from skyhammer.session import get_session

    return session or get_session()


def resolve_llm(name: str, *, session: Session | None = None) -> LLM:
    """An ``LLM`` for a name: an environment (when it points at one model), your
    fine-tuned model by that name, or else any Bedrock model alias or ID."""
    from skyhammer.llm import LLM

    session = _session(session)
    if is_ref(name):
        return Environment(name, session=session).llm()
    try:
        if registry.load_model(session, name):
            return LLM.get(name, session=session)
    except BootstrapMissingError:
        pass  # no bootstrap: serverless calls still work
    return LLM(name, session=session)


class Environment:
    """A named slot (``app@env``) holding the model, or weighted models, it serves."""

    def __init__(self, ref: str, *, session: Session | None = None) -> None:
        self.app, self.env = parse(ref)
        self._session = session

    @property
    def ref(self) -> str:
        return f"{self.app}@{self.env}"

    @property
    def session(self) -> Session:
        return _session(self._session)

    def __repr__(self) -> str:
        return f"Environment({self.ref!r})"

    # ---- state ----

    def _manifest(self) -> dict[str, Any]:
        res = self.session.resources
        data = tagging.read_manifest(self.session.client("s3"), res.bucket, _ENVS, self.app)
        return data or {"name": self.app, "envs": {}}

    def _save(self, manifest: dict[str, Any]) -> None:
        res = self.session.resources
        tagging.write_manifest(
            self.session.client("s3"), res.bucket, _ENVS, self.app, manifest, res.kms_key_arn
        )

    def _state(self) -> dict[str, Any] | None:
        state: dict[str, Any] | None = self._manifest()["envs"].get(self.env)
        return state

    @property
    def exists(self) -> bool:
        return self._state() is not None

    @property
    def targets(self) -> dict[str, float]:
        """``{model: weight}`` this environment serves (weights as saved, e.g. 90/10)."""
        state = self._state()
        if not state:
            raise SkyhammerError(
                f"{self.ref} doesn't exist yet: promote a model to it first "
                f"(Environment({self.ref!r}).promote('<model>'))"
            )
        return dict(state["targets"])

    @property
    def history(self) -> list[dict[str, Any]]:
        """Earlier targets, most recent first: ``[{"targets": {...}, "at": epoch}, ...]``."""
        state = self._state() or {}
        return list(reversed(state.get("history", [])))

    # ---- changes ----

    def _check_model(self, name: str) -> None:
        if is_ref(name):
            raise SkyhammerError("Environments point at models, not at other environments")
        manifest = registry.load_model(self.session, name)
        if manifest is None:
            return  # a Bedrock model alias or ID; checked when called
        if manifest.get("deleted"):
            raise SkyhammerError(f"Model {name!r} was deleted")

    def _set(self, targets: Mapping[str, float], *, note: str | None) -> dict[str, float]:
        normalize_weights(targets)  # validates
        for name in targets:
            self._check_model(name)
        manifest = self._manifest()
        state = manifest["envs"].get(self.env)
        history = list(state.get("history", [])) if state else []
        if state:
            history.append(
                {k: state[k] for k in ("targets", "at", "note") if state.get(k) is not None}
            )
        new = {"targets": dict(targets), "at": time.time(), "history": history[-MAX_HISTORY:]}
        if note:
            new["note"] = note
        manifest["envs"][self.env] = new
        self._save(manifest)
        return dict(targets)

    def promote(self, model: str, *, note: str | None = None) -> dict[str, float]:
        """Send all of this environment's traffic to ``model``."""
        return self._set({model: 100}, note=note)

    def split(self, weights: Mapping[str, float], *, note: str | None = None) -> dict[str, float]:
        """Share traffic by weight, e.g. ``{"v1": 90, "v2": 10}`` for a 10% canary."""
        return self._set(weights, note=note)

    def rollback(self) -> dict[str, float]:
        """Restore the previous targets (the change being undone is dropped from history)."""
        manifest = self._manifest()
        state = manifest["envs"].get(self.env)
        if not state or not state.get("history"):
            raise SkyhammerError(f"{self.ref} has no earlier version to roll back to")
        previous = state["history"][-1]
        manifest["envs"][self.env] = {**previous, "history": state["history"][:-1]}
        self._save(manifest)
        return dict(previous["targets"])

    def delete(self) -> None:
        """Remove this environment (the models it points at are untouched)."""
        manifest = self._manifest()
        if manifest["envs"].pop(self.env, None) is None:
            raise SkyhammerError(f"{self.ref} doesn't exist")
        self._save(manifest)

    # ---- calling ----

    def llm(self) -> LLM:
        """The ``LLM`` this environment serves; fails while traffic is split (use ``.chat``)."""
        live = {m: w for m, w in self.targets.items() if w > 0}
        if len(live) != 1:
            raise SkyhammerError(
                f"{self.ref} splits traffic across {', '.join(live)}; call it through "
                f"Environment({self.ref!r}).chat or .traffic()"
            )
        return resolve_llm(next(iter(live)), session=self.session)

    def traffic(self, *, salt: str | None = None) -> TrafficSplit:
        """A ``TrafficSplit`` over the current targets (reads the store once)."""
        return TrafficSplit(self.targets, session=self.session, salt=salt or self.ref)

    @property
    def chat(self) -> Any:
        """``chat.completions.create`` over the current targets; the response's
        ``variant`` says which model answered. Pass ``sticky_key=`` to pin a user."""
        return self.traffic().chat


def list_environments(session: Session | None = None) -> list[dict[str, Any]]:
    """Every environment: ``[{"ref", "targets", "at", "note"}]``."""
    session = _session(session)
    s3 = session.client("s3")
    bucket = session.resources.bucket
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"manifests/{_ENVS}/"
    ):
        for obj in page.get("Contents", []):
            app = obj["Key"].rsplit("/", 1)[-1].removesuffix(".json")
            manifest = tagging.read_manifest(s3, bucket, _ENVS, app) or {"envs": {}}
            for env, state in sorted(manifest["envs"].items()):
                out.append(
                    {
                        "ref": f"{app}@{env}",
                        "targets": state["targets"],
                        "at": state.get("at"),
                        "note": state.get("note"),
                    }
                )
    return out
