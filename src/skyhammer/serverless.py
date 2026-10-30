"""Serverless model directory: every Bedrock model you can call per token, by alias or ID.

Built from ListFoundationModels + ListInferenceProfiles (cached 24h per region), so new
Bedrock models appear without an SDK release. Each model resolves to the ID to invoke:

* ``geo`` (default): the geographic cross-region profile (``us.``/``eu.``/``apac.``) when
  one exists, else the in-region model ID. Data stays in the geography.
* ``global``: the ``global.`` profile when one exists (max capacity, any region).
* ``region``: in-region on-demand only (``cross_region: false``); profile-only models fail.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from skyhammer.errors import SkyhammerError

if TYPE_CHECKING:
    from skyhammer.session import Session

CACHE_DIR = Path.home() / ".cache" / "skyhammer"
CACHE_TTL_S = 24 * 3600

_GEO = {"us": "us", "eu": "eu", "ap": "apac", "ca": "ca", "sa": "sa"}


class Routing(StrEnum):
    GEO = "geo"
    GLOBAL = "global"
    REGION = "region"


def geo_prefix(region: str) -> str | None:
    return _GEO.get(region.split("-", 1)[0])


def derive_alias(model_id: str, provider: str = "") -> str:
    """``anthropic.claude-haiku-4-5-20251001-v1:0`` -> ``claude-haiku-4-5``.

    Drops the provider prefix, dates, ``-v1``, ``-1:0`` and ``:N`` suffixes; keeps every other
    number (``-v2``, ``-4-6``) so an alias never points at a different model version. Very
    short names get the provider back (``deepseek-r1``). Colliding aliases are dropped.
    """
    vendor, _, rest = model_id.partition(".")
    name = rest or model_id
    # Only strip suffixes that never distinguish models: "-1:0" (gpt-oss-20b-1:0), ":N" /
    # ":128k", 8-digit dates, and "-v1". Trailing numbers like claude-sonnet-4-6 are versions.
    name = re.sub(r"-1:\d+$", "", name)
    name = re.sub(r":\d+[a-z]*$", "", name)
    name = re.sub(r"-\d{8}", "", name)
    name = re.sub(r"-v1$", "", name)
    if re.fullmatch(r"[a-z]?\d[\w.]*", name):
        name = f"{vendor}-{name}"
    return name.lower()


@dataclass(frozen=True)
class ServerlessModel:
    model_id: str
    name: str
    provider: str
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]
    on_demand: bool
    streaming: bool
    profiles: dict[str, str] = field(default_factory=dict)  # {"us": "us.<id>", "global": ...}
    alias: str | None = None
    legacy: bool = False

    @property
    def kind(self) -> str:
        out = set(self.output_modalities)
        if "EMBEDDING" in out:
            return "embedding"
        if "rerank" in self.model_id:
            return "rerank"
        if "IMAGE" in out:
            return "image"
        if "SPEECH" in out:
            return "speech"
        return "chat"

    def invocation_id(self, routing: Routing, region: str) -> str:
        geo = self.profiles.get(geo_prefix(region) or "")
        glob = self.profiles.get("global")
        if routing is Routing.REGION:
            if self.on_demand:
                return self.model_id
            raise SkyhammerError(
                f"{self.model_id} needs a cross-region inference profile and cross_region is "
                "off. Enable cross_region (or routing='geo') or choose another model."
            )
        if routing is Routing.GLOBAL and glob:
            return glob
        if geo:
            return geo
        if self.on_demand:
            return self.model_id
        if glob:
            raise SkyhammerError(
                f"{self.model_id} is only offered through the global inference profile in "
                f"{region}; pass routing='global' (requests may be served from any AWS region)."
            )
        raise SkyhammerError(f"{self.model_id} has no invocation route in {region}")

    def describe_routes(self) -> str:
        routes = (["on-demand"] if self.on_demand else []) + sorted(self.profiles)
        return ", ".join(routes)


def parse_directory(data: dict[str, Any]) -> dict[str, ServerlessModel]:
    profiles: dict[str, dict[str, str]] = {}
    for p in data.get("inferenceProfileSummaries", []):
        prefix, _, base = p["inferenceProfileId"].partition(".")
        if base:
            profiles.setdefault(base, {})[prefix] = p["inferenceProfileId"]
    models: dict[str, ServerlessModel] = {}
    for m in data.get("modelSummaries", []):
        mid = m["modelId"]
        if re.search(
            r":\d+:\d+[a-z]*$|:\d+[a-z]+$", mid
        ):  # context-window variants (fine-tune ids)
            continue
        on_demand = "ON_DEMAND" in m.get("inferenceTypesSupported", [])
        if not on_demand and mid not in profiles:
            continue
        models[mid] = ServerlessModel(
            model_id=mid,
            name=m.get("modelName", mid),
            provider=m.get("providerName", mid.split(".")[0]),
            input_modalities=tuple(m.get("inputModalities", [])),
            output_modalities=tuple(m.get("outputModalities", [])),
            on_demand=on_demand,
            streaming=bool(m.get("responseStreamingSupported")),
            profiles=profiles.get(mid, {}),
            legacy=m.get("lifecycle") == "LEGACY",
        )
    aliases = {mid: derive_alias(mid) for mid in models}
    counts = Counter(aliases.values())
    return {
        mid: _with_alias(model, aliases[mid] if counts[aliases[mid]] == 1 else None)
        for mid, model in models.items()
    }


def _with_alias(model: ServerlessModel, alias: str | None) -> ServerlessModel:
    return ServerlessModel(**{**model.__dict__, "alias": alias})


class ModelDirectory:
    """Lookup over a region's serverless models. Use ``ModelDirectory.for_session``."""

    _cache: ClassVar[dict[str, ModelDirectory]] = {}

    def __init__(self, session: Session, region: str) -> None:
        self.session = session
        self.region = region
        self.models = parse_directory(self._load())
        self._by_alias = {m.alias: m for m in self.models.values() if m.alias}
        self._by_profile = {pid: m for m in self.models.values() for pid in m.profiles.values()}

    @classmethod
    def for_session(cls, session: Session, region: str | None = None) -> ModelDirectory:
        region = region or session.region
        key = f"{id(session)}:{region}"
        if key not in cls._cache:
            cls._cache[key] = cls(session, region)
        return cls._cache[key]

    def _load(self) -> dict[str, Any]:
        path = CACHE_DIR / f"models-{self.region}.json"
        try:
            cached = json.loads(path.read_text())
            if time.time() - cached["at"] < CACHE_TTL_S:
                data: dict[str, Any] = cached["data"]
                return data
        except (OSError, ValueError, KeyError):
            pass
        data = self._fetch()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"at": time.time(), "data": data}))
        except OSError:
            pass
        return data

    def _fetch(self) -> dict[str, Any]:
        bedrock = self.session.client("bedrock", region=self.region)
        fields = (
            "modelId",
            "modelName",
            "providerName",
            "inputModalities",
            "outputModalities",
            "responseStreamingSupported",
            "inferenceTypesSupported",
        )
        models = [
            {
                **{k: m[k] for k in fields if k in m},
                "lifecycle": m.get("modelLifecycle", {}).get("status", "ACTIVE"),
            }
            for m in bedrock.list_foundation_models()["modelSummaries"]
        ]
        profiles: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {"maxResults": 1000, "typeEquals": "SYSTEM_DEFINED"}
        while True:
            resp = bedrock.list_inference_profiles(**kwargs)
            profiles += resp.get("inferenceProfileSummaries", [])
            if not resp.get("nextToken"):
                break
            kwargs["nextToken"] = resp["nextToken"]
        keep = ("inferenceProfileId", "inferenceProfileName", "type")
        return {
            "modelSummaries": models,
            "inferenceProfileSummaries": [{k: p.get(k) for k in keep} for p in profiles],
        }

    def find(self, name: str) -> ServerlessModel:
        """Resolve a model ID, inference-profile ID, or alias (case-insensitive)."""
        if name in self.models:
            return self.models[name]
        if name in self._by_profile:
            return self._by_profile[name]
        alias = name.lower()
        if alias in self._by_alias:
            return self._by_alias[alias]
        choices = list(self._by_alias) + list(self.models)
        close = difflib.get_close_matches(alias, choices, n=5, cutoff=0.5)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise SkyhammerError(
            f"No Bedrock model {name!r} in {self.region}.{hint} "
            "Run `skyhammer models list --serverless` to see all models."
        )

    def list(self, kind: str | None = None, provider: str | None = None) -> list[ServerlessModel]:
        out = [
            m
            for m in self.models.values()
            if (kind is None or m.kind == kind)
            and (provider is None or m.provider.lower().startswith(provider.lower()))
        ]
        return sorted(out, key=lambda m: (m.provider.lower(), m.model_id))
