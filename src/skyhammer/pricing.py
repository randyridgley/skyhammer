"""Live training prices from the AWS Price List API (read-only, free), cached on disk.

Usage types (verified 2026-10-29, us-east-1):
  Bedrock-native:  {R}-NovaMicro-Customization-Training            per 1K tokens
                   {R}-Nova2.0Lite-RFT-Training-Hours               per hour (RFT)
                   {R}-NovaMicro-Customization-Storage             per model-month
  SageMaker:       {R}-ServerlessTraining:FineTuning-SFT-LORA-<hub-model-id>   per 1M tokens
                   ...-RLVR-LORA-<hub-model-id>                      per hour (RL techniques)
  Model import:    {R}-<Arch>-CustomModelImport-Storage-v1:0         per model-month
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from skyhammer.catalog import BackendKind, Technique, TrainingType
from skyhammer.errors import SpendLimitExceededError

if TYPE_CHECKING:
    from skyhammer.router import Route
    from skyhammer.session import Session

log = logging.getLogger(__name__)

CACHE_FILE = Path.home() / ".cache" / "skyhammer" / "prices.json"
CACHE_TTL_S = 7 * 24 * 3600
PRICING_REGION = "us-east-1"  # the Price List API endpoint; covers all regions

_REGION_PREFIX = {
    "us-east-1": "USE1",
    "us-east-2": "USE2",
    "us-west-2": "USW2",
    "eu-central-1": "EUC1",
}
_NOVA_PRICE_NAME = {
    "nova-micro": "NovaMicro",
    "nova-lite": "NovaLite",
    "nova-pro": "NovaPro",
    "nova-2-lite": "Nova2.0Lite",
}
_SM_TECHNIQUE = {Technique.SFT: "SFT", Technique.DPO: "DPO", Technique.RFT: "RLVR"}
CUSTOM_MODEL_STORAGE_USD_MONTH = 1.95  # same for Bedrock customization and import


@dataclass(frozen=True)
class TrainingPrice:
    usd_per_1k_tokens: float | None = None
    usd_per_hour: float | None = None  # RL techniques bill by time
    usage_type: str | None = None


def usage_type(route: Route, technique: Technique, training_type: TrainingType) -> str | None:
    prefix = _REGION_PREFIX.get(route.region)
    if prefix is None or route.backend is BackendKind.MANTLE:  # not in the Price List yet
        return None
    if route.backend is BackendKind.BEDROCK:
        # Open-weight Bedrock customization (Qwen, GPT-OSS) isn't in the Price List yet.
        name = _NOVA_PRICE_NAME.get(route.model.alias)
        if name and technique is Technique.RFT:
            return f"{prefix}-{name}-RFT-Training-Hours"
        return f"{prefix}-{name}-Customization-Training" if name else None
    tech = _SM_TECHNIQUE.get(technique)
    if tech is None:
        return None
    rank = "LORA" if training_type is TrainingType.LORA else "FULLRANK"
    return (
        f"{prefix}-ServerlessTraining:FineTuning-{tech}-{rank}-{route.capability.backend_model_id}"
    )


def _service(route: Route) -> str:
    return "AmazonBedrock" if route.backend is BackendKind.BEDROCK else "AmazonSageMaker"


def _read_cache() -> dict[str, Any]:
    try:
        data: dict[str, Any] = json.loads(CACHE_FILE.read_text())
        return data
    except (OSError, ValueError):
        return {}


def _lookup(session: Session, service: str, usage: str) -> dict[str, Any] | None:
    cache = _read_cache()
    hit = cache.get(usage)
    if hit and time.time() - hit["at"] < CACHE_TTL_S:
        return hit.get("price")  # type: ignore[no-any-return]
    pc = session.client("pricing", region=PRICING_REGION)
    resp = pc.get_products(
        ServiceCode=service,
        MaxResults=10,
        Filters=[{"Type": "TERM_MATCH", "Field": "usagetype", "Value": usage}],
    )
    price = None
    for raw in resp.get("PriceList", []):
        product = json.loads(raw)
        for term in product["terms"].get("OnDemand", {}).values():
            for dim in term["priceDimensions"].values():
                price = {"unit": dim["unit"], "usd": float(dim["pricePerUnit"]["USD"])}
    cache[usage] = {"at": time.time(), "price": price}
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache))
    except OSError:
        pass
    return price


def training_price(
    session: Session, route: Route, technique: Technique, training_type: TrainingType
) -> TrainingPrice:
    """Best-effort price lookup; returns an empty TrainingPrice when unknown or offline."""
    usage = usage_type(route, technique, training_type)
    if usage is None:
        return TrainingPrice()
    try:
        price = _lookup(session, _service(route), usage)
    except Exception as e:  # no credentials, no network, throttling: estimate without price
        log.debug("price lookup for %s failed: %s", usage, e)
        return TrainingPrice(usage_type=usage)
    if price is None:
        return TrainingPrice(usage_type=usage)
    unit = price["unit"].lower()
    if unit == "1k tokens":
        return TrainingPrice(usd_per_1k_tokens=price["usd"], usage_type=usage)
    if unit == "1m tokens":
        return TrainingPrice(usd_per_1k_tokens=price["usd"] / 1000, usage_type=usage)
    if unit.startswith("hour"):
        return TrainingPrice(usd_per_hour=price["usd"], usage_type=usage)
    return TrainingPrice(usage_type=usage)


# ---------------------------------------------------------------- inference prices

_INFERENCE_SERVICES = ("AmazonBedrock", "AmazonBedrockFoundationModels")
_INFERENCE_FEATURES = ("On-demand Inference", "Batch Inference")
_DROP_TOKENS = {"instruct", "it", "edition", "model", "amazon", "bedrock"}
_KINDS = (
    ("prompt cache read", "cache_read"),
    ("prompt cache write", "cache_write"),
    ("input tokens", "input"),
    ("output tokens", "output"),
)


# Tokens that may differ between Bedrock and Price List names without changing the price:
# provider names and neutral descriptors. Variant words (flash, mini, turbo, thinking,
# latency-optimized) and any number are never ignored.
_NEUTRAL_TOKENS = {
    "ai21",
    "amazon",
    "anthropic",
    "cohere",
    "deepseek",
    "google",
    "meta",
    "minimax",
    "mistral",
    "moonshot",
    "moonshotai",
    "nvidia",
    "openai",
    "qwen",
    "stability",
    "twelvelabs",
    "writer",
    "xai",
    "zai",
    "dense",
    "pt",
    "chat",
}


def _norm_token(t: str) -> str:
    t = re.sub(r"^v(?=\d)", "", t)  # v1.5 -> 1.5
    return re.sub(r"^(\d+)\.0$", r"\1", t)  # 2.0 -> 2


def normalize_model_name(name: str) -> str:
    """Canonical tokens: lowercase, parentheses kept as tokens, versions normalized."""
    tokens = [_norm_token(t) for t in re.split(r"[\s\-_()]+", name.lower())]
    return " ".join(t for t in tokens if t and t not in _DROP_TOKENS)


def parse_price_row(product: dict[str, Any]) -> dict[str, Any] | None:
    """One Price List product -> {model, kind, tier, routing, usd_per_token}, or None."""
    a = product["product"]["attributes"]
    usage = a.get("usagetype", "")
    itype = (a.get("inferenceType") or "").lower()
    if not a.get("model") or any(
        k in usage for k in ("mantle", "custom-model", "Reserved", "1h", "long_ctx")
    ):
        return None
    kind = next((k for needle, k in _KINDS if itype.startswith(needle)), None)
    if kind is None:
        return None
    if "search_units" in usage:
        kind = "search_unit"  # rerank: priced per search unit, not per token
    feature = a.get("feature", "")
    tier = (
        "batch"
        if "batch" in itype or feature == "Batch Inference" or "Batch" in usage
        else "flex"
        if "flex" in itype
        else "priority"
        if "priority" in itype
        else "standard"
    )
    routing = "global" if "global" in usage.lower() else "regional"
    for term in product["terms"].get("OnDemand", {}).values():
        for dim in term["priceDimensions"].values():
            unit = dim["unit"].lower()
            scale = (
                1000
                if unit == "1k tokens"
                else 1_000_000
                if unit == "1m tokens"
                else 1
                if unit == "search units"
                else None
            )
            if scale is None:
                return None
            return {
                "model": normalize_model_name(a["model"]),
                "kind": kind,
                "tier": tier,
                "routing": routing,
                "usd_per_token": float(dim["pricePerUnit"]["USD"]) / scale,
            }
    return None


@dataclass(frozen=True)
class TokenPrices:
    input: float | None = None
    output: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None


class InferencePrices:
    """Per-token prices for serverless Bedrock models, from the AWS Price List API."""

    _cache: ClassVar[dict[str, InferencePrices]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.index: dict[tuple[str, str, str], dict[str, float]] = {}
        for r in rows:
            key = (r["model"], r["tier"], r["routing"])
            self.index.setdefault(key, {})[r["kind"]] = r["usd_per_token"]
        self.names = {k[0] for k in self.index}

    @classmethod
    def for_session(cls, session: Session) -> InferencePrices:
        region = session.region
        with cls._lock:  # concurrent first calls load the price list once
            if region not in cls._cache:
                cls._cache[region] = cls(_load_inference_rows(session, region))
            return cls._cache[region]

    def match(self, model_name: str) -> str | None:
        """Price-list name for a Bedrock model name. Names may differ only by neutral tokens
        (provider names, "dense"); never by versions, sizes or variant words, so a wrong price
        is never returned (Nova 2 Lite must not match Nova Lite)."""
        norm = normalize_model_name(model_name)
        if norm in self.names:
            return norm
        tokens = set(norm.split())
        scored = []
        for name in self.names:
            diff = tokens ^ set(name.split())
            if not diff <= _NEUTRAL_TOKENS:  # e.g. "turbo"/"flash"/"2" change the price
                continue
            if not (tokens <= set(name.split()) or set(name.split()) <= tokens):
                continue
            scored.append((len(diff), name))
        scored.sort()
        if not scored or (len(scored) > 1 and scored[0][0] == scored[1][0]):
            return None  # nothing, or ambiguous: unknown beats wrong
        return scored[0][1]

    def lookup(
        self, model_name: str, tier: str = "standard", routing: str = "regional"
    ) -> TokenPrices | None:
        name = self.match(model_name)
        if name is None:
            return None
        prices = self.index.get((name, tier, routing)) or self.index.get((name, tier, "regional"))
        if not prices:
            return None
        return TokenPrices(
            prices.get("input"),
            prices.get("output"),
            prices.get("cache_read"),
            prices.get("cache_write"),
        )

    def search_unit_price(self, model_name: str) -> float | None:
        name = self.match(model_name)
        if name is None:
            return None
        return (self.index.get((name, "standard", "regional")) or {}).get("search_unit")

    def cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        tier: str = "standard",
        routing: str = "regional",
    ) -> float | None:
        p = self.lookup(model_name, tier, routing)
        if p is None or p.input is None or (p.output is None and output_tokens):
            return None
        return (
            input_tokens * p.input
            + output_tokens * (p.output or 0.0)
            + cache_read_tokens * (p.cache_read if p.cache_read is not None else p.input)
            + cache_write_tokens * (p.cache_write if p.cache_write is not None else p.input)
        )


def _load_inference_rows(session: Session, region: str) -> list[dict[str, Any]]:
    path = CACHE_FILE.parent / f"inference-prices-{region}.json"
    try:
        cached = json.loads(path.read_text())
        if time.time() - cached["at"] < CACHE_TTL_S:
            rows: list[dict[str, Any]] = cached["rows"]
            return rows
    except (OSError, ValueError, KeyError):
        pass
    rows = fetch_inference_rows(session, region)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"at": time.time(), "rows": rows}))
    except OSError:
        pass
    return rows


def fetch_inference_rows(session: Session, region: str) -> list[dict[str, Any]]:
    pc = session.client("pricing", region=PRICING_REGION)
    rows = []
    for service in _INFERENCE_SERVICES:
        for feature in _INFERENCE_FEATURES:
            pages = pc.get_paginator("get_products").paginate(
                ServiceCode=service,
                Filters=[
                    {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                    {"Type": "TERM_MATCH", "Field": "feature", "Value": feature},
                ],
            )
            for page in pages:
                for raw in page["PriceList"]:
                    row = parse_price_row(json.loads(raw))
                    if row:
                        rows.append(row)
    return rows


# ---------------------------------------------------------------- spend tracking


@dataclass
class SpendTracker:
    """Running serverless spend for this process; enforces ``max_spend_usd``."""

    max_spend_usd: float | None = None
    total_usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)
    unpriced_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, model: str, usd: float | None) -> None:
        with self._lock:
            if usd is None:
                self.unpriced_calls += 1
                return
            self.total_usd += usd
            self.by_model[model] = self.by_model.get(model, 0.0) + usd

    def check(self) -> None:
        if self.max_spend_usd is not None and self.total_usd >= self.max_spend_usd:
            raise SpendLimitExceededError(self.total_usd, self.max_spend_usd)
