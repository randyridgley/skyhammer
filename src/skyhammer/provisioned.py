"""Bedrock Provisioned Throughput (SV-10): reserved model units, billed by the hour.

``llm.deploy(type="provisioned", model_units=1, max_hourly_usd=70)`` buys capacity for a
fine-tune or a base model and calls go to it from then on. It suits steady, high-volume
traffic that on-demand quotas can't carry, and models Bedrock serves no other way. For
anything else, on-demand (pay per token) is far cheaper: one Nova model unit costs
$60.50/hour in us-east-1 whether or not it's used.

- **Price first.** ``price(...)`` reads the hourly rate from the AWS Price List; purchases
  refuse to run without ``max_hourly_usd`` at or above it.
- **Commitments can't be cancelled.** No commitment (the default) can be deleted any
  time; ``commitment="1_month"`` or ``"6_months"`` is cheaper per hour but bills for the
  whole term, so it also needs ``accept_commitment=True``.
- **Swap models in place.** Deploying another fine-tune of the same base model under an
  existing Provisioned Throughput's name moves the capacity to it (no new purchase).
- **Quota.** Accounts start with 0 model units; request them in Service Quotas
  ("Model units no-commitment Provisioned Throughputs ...") or through AWS Support.

Only Provisioned Throughputs tagged ``skyhammer:name`` are listed, changed or deleted.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from botocore.exceptions import ClientError

from skyhammer.errors import CostLimitExceededError, JobFailedError, SkyhammerError
from skyhammer.pricing import PRICING_REGION, match_model_name, normalize_model_name

if TYPE_CHECKING:
    from skyhammer.session import Session

TAG = "skyhammer:name"
# API value -> hours billed for the term (None: hourly, delete any time).
COMMITMENTS: dict[str | None, int | None] = {None: None, "OneMonth": 730, "SixMonths": 4380}
_ALIASES = {
    None: None,
    "none": None,
    "no_commitment": None,
    "1_month": "OneMonth",
    "1m": "OneMonth",
    "onemonth": "OneMonth",
    "6_months": "SixMonths",
    "6m": "SixMonths",
    "sixmonths": "SixMonths",
}
_FEATURE_TERMS = {"no commit": None, "1 month": "OneMonth", "6 months": "SixMonths"}
READY = "InService"


def commitment_term(value: str | None) -> str | None:
    """``None``/"none", "1_month" or "6_months" (or the API's "OneMonth"/"SixMonths")."""
    key = value.lower() if isinstance(value, str) else None
    if key not in _ALIASES:
        raise SkyhammerError(f"commitment must be none, 1_month or 6_months, not {value!r}")
    return _ALIASES[key]


def is_provisioned_arn(arn: str) -> bool:
    return ":provisioned-model/" in arn


# ---- prices ----


def parse_price_row(product: dict[str, Any]) -> dict[str, Any] | None:
    """One Price List product -> {model, commitment, custom, usd_per_hour}, or None."""
    a = product["product"]["attributes"]
    usage, feature = a.get("usagetype", ""), a.get("feature", "")
    if "ProvisionedThroughput" not in usage or "ModelUnits" not in usage:
        return None
    name = a.get("model") or a.get("titanModelUnit")
    term = next((t for label, t in _FEATURE_TERMS.items() if feature.endswith(label)), "?")
    if not name or term == "?":
        return None
    for terms in product["terms"].get("OnDemand", {}).values():
        for dim in terms["priceDimensions"].values():
            usd = float(dim["pricePerUnit"].get("USD", 0))
            if dim["unit"].lower() == "hour" and usd > 0:  # $0 rows are placeholders
                return {
                    "model": normalize_model_name(name),
                    "commitment": term,
                    "custom": "-Custom-" in usage,
                    "usd_per_hour": usd,
                }
    return None


class ProvisionedPrices:
    """Hourly price per model unit, from the AWS Price List (cached per region)."""

    _cache: ClassVar[dict[str, ProvisionedPrices]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.index: dict[tuple[str, str | None, bool], float] = {
            (r["model"], r["commitment"], r["custom"]): r["usd_per_hour"] for r in rows
        }
        self.names = {r["model"] for r in rows}

    @classmethod
    def for_session(cls, session: Session, region: str | None = None) -> ProvisionedPrices:
        region = region or session.region
        with cls._lock:
            if region not in cls._cache:
                cls._cache[region] = cls(fetch_rows(session, region))
            return cls._cache[region]

    def hourly(self, model_name: str, commitment: str | None, *, custom: bool) -> float | None:
        """USD per model unit per hour; custom models fall back to their base model's price
        (Bedrock prices them the same unless the Price List says otherwise)."""
        name = match_model_name(model_name, self.names)
        if name is None:
            return None
        if custom and (name, commitment, True) in self.index:
            return self.index[(name, commitment, True)]
        return self.index.get((name, commitment, False))


def fetch_rows(session: Session, region: str) -> list[dict[str, Any]]:
    pc = session.client("pricing", region=PRICING_REGION)
    rows = []
    for custom in ("", "Custom - "):
        for label in _FEATURE_TERMS:
            pages = pc.get_paginator("get_products").paginate(
                ServiceCode="AmazonBedrock",
                Filters=[
                    {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                    {
                        "Type": "TERM_MATCH",
                        "Field": "feature",
                        "Value": f"Provisioned Throughput Inference - {custom}{label}",
                    },
                ],
            )
            for page in pages:
                for raw in page["PriceList"]:
                    row = parse_price_row(json.loads(raw))
                    if row:
                        rows.append(row)
    return rows


@dataclass(frozen=True)
class Estimate:
    model_units: int
    commitment: str | None
    usd_per_hour: float | None  # all units together; None when the Price List has no row
    term_usd: float | None  # the whole commitment; None without one

    def __str__(self) -> str:
        if self.usd_per_hour is None:
            return f"{self.model_units} model unit(s): price unknown (check Amazon Bedrock pricing)"
        text = f"{self.model_units} model unit(s): ${self.usd_per_hour:,.2f}/hour"
        if self.term_usd is not None:
            text += f", ${self.term_usd:,.0f} for the {self.commitment} term (not cancellable)"
        else:
            text += f" (about ${self.usd_per_hour * 730:,.0f}/month if left running)"
        return text


def price(
    session: Session,
    model_name: str,
    *,
    model_units: int = 1,
    commitment: str | None = None,
    custom: bool = False,
    region: str | None = None,
) -> Estimate:
    """What ``model_units`` of Provisioned Throughput for ``model_name`` cost."""
    term = commitment_term(commitment)
    unit = ProvisionedPrices.for_session(session, region).hourly(model_name, term, custom=custom)
    hourly = unit * model_units if unit is not None else None
    hours = COMMITMENTS[term]
    total = hourly * hours if hourly is not None and hours else None
    return Estimate(model_units, term, hourly, total)


def check_budget(
    estimate: Estimate, max_hourly_usd: float | None, *, accept_commitment: bool
) -> None:
    """Refuse a purchase that isn't explicitly within budget."""
    if estimate.commitment and not accept_commitment:
        raise SkyhammerError(
            f"{estimate}. A {estimate.commitment} commitment can't be deleted before the term "
            "ends; pass accept_commitment=True to buy it"
        )
    if max_hourly_usd is None:
        raise SkyhammerError(
            f"Provisioned Throughput bills by the hour: {estimate}. "
            "Pass max_hourly_usd= to confirm the spend"
        )
    if estimate.usd_per_hour is None:
        raise SkyhammerError(
            f"{estimate}; Skyhammer won't buy capacity it can't price. "
            "Check https://aws.amazon.com/bedrock/pricing/"
        )
    if estimate.usd_per_hour > max_hourly_usd:
        raise CostLimitExceededError(
            estimate.usd_per_hour,
            max_hourly_usd,
            hint="Raise max_hourly_usd or buy fewer model units.",
        )


# ---- the resource ----


@dataclass(frozen=True)
class ProvisionedThroughput:
    name: str
    arn: str  # pass as modelId to invoke it
    model_arn: str
    status: str  # Creating | InService | Updating | Failed
    model_units: int
    commitment: str | None = None
    commitment_expires: dt.datetime | None = None
    desired_model_arn: str | None = None
    failure: str | None = None

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> ProvisionedThroughput:
        return cls(
            name=d["provisionedModelName"],
            arn=d["provisionedModelArn"],
            model_arn=d["modelArn"],
            status=d["status"],
            model_units=d.get("modelUnits", 0),
            commitment=d.get("commitmentDuration"),
            commitment_expires=d.get("commitmentExpirationTime"),
            desired_model_arn=d.get("desiredModelArn"),
            failure=d.get("failureMessage"),
        )


def effective_status(pt: ProvisionedThroughput) -> str:
    """``Updating`` until a model swap has landed: right after the update call Bedrock can
    still report InService for the old model."""
    if pt.desired_model_arn and pt.desired_model_arn != pt.model_arn and pt.status == READY:
        return "Updating"
    return pt.status


class Provisioned:
    """Provisioned Throughputs in one region, limited to ones Skyhammer tagged."""

    def __init__(self, session: Session, region: str | None = None) -> None:
        self.session = session
        self.region = region or session.region
        self.bedrock = session.client("bedrock", region=self.region)

    def _tags(self, arn: str) -> dict[str, str]:
        tags = self.bedrock.list_tags_for_resource(resourceARN=arn).get("tags", [])
        return {t["key"]: t["value"] for t in tags}

    def _ours(self, arn: str) -> ProvisionedThroughput:
        pt = self.get(arn)
        if TAG not in self._tags(pt.arn):
            raise SkyhammerError(f"{pt.name} wasn't created by Skyhammer; leaving it alone")
        return pt

    def get(self, name_or_arn: str) -> ProvisionedThroughput:
        try:
            d = self.bedrock.get_provisioned_model_throughput(provisionedModelId=name_or_arn)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                raise SkyhammerError(f"No Provisioned Throughput {name_or_arn!r}") from None
            raise
        return ProvisionedThroughput.from_api(d)

    def find(self, name: str) -> ProvisionedThroughput | None:
        for pt in self._all(nameContains=name):
            if pt.name == name:
                return pt
        return None

    def _all(self, **filters: Any) -> list[ProvisionedThroughput]:
        out, token = [], None
        while True:
            kwargs = {**filters, **({"nextToken": token} if token else {})}
            resp = self.bedrock.list_provisioned_model_throughputs(**kwargs)
            out += [ProvisionedThroughput.from_api(d) for d in resp["provisionedModelSummaries"]]
            token = resp.get("nextToken")
            if not token:
                return out

    def tagged(self) -> list[tuple[ProvisionedThroughput, dict[str, str]]]:
        """(Provisioned Throughput, its tags) for each one tagged ``skyhammer:name``."""
        out = []
        for pt in self._all():
            tags = self._tags(pt.arn)
            if TAG in tags:
                out.append((pt, tags))
        return out

    def purchase(
        self,
        name: str,
        model_id: str,
        *,
        model_units: int,
        commitment: str | None,
        tags: dict[str, str],
    ) -> ProvisionedThroughput:
        """Buy, or reuse the one already named ``name``: if it serves another model, swap
        that model for ``model_id`` (same base model only; Bedrock checks)."""
        if model_units < 1:
            raise SkyhammerError("model_units must be at least 1")
        if TAG not in tags:  # untagged capacity would be invisible to list, prune and sweep
            raise SkyhammerError(f"Provisioned Throughput needs a {TAG} tag")
        existing = self.find(name)
        if existing and TAG not in self._tags(existing.arn):  # failed or not, never touch it
            raise SkyhammerError(f"{name} wasn't created by Skyhammer; pick another name")
        if existing and existing.status != "Failed":
            if existing.model_units != model_units:
                raise SkyhammerError(
                    f"{name} has {existing.model_units} model unit(s) and Bedrock can't change "
                    "that in place; delete it and buy again, or use another name"
                )
            if model_id not in (existing.model_arn, existing.desired_model_arn):
                self.bedrock.update_provisioned_model_throughput(
                    provisionedModelId=existing.arn, desiredModelId=model_id
                )
                return self.get(existing.arn)
            return existing  # idempotent: re-running deploy/apply reuses it
        if existing:  # a failed purchase holds the name; clear it first
            self.bedrock.delete_provisioned_model_throughput(provisionedModelId=existing.arn)
        kwargs: dict[str, Any] = {
            "provisionedModelName": name,
            "modelId": model_id,
            "modelUnits": model_units,
            "tags": [{"key": k, "value": v} for k, v in tags.items()],
        }
        if commitment:
            kwargs["commitmentDuration"] = commitment
        arn = self.bedrock.create_provisioned_model_throughput(**kwargs)["provisionedModelArn"]
        return self.get(arn)

    def wait(
        self, name_or_arn: str, *, poll_seconds: float = 30, timeout_seconds: float = 3600
    ) -> ProvisionedThroughput:
        deadline = time.monotonic() + timeout_seconds
        while True:
            pt = self.get(name_or_arn)
            if effective_status(pt) == READY:
                return pt
            if pt.status == "Failed":
                raise JobFailedError(pt.name, pt.failure or "Provisioned Throughput failed")
            if time.monotonic() > deadline:
                raise TimeoutError(f"{pt.name} not in service after {timeout_seconds:.0f}s")
            time.sleep(poll_seconds)

    def delete(self, name_or_arn: str) -> None:
        """Stop the hourly billing. Refuses while a commitment term is still running."""
        pt = self._ours(name_or_arn)
        expires = pt.commitment_expires
        if expires and expires > dt.datetime.now(dt.UTC):
            raise SkyhammerError(
                f"{pt.name} is committed until {expires:%Y-%m-%d}; Bedrock won't delete it sooner"
            )
        self.bedrock.delete_provisioned_model_throughput(provisionedModelId=pt.arn)

    def base_model_id(self, foundation_model_id: str) -> str:
        """The Provisioned Throughput variant of a base model: Bedrock sells capacity per
        context length (``amazon.nova-micro-v1:0:128k``); the longest is picked."""
        models = self.bedrock.list_foundation_models(byInferenceType="PROVISIONED")
        ids = [
            m["modelId"]
            for m in models["modelSummaries"]
            if m["modelId"] == foundation_model_id
            or m["modelId"].startswith(f"{foundation_model_id}:")
        ]
        if not ids:
            raise SkyhammerError(
                f"Bedrock doesn't sell Provisioned Throughput for {foundation_model_id} "
                f"in {self.region}"
            )
        best: str = max(ids, key=_context_tokens)
        return best


def _context_tokens(model_id: str) -> int:
    suffix = model_id.rsplit(":", 1)[-1].lower()
    scale = {"k": 1_000, "m": 1_000_000}.get(suffix[-1:], 0)
    try:
        return int(suffix[:-1]) * scale if scale else 0
    except ValueError:
        return 0
