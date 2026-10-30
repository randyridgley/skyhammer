"""Actual spend per model from Cost Explorer, and per-model budgets with alerts (OP-4, OP-6).

Everything Skyhammer creates carries ``skyhammer:name`` (the model), ``skyhammer:base-model``
and ``skyhammer:technique`` tags. Billing can group by them once they're activated as cost
allocation tags (``activate_tags``, from the management account; data shows up within
24 hours and covers spend from then on).

Cost Explorer bills $0.01 per request; budgets are free for the first two per account,
then $0.02 per budget per day.
"""

from __future__ import annotations

import contextlib
import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skyhammer.errors import SkyhammerError

if TYPE_CHECKING:
    from skyhammer.session import Session

TAGS = ("skyhammer:name", "skyhammer:base-model", "skyhammer:technique")
BY = {
    "name": "skyhammer:name",
    "base-model": "skyhammer:base-model",
    "technique": "skyhammer:technique",
}
BUDGET_PREFIX = "skyhammer-"


@dataclass(frozen=True)
class SpendRow:
    key: str  # the tag value: model name, base model or technique
    usd: float
    services: dict[str, float]


def tag_status(session: Session) -> dict[str, str]:
    """``{"skyhammer:name": "Active" | "Inactive", ...}`` (missing: never seen in billing)."""
    ce = session.client("ce", region="us-east-1")
    resp = ce.list_cost_allocation_tags(TagKeys=list(TAGS))
    return {t["TagKey"]: t["Status"] for t in resp.get("CostAllocationTags", [])}


def activate_tags(session: Session) -> dict[str, str]:
    """Activate Skyhammer's tags for cost allocation (management account only)."""
    ce = session.client("ce", region="us-east-1")
    resp = ce.update_cost_allocation_tags_status(
        CostAllocationTagsStatus=[{"TagKey": t, "Status": "Active"} for t in TAGS]
    )
    errors = resp.get("Errors") or []
    if errors:
        raise SkyhammerError(f"Couldn't activate cost allocation tags: {errors}")
    return tag_status(session)


def spend_request(by: str, start: dt.date, end: dt.date) -> dict[str, Any]:
    """GetCostAndUsage kwargs: unblended cost, grouped by a Skyhammer tag and service, only
    for spend that carries the tag."""
    if by not in BY:
        raise ValueError(f"by must be one of {sorted(BY)}")
    tag = BY[by]
    return {
        "TimePeriod": {"Start": start.isoformat(), "End": end.isoformat()},
        "Granularity": "MONTHLY",
        "Metrics": ["UnblendedCost"],
        "GroupBy": [{"Type": "TAG", "Key": tag}, {"Type": "DIMENSION", "Key": "SERVICE"}],
        "Filter": {"Not": {"Tags": {"Key": tag, "MatchOptions": ["ABSENT"]}}},
    }


def report(session: Session, *, by: str = "name", days: int = 30) -> list[SpendRow]:
    """Spend over the last ``days`` (Cost Explorer lags by up to a day), largest first."""
    end = dt.date.today() + dt.timedelta(days=1)  # End is exclusive; include today
    start = end - dt.timedelta(days=days)
    ce = session.client("ce", region="us-east-1")
    kwargs = spend_request(by, start, end)
    totals: dict[str, dict[str, float]] = {}
    while True:
        resp = ce.get_cost_and_usage(**kwargs)
        for period in resp.get("ResultsByTime", []):
            for group in period.get("Groups", []):
                tag_value, service = group["Keys"]
                key = tag_value.split("$", 1)[1] if "$" in tag_value else tag_value
                if not key:
                    continue  # untagged spend
                amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
                by_service = totals.setdefault(key, {})
                by_service[service] = by_service.get(service, 0.0) + amount
        token = resp.get("NextPageToken")
        if not token:
            break
        kwargs["NextPageToken"] = token
    rows = [SpendRow(k, sum(v.values()), v) for k, v in totals.items()]
    return sorted(rows, key=lambda r: r.usd, reverse=True)


def budget_request(
    account_id: str, model: str, monthly_usd: float, emails: list[str]
) -> dict[str, Any]:
    """CreateBudget kwargs: a monthly cost budget on one model's ``skyhammer:name`` tag that
    emails at 80% and 100% of actual spend and 100% of forecast."""
    if monthly_usd <= 0:
        raise ValueError("monthly_usd must be positive")
    if not emails:
        raise ValueError("at least one email is needed for budget alerts")
    subscribers = [{"SubscriptionType": "EMAIL", "Address": e} for e in emails]

    def alert(kind: str, percent: float) -> dict[str, Any]:
        return {
            "Notification": {
                "NotificationType": kind,
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": percent,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": subscribers,
        }

    return {
        "AccountId": account_id,
        "Budget": {
            "BudgetName": f"{BUDGET_PREFIX}{model}"[:100],
            "BudgetLimit": {"Amount": f"{monthly_usd:.2f}", "Unit": "USD"},
            "TimeUnit": "MONTHLY",
            "BudgetType": "COST",
            "CostFilters": {"TagKeyValue": [f"user:skyhammer:name${model}"]},
        },
        "NotificationsWithSubscribers": [
            alert("ACTUAL", 80.0),
            alert("ACTUAL", 100.0),
            alert("FORECASTED", 100.0),
        ],
    }


def _account(session: Session) -> str:
    return str(session.client("sts").get_caller_identity()["Account"])


def create_budget(session: Session, model: str, monthly_usd: float, emails: list[str]) -> str:
    """Create (or replace) the monthly budget for ``model``; returns the budget name."""
    budgets = session.client("budgets", region="us-east-1")
    req = budget_request(_account(session), model, monthly_usd, emails)
    name = req["Budget"]["BudgetName"]
    delete_budget(session, model)
    budgets.create_budget(**req)
    return str(name)


def list_budgets(session: Session) -> list[dict[str, Any]]:
    budgets = session.client("budgets", region="us-east-1")
    out = []
    for page in budgets.get_paginator("describe_budgets").paginate(AccountId=_account(session)):
        for b in page.get("Budgets", []):
            if b["BudgetName"].startswith(BUDGET_PREFIX):
                out.append(b)
    return out


def delete_budget(session: Session, model: str) -> None:
    budgets = session.client("budgets", region="us-east-1")
    with contextlib.suppress(budgets.exceptions.NotFoundException):
        budgets.delete_budget(
            AccountId=_account(session), BudgetName=f"{BUDGET_PREFIX}{model}"[:100]
        )
