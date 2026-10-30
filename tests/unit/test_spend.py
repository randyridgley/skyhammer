"""OP-4/OP-6: spend per model from Cost Explorer, per-model budgets with alerts."""

import datetime as dt
from typing import Any

import boto3
import pytest
from botocore.validate import validate_parameters

from skyhammer import spend


def valid(service: str, op: str, kwargs: dict[str, Any]) -> None:
    model = boto3.client(service, region_name="us-east-1").meta.service_model
    validate_parameters(kwargs, model.operation_model(op).input_shape)


def test_spend_request_groups_tagged_cost_by_model_and_service() -> None:
    req = spend.spend_request("name", dt.date(2026, 10, 1), dt.date(2026, 10, 31))
    assert req["GroupBy"] == [
        {"Type": "TAG", "Key": "skyhammer:name"},
        {"Type": "DIMENSION", "Key": "SERVICE"},
    ]
    assert req["Filter"] == {"Not": {"Tags": {"Key": "skyhammer:name", "MatchOptions": ["ABSENT"]}}}
    valid("ce", "GetCostAndUsage", req)
    with pytest.raises(ValueError):
        spend.spend_request("team", dt.date(2026, 10, 1), dt.date(2026, 10, 2))


class FakeCE:
    def __init__(self) -> None:
        self.calls = 0

    def get_cost_and_usage(self, **kw: Any) -> dict[str, Any]:
        self.calls += 1
        group = lambda tag, svc, usd: {  # noqa: E731
            "Keys": [f"skyhammer:name${tag}", svc],
            "Metrics": {"UnblendedCost": {"Amount": str(usd), "Unit": "USD"}},
        }
        if self.calls == 1:
            return {
                "ResultsByTime": [
                    {
                        "Groups": [
                            group("support-qwen", "Amazon SageMaker", 12.5),
                            group("support-qwen", "Amazon Bedrock", 1.25),
                            group("", "Amazon Bedrock", 99.0),  # untagged: ignored
                        ]
                    }
                ],
                "NextPageToken": "p2",
            }
        assert kw["NextPageToken"] == "p2"
        return {"ResultsByTime": [{"Groups": [group("support-nova", "Amazon Bedrock", 30.0)]}]}


def test_report_sums_per_model_across_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    session = type("S", (), {"client": lambda self, svc, region=None: ce})()
    ce = FakeCE()
    rows = spend.report(session, days=7)  # type: ignore[arg-type]
    assert [(r.key, r.usd) for r in rows] == [("support-nova", 30.0), ("support-qwen", 13.75)]
    assert rows[1].services == {"Amazon SageMaker": 12.5, "Amazon Bedrock": 1.25}


def test_budget_request_alerts_and_filters_one_model() -> None:
    req = spend.budget_request("123456789012", "support-qwen", 50, ["ops@example.com"])
    assert req["Budget"]["BudgetName"] == "skyhammer-support-qwen"
    assert req["Budget"]["CostFilters"] == {"TagKeyValue": ["user:skyhammer:name$support-qwen"]}
    kinds = [
        (n["Notification"]["NotificationType"], n["Notification"]["Threshold"])
        for n in req["NotificationsWithSubscribers"]
    ]
    assert kinds == [("ACTUAL", 80.0), ("ACTUAL", 100.0), ("FORECASTED", 100.0)]
    valid("budgets", "CreateBudget", req)
    with pytest.raises(ValueError):
        spend.budget_request("1", "m", 0, ["a@b.c"])
    with pytest.raises(ValueError):
        spend.budget_request("1", "m", 10, [])


def test_job_spec_budget(tmp_path: Any) -> None:
    from skyhammer.errors import ConfigError
    from skyhammer.spec import load_spec

    base = (
        "apiVersion: skyhammer/v1\nkind: FineTuningJob\nname: m\nbase_model: nova-micro\n"
        "technique: sft\ndataset: {path: d.jsonl}\n"
    )
    p = tmp_path / "s.yaml"
    p.write_text(base + "budget: {monthly_usd: 25, emails: [ops@example.com]}\n")
    assert load_spec(p).budget.monthly_usd == 25  # type: ignore[union-attr]
    p.write_text(base + "budget: {monthly_usd: 25, emails: []}\n")
    with pytest.raises(ConfigError):
        load_spec(p)
