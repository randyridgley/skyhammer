"""PK-5 sweeper: only prefixed, tagged, old resources; graders only when no RL job runs."""

import datetime as dt
from typing import Any

import pytest

from skyhammer.sweeper import _Finder, find_leftovers, sweep

NOW = dt.datetime.now(dt.UTC)
OLD, NEW = NOW - dt.timedelta(hours=10), NOW - dt.timedelta(hours=1)
TAG = {"skyhammer:name": "x"}


class Pager:
    def __init__(self, key: str, items: list[dict[str, Any]]) -> None:
        self.key, self.items = key, items

    def paginate(self, **_: Any) -> list[dict[str, Any]]:
        return [{self.key: self.items}]


class FakeSageMaker:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.tags = {"arn:ep/skyhammer-it-old": TAG, "arn:ep/skyhammer-it-untagged": {}}

    def get_paginator(self, op: str) -> Pager:
        return {
            "list_endpoints": Pager(
                "Endpoints",
                [
                    {
                        "EndpointName": "skyhammer-it-old",
                        "EndpointArn": "arn:ep/skyhammer-it-old",
                        "CreationTime": OLD,
                    },
                    {
                        "EndpointName": "skyhammer-it-untagged",
                        "EndpointArn": "arn:ep/skyhammer-it-untagged",
                        "CreationTime": OLD,
                    },
                    {
                        "EndpointName": "customer-prod",
                        "EndpointArn": "arn:ep/customer-prod",
                        "CreationTime": OLD,
                    },
                ],
            ),
            "list_endpoint_configs": Pager(
                "EndpointConfigs",
                [
                    {
                        "EndpointConfigName": "skyhammer-it-old-cfg",
                        "CreationTime": OLD,
                    },  # its endpoint's
                    {"EndpointConfigName": "skyhammer-it-orphan-cfg", "CreationTime": OLD},
                ],
            ),
            "list_models": Pager(
                "Models", [{"ModelName": "skyhammer-it-fresh-model", "CreationTime": NEW}]
            ),
            "list_training_jobs": Pager("TrainingJobSummaries", []),
        }[op]

    def list_tags(self, ResourceArn: str) -> dict[str, Any]:
        return {"Tags": [{"Key": k, "Value": v} for k, v in self.tags.get(ResourceArn, {}).items()]}

    def list_training_jobs(self, **_: Any) -> dict[str, Any]:
        return {"TrainingJobSummaries": []}

    def delete_endpoint_config(self, EndpointConfigName: str) -> None:
        self.deleted.append(EndpointConfigName)


class FakeBedrock:
    def __init__(self, rl_running: bool) -> None:
        self.rl_running = rl_running
        self.deleted: list[str] = []

    def list_tags_for_resource(self, resourceARN: str) -> dict[str, Any]:
        return {
            "tags": [{"key": "skyhammer:name", "value": "x"}] if "skyhammer" in resourceARN else []
        }

    def list_custom_model_deployments(self) -> dict[str, Any]:
        return {"modelDeploymentSummaries": []}

    def list_custom_models(self) -> dict[str, Any]:
        return {"modelSummaries": []}

    def list_imported_models(self) -> dict[str, Any]:
        return {
            "modelSummaries": [
                {
                    "modelName": "skyhammer-it-imp",
                    "modelArn": "arn:skyhammer-it-imp",
                    "creationTime": OLD,
                }
            ]
        }

    def list_model_customization_jobs(self, **_: Any) -> dict[str, Any]:
        jobs = [
            {
                "jobName": "user-rft",
                "jobArn": "arn:j",
                "creationTime": NEW,
                "customizationType": "REINFORCEMENT_FINE_TUNING",
            }
        ]
        return {"modelCustomizationJobSummaries": jobs if self.rl_running else []}

    def list_guardrails(self) -> dict[str, Any]:
        return {
            "guardrails": [
                {
                    "name": "skyhammer-it-gr",
                    "arn": "arn:skyhammer-it-gr",
                    "id": "g1",
                    "createdAt": OLD,
                }
            ]
        }

    def delete_imported_model(self, modelIdentifier: str) -> None:
        self.deleted.append(modelIdentifier)

    def delete_guardrail(self, guardrailIdentifier: str) -> None:
        self.deleted.append(guardrailIdentifier)


class FakeLambda:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def get_paginator(self, op: str) -> Pager:
        return Pager(
            "Functions",
            [
                {
                    "FunctionName": "skyhammer-grader-sums-br-1",
                    "FunctionArn": "arn:fn1",
                    "LastModified": "2026-10-29T10:00:00.000+0000",
                },
                {
                    "FunctionName": "someone-elses-fn",
                    "FunctionArn": "arn:fn2",
                    "LastModified": "2026-10-29T10:00:00.000+0000",
                },
            ],
        )

    def list_tags(self, Resource: str) -> dict[str, Any]:
        return {"Tags": TAG}

    def delete_function(self, FunctionName: str) -> None:
        self.deleted.append(FunctionName)


class FakeSession:
    def __init__(self, rl_running: bool = False) -> None:
        self.clients = {
            "sagemaker": FakeSageMaker(),
            "bedrock": FakeBedrock(rl_running),
            "lambda": FakeLambda(),
        }

    def client(self, service: str, region: str | None = None) -> Any:
        return self.clients[service]


def kinds(found: list[Any]) -> list[tuple[str, str]]:
    return sorted((x.kind, x.name) for x in found)


def test_only_prefixed_tagged_and_old_resources() -> None:
    found = find_leftovers(FakeSession(), ["us-east-1"])  # type: ignore[arg-type]
    assert kinds(found) == [
        ("bedrock-guardrail", "skyhammer-it-gr"),
        ("bedrock-imported-model", "skyhammer-it-imp"),
        ("sagemaker-endpoint", "skyhammer-it-old"),
        ("sagemaker-endpoint-config", "skyhammer-it-orphan-cfg"),  # not the live endpoint's
    ]  # no untagged endpoint, no customer-prod, no 1-hour-old model


def test_graders_only_when_asked_and_no_rl_job_runs() -> None:
    assert not any(x.kind == "grader-lambda" for x in find_leftovers(FakeSession(), ["r"]))  # type: ignore[arg-type]
    found = find_leftovers(FakeSession(), ["r"], include_graders=True)  # type: ignore[arg-type]
    assert ("grader-lambda", "skyhammer-grader-sums-br-1") in kinds(found)
    busy = find_leftovers(FakeSession(rl_running=True), ["r"], include_graders=True)  # type: ignore[arg-type]
    assert not any(x.kind == "grader-lambda" for x in busy)


def test_dry_run_deletes_nothing_and_real_run_deletes_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoints: list[str] = []
    monkeypatch.setattr(_Finder, "_delete_endpoint", lambda self, n: endpoints.append(n.endpoint))
    session = FakeSession()
    sweep(session, ["r"], include_graders=True)  # type: ignore[arg-type]
    assert session.clients["bedrock"].deleted == [] and session.clients["lambda"].deleted == []
    removed = sweep(
        session, ["r"], dry_run=False, prefixes=("skyhammer-it-",), include_graders=True
    )  # type: ignore[arg-type]
    order = [x.kind for x in removed]
    assert order.index("sagemaker-endpoint") < order.index("sagemaker-endpoint-config")
    assert order[-1] == "grader-lambda"
    assert session.clients["bedrock"].deleted == ["arn:skyhammer-it-imp", "g1"]
    assert session.clients["lambda"].deleted == ["skyhammer-grader-sums-br-1"]
    assert endpoints == ["skyhammer-it-old"]
