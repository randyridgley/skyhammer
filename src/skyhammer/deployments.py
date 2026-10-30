"""List and prune Skyhammer deployments (only resources tagged ``skyhammer:name``)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skyhammer.serving import EndpointDeployer, EndpointNames

if TYPE_CHECKING:
    from skyhammer.session import Session

TAG = "skyhammer:name"


@dataclass(frozen=True)
class DeploymentInfo:
    kind: str  # "sagemaker-endpoint" | "bedrock-custom-deployment"
    name: str
    arn: str
    status: str
    model: str
    invocations: int | None = None  # over the idle window


def _invocations(
    cw: Any, namespace: str, metric: str, dims: list[dict[str, str]], days: float
) -> int:
    end = dt.datetime.now(dt.UTC)
    resp = cw.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric,
        Dimensions=dims,
        StartTime=end - dt.timedelta(days=days),
        EndTime=end,
        Period=86400,
        Statistics=["Sum"],
    )
    return int(sum(p["Sum"] for p in resp.get("Datapoints", [])))


def list_deployments(session: Session, *, idle_days: float = 7) -> list[DeploymentInfo]:
    sm = session.client("sagemaker")
    bedrock = session.client("bedrock")
    cw = session.client("cloudwatch")
    out: list[DeploymentInfo] = []
    for page in sm.get_paginator("list_endpoints").paginate():
        for ep in page["Endpoints"]:
            tags = {
                t["Key"]: t["Value"]
                for t in sm.list_tags(ResourceArn=ep["EndpointArn"]).get("Tags", [])
            }
            if TAG not in tags:
                continue  # never touch endpoints Skyhammer didn't create
            names = EndpointNames.for_deployment(ep["EndpointName"])
            calls = _invocations(
                cw,
                "AWS/SageMaker",
                "Invocations",
                [{"Name": "InferenceComponentName", "Value": names.component}],
                idle_days,
            )
            out.append(
                DeploymentInfo(
                    "sagemaker-endpoint",
                    ep["EndpointName"],
                    ep["EndpointArn"],
                    ep["EndpointStatus"],
                    tags[TAG],
                    calls,
                )
            )
    for d in bedrock.list_custom_model_deployments().get("modelDeploymentSummaries", []):
        arn = d["customModelDeploymentArn"]
        tags = {
            t["key"]: t["value"]
            for t in bedrock.list_tags_for_resource(resourceARN=arn).get("tags", [])
        }
        if TAG not in tags:
            continue
        calls = _invocations(
            cw, "AWS/Bedrock", "Invocations", [{"Name": "ModelId", "Value": arn}], idle_days
        )
        out.append(
            DeploymentInfo(
                "bedrock-custom-deployment",
                d["customModelDeploymentName"],
                arn,
                d["status"],
                tags[TAG],
                calls,
            )
        )
    return out


def delete_deployment(session: Session, info: DeploymentInfo) -> None:
    if info.kind == "sagemaker-endpoint":
        EndpointDeployer(session, session.region).delete(EndpointNames.for_deployment(info.name))
    else:
        session.client("bedrock").delete_custom_model_deployment(
            customModelDeploymentIdentifier=info.arn
        )
