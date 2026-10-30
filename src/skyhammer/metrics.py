"""Serving metrics from CloudWatch: traffic, latency, tokens and errors for any model.

    LLM("nova-micro").metrics(hours=24)          # serverless (by invoked model/profile ID)
    LLM.get("support-qwen-v1").metrics()          # fine-tuned: Bedrock or SageMaker endpoint

Bedrock reports per invoked ID (inference profile, model ID or imported-model ARN), in
``AWS/Bedrock``. SageMaker inference components report in ``AWS/SageMaker``. For shared
serverless models the numbers cover every caller in the account and region, not just this
process; fine-tuned models and dedicated endpoints have their own IDs.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skyhammer.session import Session


@dataclass(frozen=True)
class ServingMetrics:
    model_id: str
    hours: float
    invocations: int
    client_errors: int
    server_errors: int
    throttles: int | None  # Bedrock only
    latency_ms_avg: float | None
    latency_ms_p50: float | None
    latency_ms_p99: float | None
    input_tokens: int | None  # Bedrock only
    output_tokens: int | None

    @property
    def error_rate(self) -> float:
        return (
            (self.client_errors + self.server_errors) / self.invocations
            if self.invocations
            else 0.0
        )


# (key, metric, stat, scale): scale converts the metric's unit to ours.
_BEDROCK = [
    ("invocations", "Invocations", "Sum", 1.0),
    ("client_errors", "InvocationClientErrors", "Sum", 1.0),
    ("server_errors", "InvocationServerErrors", "Sum", 1.0),
    ("throttles", "InvocationThrottles", "Sum", 1.0),
    ("latency_ms_avg", "InvocationLatency", "Average", 1.0),
    ("latency_ms_p50", "InvocationLatency", "p50", 1.0),
    ("latency_ms_p99", "InvocationLatency", "p99", 1.0),
    ("input_tokens", "InputTokenCount", "Sum", 1.0),
    ("output_tokens", "OutputTokenCount", "Sum", 1.0),
]
_SAGEMAKER = [  # inference components; ModelLatency is in microseconds
    ("invocations", "Invocations", "Sum", 1.0),
    ("client_errors", "Invocation4XXErrors", "Sum", 1.0),
    ("server_errors", "Invocation5XXErrors", "Sum", 1.0),
    ("latency_ms_avg", "ModelLatency", "Average", 1e-3),
    ("latency_ms_p50", "ModelLatency", "p50", 1e-3),
    ("latency_ms_p99", "ModelLatency", "p99", 1e-3),
]


def queries(
    namespace: str, dimension: dict[str, str], spec: list[tuple[str, str, str, float]], period: int
) -> list[dict[str, Any]]:
    return [
        {
            "Id": key,
            "MetricStat": {
                "Metric": {
                    "Namespace": namespace,
                    "MetricName": metric,
                    "Dimensions": [dimension],
                },
                "Period": period,
                "Stat": stat,
            },
            "ReturnData": True,
        }
        for key, metric, stat, _ in spec
    ]


def fetch(
    session: Session, model_id: str, *, component: str | None = None, hours: float = 24
) -> ServingMetrics:
    """Totals and latency percentiles over the last ``hours``. ``component`` selects a
    SageMaker inference component instead of a Bedrock model ID."""
    if component:
        namespace, dim, spec = (
            "AWS/SageMaker",
            {"Name": "InferenceComponentName", "Value": component},
            _SAGEMAKER,
        )
    else:
        namespace, dim, spec = "AWS/Bedrock", {"Name": "ModelId", "Value": model_id}, _BEDROCK
    end = dt.datetime.now(dt.UTC).replace(second=0, microsecond=0)
    period = max(60, int(hours * 3600) // 60 * 60)  # one datapoint for the whole window
    resp = session.client("cloudwatch").get_metric_data(
        MetricDataQueries=queries(namespace, dim, spec, period),
        StartTime=end - dt.timedelta(seconds=period),
        EndTime=end,
    )
    values: dict[str, float | None] = {}
    scale = {key: s for key, _, _, s in spec}
    for r in resp.get("MetricDataResults", []):
        vals = r.get("Values") or []
        values[r["Id"]] = sum(vals) * scale[r["Id"]] if vals else None

    def total(key: str) -> int:
        return int(values.get(key) or 0)

    def opt_total(key: str) -> int | None:
        return int(values[key] or 0) if key in scale else None

    return ServingMetrics(
        model_id=component or model_id,
        hours=period / 3600,
        invocations=total("invocations"),
        client_errors=total("client_errors"),
        server_errors=total("server_errors"),
        throttles=opt_total("throttles"),
        latency_ms_avg=values.get("latency_ms_avg"),
        latency_ms_p50=values.get("latency_ms_p50"),
        latency_ms_p99=values.get("latency_ms_p99"),
        input_tokens=opt_total("input_tokens"),
        output_tokens=opt_total("output_tokens"),
    )
