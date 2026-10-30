from typing import Any

import pytest
from botocore.stub import ANY, Stubber

from skyhammer import LLM
from skyhammer.metrics import fetch
from skyhammer.session import Session


def results(values: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "MetricDataResults": [
            {"Id": k, "Label": k, "Timestamps": [], "Values": v, "StatusCode": "Complete"}
            for k, v in values.items()
        ]
    }


def stub_cloudwatch(session: Session, monkeypatch: pytest.MonkeyPatch, resp: dict[str, Any]) -> Any:
    cw = session.client("cloudwatch")
    stub = Stubber(cw)
    stub.add_response(
        "get_metric_data",
        resp,
        {"MetricDataQueries": ANY, "StartTime": ANY, "EndTime": ANY},
    )
    stub.activate()
    real = session.client
    monkeypatch.setattr(
        session, "client", lambda svc, region=None: cw if svc == "cloudwatch" else real(svc, region)
    )
    return stub


def test_bedrock_metrics(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_cloudwatch(
        session,
        monkeypatch,
        results(
            {
                "invocations": [291.0],
                "client_errors": [7.0],
                "server_errors": [],
                "throttles": [0.0],
                "latency_ms_avg": [440.2],
                "latency_ms_p50": [431.0],
                "latency_ms_p99": [700.0],
                "input_tokens": [8573.0],
                "output_tokens": [7995.0],
            }
        ),
    )
    m = LLM("nova-micro", session=session).metrics(hours=6)
    assert m.model_id.endswith("nova-micro-v1:0") and m.hours == 6
    assert (m.invocations, m.client_errors, m.server_errors, m.throttles) == (291, 7, 0, 0)
    assert m.latency_ms_p99 == 700.0 and m.input_tokens == 8573
    assert m.error_rate == pytest.approx(7 / 291)


def test_sagemaker_component_latency_is_converted_to_ms(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_cloudwatch(
        session,
        monkeypatch,
        results({"invocations": [3.0], "latency_ms_avg": [1_500_000.0], "server_errors": [1.0]}),
    )
    m = fetch(session, "arn:aws:sagemaker:...", component="ep-ic", hours=1)
    assert m.model_id == "ep-ic" and m.latency_ms_avg == 1500.0
    assert m.throttles is None and m.input_tokens is None  # not reported by SageMaker
