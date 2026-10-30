import json
from typing import Any

import boto3
import pytest
from moto import mock_aws

from skyhammer.deployments import DeploymentInfo, list_deployments
from skyhammer.errors import SkyhammerError
from skyhammer.imports import preflight
from skyhammer.session import Session


def _model_dir(
    s3: Any,
    arch: str = "Qwen3ForCausalLM",
    ctx: int = 32768,
    files: tuple[str, ...] = ("model.safetensors", "tokenizer.json"),
) -> str:
    s3.create_bucket(Bucket="skyhammer-weights")
    s3.put_object(
        Bucket="skyhammer-weights",
        Key="m/config.json",
        Body=json.dumps({"architectures": [arch], "max_position_embeddings": ctx}),
    )
    for f in files:
        s3.put_object(Bucket="skyhammer-weights", Key=f"m/{f}", Body=b"x" * 10)
    return "s3://skyhammer-weights/m"


@mock_aws
def test_preflight_accepts_supported_model() -> None:
    s3 = boto3.client("s3", region_name="us-east-1")
    info = preflight(s3, _model_dir(s3))
    assert info["architectures"] == ["Qwen3ForCausalLM"] and info["context"] == 32768


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"arch": "Gemma3ForCausalLM"}, "isn't supported by Bedrock Custom Model Import"),
        ({"ctx": 262144}, "exceeds the 128K import limit"),
        ({"files": ("tokenizer.json",)}, "no .safetensors"),
        ({"files": ("model.safetensors",)}, "no tokenizer"),
    ],
)
@mock_aws
def test_preflight_rejects_before_import(kwargs: dict[str, Any], match: str) -> None:
    s3 = boto3.client("s3", region_name="us-east-1")
    with pytest.raises(SkyhammerError, match=match):
        preflight(s3, _model_dir(s3, **kwargs))


class Fake:
    def __init__(self, **responses: Any) -> None:
        self.responses = responses

    def __getattr__(self, op: str) -> Any:
        r = self.responses[op]
        return (lambda **kw: r(**kw)) if callable(r) else (lambda **kw: r)

    def get_paginator(self, op: str) -> Any:
        r = self.responses[op]
        return type("P", (), {"paginate": staticmethod(lambda **kw: [r])})()


def test_only_skyhammer_tagged_resources_are_listed(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    tags = {
        "arn:ep/mine": [{"Key": "skyhammer:name", "Value": "support-gemma"}],
        "arn:ep/theirs": [{"Key": "team", "Value": "other"}],
    }
    sm = Fake(
        list_endpoints={
            "Endpoints": [
                {
                    "EndpointName": "mine",
                    "EndpointArn": "arn:ep/mine",
                    "EndpointStatus": "InService",
                },
                {
                    "EndpointName": "theirs",
                    "EndpointArn": "arn:ep/theirs",
                    "EndpointStatus": "InService",
                },
            ]
        },
        list_tags=lambda ResourceArn: {"Tags": tags[ResourceArn]},
    )
    bedrock = Fake(
        list_custom_model_deployments={
            "modelDeploymentSummaries": [
                {
                    "customModelDeploymentArn": "arn:dep/x",
                    "customModelDeploymentName": "x",
                    "status": "Active",
                }
            ]
        },
        list_tags_for_resource={"tags": []},
    )
    cw = Fake(get_metric_statistics={"Datapoints": []})
    clients = {"sagemaker": sm, "bedrock": bedrock, "cloudwatch": cw}
    monkeypatch.setattr(session, "client", lambda svc, region=None: clients[svc])
    found = list_deployments(session)
    assert found == [
        DeploymentInfo("sagemaker-endpoint", "mine", "arn:ep/mine", "InService", "support-gemma", 0)
    ]
