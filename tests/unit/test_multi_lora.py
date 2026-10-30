"""SV-8: LoRA adapters served from one shared SageMaker endpoint."""

from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.validate import validate_parameters

from skyhammer.backends.sagemaker import resolve_adapter_dir
from skyhammer.errors import SkyhammerError
from skyhammer.serving import EndpointDeployer, EndpointNames, adapter_component_request

SM = boto3.client("sagemaker", region_name="us-east-1")


def test_adapter_names_live_on_the_shared_endpoint() -> None:
    names = EndpointNames.for_adapter("qwen-shared", "support-v1")
    assert (names.endpoint, names.component, names.adapter) == (
        "qwen-shared",
        "qwen-shared-support-v1",
        True,
    )
    assert names.deployment_name == "qwen-shared/qwen-shared-support-v1"
    # A recorded deployment name round-trips (LLM.get -> chat routes to the adapter).
    assert EndpointNames.for_deployment(names.deployment_name) == names
    assert not EndpointNames.for_deployment("qwen-shared").adapter


def test_adapter_request_is_schema_valid() -> None:
    names = EndpointNames.for_adapter("qwen-shared", "support-v1")
    req = adapter_component_request(names, "s3://b/jobs/x/checkpoints/hf", tags=[])
    spec = req["Specification"]
    assert spec == {
        "BaseInferenceComponentName": "qwen-shared-ic",
        "Container": {"ArtifactUrl": "s3://b/jobs/x/checkpoints/hf/"},
    }
    assert "RuntimeConfig" not in req  # adapters take no copies or compute of their own
    validate_parameters(
        req, SM.meta.service_model.operation_model("CreateInferenceComponent").input_shape
    )


class FakeS3:
    exceptions = type("E", (), {"ClientError": ClientError})

    def __init__(self, keys: set[str]) -> None:
        self.keys = keys

    def head_object(self, Bucket: str, Key: str) -> None:
        if Key not in self.keys:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")


def test_adapter_dir_comes_from_unmerged_lora_output() -> None:
    s3 = FakeS3({"jobs/x/out/checkpoints/hf/adapter_config.json"})
    assert resolve_adapter_dir(s3, "s3://b/jobs/x/out") == "s3://b/jobs/x/out/checkpoints/hf/"
    with pytest.raises(SkyhammerError, match='merge_weights": False'):
        resolve_adapter_dir(FakeS3(set()), "s3://b/jobs/y/out")


def test_deleting_an_adapter_leaves_the_endpoint() -> None:
    calls: list[str] = []

    class FakeSM:
        def describe_inference_component(self, InferenceComponentName: str) -> dict[str, Any]:
            if InferenceComponentName in calls:  # deleted: gone on the next describe
                raise ClientError(
                    {"Error": {"Code": "ValidationException", "Message": "Could not find"}},
                    "DescribeInferenceComponent",
                )
            return {}

        def delete_inference_component(self, InferenceComponentName: str) -> None:
            calls.append(InferenceComponentName)

        def delete_endpoint(self, **_: Any) -> None:
            raise AssertionError("the shared endpoint must stay")

    deployer = EndpointDeployer.__new__(EndpointDeployer)
    deployer.sm = FakeSM()
    deployer.delete(EndpointNames.for_adapter("qwen-shared", "support-v1"))
    assert calls == ["qwen-shared-support-v1"]


def test_deleting_a_shared_endpoint_removes_its_adapters_first() -> None:
    order: list[str] = []
    gone: set[str] = set()

    class FakeSM:
        def get_paginator(self, op: str) -> Any:
            assert op == "list_inference_components"
            names = ["shared-ic", "shared-support-v1", "shared-support-v2"]

            class P:
                def paginate(self, **kw: Any) -> list[dict[str, Any]]:
                    assert kw == {"EndpointNameEquals": "shared"}
                    return [{"InferenceComponents": [{"InferenceComponentName": n} for n in names]}]

            return P()

        def describe_inference_component(self, InferenceComponentName: str) -> dict[str, Any]:
            if InferenceComponentName in gone:
                raise ClientError(
                    {"Error": {"Code": "ValidationException", "Message": "Could not find"}},
                    "DescribeInferenceComponent",
                )
            return {}

        def delete_inference_component(self, InferenceComponentName: str) -> None:
            order.append(InferenceComponentName)
            gone.add(InferenceComponentName)

        def __getattr__(self, name: str) -> Any:  # endpoint/config/model: already gone
            def missing(**_: Any) -> Any:
                raise ClientError(
                    {"Error": {"Code": "ValidationException", "Message": "Could not find"}}, name
                )

            return missing

    class Quiet:
        def deregister_scalable_target(self, **_: Any) -> None: ...
        def delete_alarms(self, **_: Any) -> None: ...

    deployer = EndpointDeployer.__new__(EndpointDeployer)
    deployer.sm, deployer.autoscaling, deployer.cloudwatch = FakeSM(), Quiet(), Quiet()
    deployer.delete(EndpointNames.for_deployment("shared"))
    assert order == ["shared-support-v1", "shared-support-v2", "shared-ic"]
