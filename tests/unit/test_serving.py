from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.validate import validate_parameters
from pydantic import ValidationError

from skyhammer.backends.sagemaker import SageMakerServerlessBackend
from skyhammer.config import Settings
from skyhammer.dataset import Dataset
from skyhammer.errors import SkyhammerError
from skyhammer.jobs import ModelRef
from skyhammer.llm import LLM
from skyhammer.serving import (
    EndpointNames,
    Scaling,
    choose_instance,
    endpoint_config_request,
    inference_component_request,
    scale_from_zero_alarm,
    scaling_plan,
)
from skyhammer.session import Session

NAMES = EndpointNames.for_deployment("support-gemma")
TAGS = [{"Key": "skyhammer:name", "Value": "support-gemma"}]

MPG = "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/m"


def _valid(service: str, op: str, kwargs: dict[str, Any]) -> None:
    """Validate request params against the botocore operation's input shape (no network)."""
    client = boto3.client(service, region_name="us-east-1")
    api_name = client.meta.method_to_api_mapping[op]
    validate_parameters(kwargs, client.meta.service_model.operation_model(api_name).input_shape)


def test_defaults_are_serverless_like() -> None:
    s = Scaling()
    assert (s.min_copies, s.max_copies, s.scales_to_zero) == (0, 1, True)
    with pytest.raises(ValidationError):
        Scaling(min_copies=3, max_copies=2)
    with pytest.raises(ValidationError):
        Scaling(bogus=1)  # type: ignore[call-arg]


def test_choose_instance() -> None:
    assert choose_instance(8).instance_type == "ml.g6e.xlarge"
    big = choose_instance(31)
    assert (big.instance_type, big.gpus) == ("ml.g6e.12xlarge", 4)
    assert choose_instance(8, instance_type="ml.p5.48xlarge").gpus == 8
    with pytest.raises(ValueError, match="No single instance"):
        choose_instance(2000)


def test_endpoint_and_component_requests_are_schema_valid() -> None:
    choice = choose_instance(31)
    cfg = endpoint_config_request(
        NAMES,
        choice,
        Scaling(),
        role_arn="arn:aws:iam::1:role/r",
        kms_key_arn="arn:aws:kms:us-east-1:1:key/k",
        tags=TAGS,
    )
    assert cfg["ProductionVariants"][0]["ManagedInstanceScaling"]["MinInstanceCount"] == 0
    _valid("sagemaker", "create_endpoint_config", cfg)
    ic = inference_component_request(NAMES, choice, tags=TAGS)
    assert (
        ic["Specification"]["ComputeResourceRequirements"]["NumberOfAcceleratorDevicesRequired"]
        == 4
    )
    _valid("sagemaker", "create_inference_component", ic)


def test_scale_to_zero_plan() -> None:
    plan = scaling_plan(NAMES, Scaling(max_copies=4, target_invocations_per_copy=10))
    assert plan.register_target["MinCapacity"] == 0
    assert plan.register_target["MaxCapacity"] == 4
    _valid("application-autoscaling", "register_scalable_target", plan.register_target)
    for policy in plan.policies:
        _valid("application-autoscaling", "put_scaling_policy", policy)
    assert plan.scale_from_zero_policy is not None
    _valid("application-autoscaling", "put_scaling_policy", plan.scale_from_zero_policy)
    alarm = scale_from_zero_alarm(NAMES, "arn:aws:autoscaling:us-east-1:1:scalingPolicy:x")
    assert alarm["MetricName"] == "NoCapacityInvocationFailures"
    _valid("cloudwatch", "put_metric_alarm", alarm)


def test_warm_plan_has_no_scale_from_zero_and_keeps_custom_policies() -> None:
    custom = {
        "PolicyName": "cpu",
        "PolicyType": "TargetTrackingScaling",
        "TargetTrackingScalingPolicyConfiguration": {
            "TargetValue": 70.0,
            "CustomizedMetricSpecification": {
                "MetricName": "CPUUtilization",
                "Namespace": "/aws/sagemaker/Endpoints",
                "Statistic": "Average",
            },
        },
    }
    plan = scaling_plan(NAMES, Scaling(min_copies=1, max_copies=3, custom_policies=[custom]))
    assert plan.scale_from_zero_policy is None
    assert [p["PolicyName"] for p in plan.policies] == ["support-gemma-ic-invocations", "cpu"]
    assert plan.policies[1]["ResourceId"] == "inference-component/support-gemma-ic"
    for policy in plan.policies:
        _valid("application-autoscaling", "put_scaling_policy", policy)


def test_mlflow_config_passed_to_sagemaker_jobs(settings: Settings, sft_file: Path) -> None:
    arn = "arn:aws:sagemaker:us-east-1:123456789012:mlflow-app/app-ABC"
    session = Session(settings.model_copy(update={"mlflow_arn": arn}))
    request, _ = LLM("qwen3-8b", session=session).plan(
        "sft", Dataset.from_file(sft_file), output_model="support-qwen-v1"
    )
    backend = SageMakerServerlessBackend(session)
    kw = backend.build_request(request, "s3://b/t", None, "s3://b/o/", model_package_group_arn=MPG)
    assert kw["MlflowConfig"] == {
        "MlflowResourceArn": arn,
        "MlflowExperimentName": "support-qwen-v1",
        "MlflowRunName": request.job_name,
    }
    _valid("sagemaker", "create_training_job", kw)


def test_no_mlflow_by_default(session: Session, sft_file: Path) -> None:
    request, _ = LLM("qwen3-8b", session=session).plan(
        "sft", Dataset.from_file(sft_file), output_model="m"
    )
    assert "MlflowConfig" not in SageMakerServerlessBackend(session).build_request(
        request, "s3://b/t", None, "s3://b/o/", model_package_group_arn=MPG
    )


def test_scaling_rejected_for_bedrock_served_models(session: Session) -> None:
    ref = ModelRef(
        name="m",
        model_arn="arn:aws:bedrock:us-east-1:1:custom-model/m",
        backend="bedrock",
        base_alias="nova-micro",
        region="us-east-1",
    )
    with pytest.raises(SkyhammerError, match="already serverless"):
        LLM.from_model_ref(ref, session).deploy(scaling={"min_copies": 1})
