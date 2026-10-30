import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from botocore.validate import validate_parameters
from pydantic import ValidationError

from skyhammer.backends.sagemaker import SageMakerServerlessBackend
from skyhammer.config import Settings
from skyhammer.dataset import Dataset
from skyhammer.errors import SkyhammerError
from skyhammer.hub import HostingConfig
from skyhammer.jobs import ModelRef
from skyhammer.llm import LLM
from skyhammer.serving import (
    GPU_INSTANCES,
    MAX_INSTANCE_POOLS,
    EndpointNames,
    InstanceChoice,
    Scaling,
    choose_instance,
    endpoint_config_request,
    fallback_instances,
    inference_component_request,
    scale_from_zero_alarm,
    scaling_plan,
)
from skyhammer.session import Session

NAMES = EndpointNames.for_deployment("support-gemma")
TAGS = [{"Key": "skyhammer:name", "Value": "support-gemma"}]

FIXTURES = Path(__file__).parents[1] / "fixtures"
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
    small = choose_instance(8)  # ~21 GB: fits one 24 GB GPU
    assert small.instance_type == "ml.g6.xlarge"
    assert small.fallbacks[:2] == ("ml.g5.xlarge", "ml.g6.2xlarge")
    big = choose_instance(31)
    assert (big.instance_type, big.gpus) == ("ml.g6.12xlarge", 4)
    assert all(GPU_INSTANCES[f].gpus == 4 for f in big.fallbacks)
    assert choose_instance(8, instance_type="ml.p5.48xlarge").gpus == 8
    with pytest.raises(ValueError, match="No single instance"):
        choose_instance(2000)


HOSTING = HostingConfig(
    image_uri="img", instance_type="ml.g6e.2xlarge", gpus=1, memory_mb=20_000, cpus=4
)


def test_fallbacks_fit_the_same_component() -> None:
    # Unknown model size: keep at least the primary's GPU memory (48 GB L40S).
    auto = InstanceChoice.from_hosting(HOSTING)
    assert auto.instance_type == "ml.g6e.2xlarge"
    assert auto.fallbacks == ("ml.g6e.4xlarge", "ml.g6e.8xlarge")  # >= primary host size
    # A 4B model needs ~10 GB, so 24 GB GPUs (other families, other capacity) qualify,
    # on hosts at least as big as the validated ml.g6e.2xlarge (64 GB, 8 vCPUs).
    sized = InstanceChoice.from_hosting(HOSTING, params_b=4)
    assert len(sized.instance_types) == MAX_INSTANCE_POOLS
    assert {"ml.g6.4xlarge", "ml.g5.4xlarge"} <= set(sized.fallbacks)
    for itype in sized.fallbacks:
        spec = GPU_INSTANCES[itype]
        assert spec.gpus == 1 and spec.memory_gb >= 64 and spec.vcpus >= 8
    assert fallback_instances("ml.unknown", gpus=1, memory_mb=1) == []


def test_instance_type_pins_or_orders_fallbacks() -> None:
    pinned = InstanceChoice.from_hosting(HOSTING, "ml.g5.2xlarge")
    assert (pinned.instance_type, pinned.fallbacks) == ("ml.g5.2xlarge", ())
    ordered = InstanceChoice.from_hosting(HOSTING, ["ml.g5.2xlarge", "ml.g6.2xlarge"])
    assert ordered.instance_types == ["ml.g5.2xlarge", "ml.g6.2xlarge"]
    with pytest.raises(ValueError, match="1-5 types"):
        InstanceChoice.from_hosting(HOSTING, [f"ml.g5.{i}xlarge" for i in range(6)])


def test_endpoint_config_uses_instance_pools_for_fallbacks() -> None:
    kw: dict[str, Any] = {"role_arn": "arn:aws:iam::1:role/r", "kms_key_arn": None, "tags": TAGS}
    pooled = endpoint_config_request(
        NAMES, InstanceChoice.from_hosting(HOSTING, params_b=4), Scaling(), **kw
    )
    variant = pooled["ProductionVariants"][0]
    assert "InstanceType" not in variant
    assert [p["Priority"] for p in variant["InstancePools"]] == [1, 2, 3, 4, 5]
    assert variant["InstancePools"][0]["InstanceType"] == "ml.g6e.2xlarge"
    assert 300 <= variant["VariantInstanceProvisionTimeoutInSeconds"] <= 3600
    _valid("sagemaker", "create_endpoint_config", pooled)
    single = endpoint_config_request(
        NAMES, InstanceChoice.from_hosting(HOSTING, "ml.g5.2xlarge"), Scaling(), **kw
    )
    variant = single["ProductionVariants"][0]
    assert variant["InstanceType"] == "ml.g5.2xlarge" and "InstancePools" not in variant
    _valid("sagemaker", "create_endpoint_config", single)


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


def test_endpoint_config_matches_model_network_isolation() -> None:
    # CreateInferenceComponent rejects a model and endpoint config that disagree (seen live).
    kw: dict[str, Any] = {"role_arn": "arn:aws:iam::1:role/r", "kms_key_arn": None, "tags": TAGS}
    choice = InstanceChoice.from_hosting(HOSTING)
    isolated = endpoint_config_request(NAMES, choice, Scaling(), network_isolation=True, **kw)
    assert isolated["EnableNetworkIsolation"] is True
    _valid("sagemaker", "create_endpoint_config", isolated)
    assert "EnableNetworkIsolation" not in endpoint_config_request(NAMES, choice, Scaling(), **kw)


def test_instance_types_reads_the_running_pool() -> None:
    # Recorded live 2026-10-30: ml.g6e.2xlarge had no capacity; the first fallback ran.
    from skyhammer.serving import EndpointDeployer

    recorded = json.loads((FIXTURES / "describe_endpoint_instance_pools.json").read_text())
    deployer = EndpointDeployer.__new__(EndpointDeployer)
    deployer.sm = SimpleNamespace(describe_endpoint=lambda **_: recorded)
    assert deployer.instance_types(NAMES) == ["ml.g6.4xlarge"]
    single = {"ProductionVariants": [{"InstanceType": "ml.g5.2xlarge", "CurrentInstanceCount": 1}]}
    deployer.sm = SimpleNamespace(describe_endpoint=lambda **_: single)
    assert deployer.instance_types(NAMES) == ["ml.g5.2xlarge"]
