"""Serverless-feeling SageMaker endpoints: scale to zero by default, user-tunable autoscaling.

A SageMaker-served model is one endpoint hosting one inference component (IC). Inference
components are what make scale-to-zero possible: the endpoint's ``ManagedInstanceScaling``
allows 0 instances, Application Auto Scaling tracks invocations per IC copy, and a
CloudWatch alarm on ``NoCapacityInvocationFailures`` scales out from zero when a request
arrives. While at zero, the first requests fail until an instance is provisioned (several
minutes); the chat shim hides that by retrying with a "warming up" notice.

Everything here is a pure request builder so it can be validated against botocore models.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError, WaiterError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from skyhammer.errors import JobFailedError, SkyhammerError

if TYPE_CHECKING:
    from skyhammer.hub import HostingConfig
    from skyhammer.session import Session

SCALABLE_DIMENSION = "sagemaker:inference-component:DesiredCopyCount"
INVOCATIONS_PER_COPY = "SageMakerInferenceComponentInvocationsPerCopy"


class Scaling(BaseModel):
    """Autoscaling for a SageMaker-served model.

    The defaults behave like serverless: scale to zero when idle, one copy when in use.
    ``min_copies=1`` keeps the model warm (no cold starts, but you pay while idle).
    """

    model_config = ConfigDict(extra="forbid")

    min_copies: int = Field(default=0, ge=0)
    max_copies: int = Field(default=1, ge=1)
    target_invocations_per_copy: float = Field(default=5.0, gt=0)
    scale_in_cooldown_s: int = Field(default=300, ge=0)
    scale_out_cooldown_s: int = Field(default=60, ge=0)
    # Escape hatch: raw Application Auto Scaling policies, e.g.
    # {"PolicyName": ..., "PolicyType": "TargetTrackingScaling",
    #  "TargetTrackingScalingPolicyConfiguration": {...}}. Added after the SDK's policies.
    custom_policies: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _bounds(self) -> Scaling:
        if self.min_copies > self.max_copies:
            raise ValueError("min_copies must be <= max_copies")
        return self

    @property
    def scales_to_zero(self) -> bool:
        return self.min_copies == 0


# GPU instance types for LLM serving: (instance type, GPUs, GPU memory per GPU in GB).
# TODO(M2): confirm per-region availability and quotas; allow per-model overrides in catalog.
_INSTANCES = [
    ("ml.g6e.xlarge", 1, 48),
    ("ml.g6e.12xlarge", 4, 48),
    ("ml.g6e.48xlarge", 8, 48),
    ("ml.p5.48xlarge", 8, 80),
]


@dataclass(frozen=True)
class InstanceChoice:
    instance_type: str
    gpus: int
    memory_mb: int
    cpus: int | None = None

    @classmethod
    def from_hosting(
        cls, hosting: HostingConfig, instance_type: str | None = None
    ) -> InstanceChoice:
        """AWS's validated hosting profile for the recipe (Hub ``HostingConfigs``)."""
        return cls(
            instance_type or hosting.instance_type, hosting.gpus, hosting.memory_mb, hosting.cpus
        )


def choose_instance(params_b: float, *, instance_type: str | None = None) -> InstanceChoice:
    """Smallest instance whose GPU memory fits bf16 weights plus ~30% for KV cache."""
    needed_gb = params_b * 2 * 1.3
    for itype, gpus, mem in _INSTANCES:
        if instance_type and itype != instance_type:
            continue
        if instance_type or gpus * mem >= needed_gb:
            # MinMemoryRequiredInMb reserves *host* RAM for the copy (weights load through it).
            return InstanceChoice(itype, gpus, memory_mb=max(8192, int(needed_gb * 1024 // 4)))
    if instance_type:
        raise ValueError(
            f"Unknown instance type {instance_type!r}; known: "
            + ", ".join(i[0] for i in _INSTANCES)
        )
    raise ValueError(f"No single instance fits a {params_b}B model; pass instance_type=")


@dataclass(frozen=True)
class EndpointNames:
    endpoint: str
    config: str
    component: str
    model: str
    variant: str = "AllTraffic"

    @classmethod
    def for_deployment(cls, name: str) -> EndpointNames:
        return cls(
            endpoint=name, config=f"{name}-cfg", component=f"{name}-ic", model=f"{name}-model"
        )


def endpoint_config_request(
    names: EndpointNames,
    choice: InstanceChoice,
    scaling: Scaling,
    *,
    role_arn: str,
    kms_key_arn: str | None,
    tags: list[dict[str, str]],
) -> dict[str, Any]:
    instances_per_copy = 1  # one model copy per instance
    req: dict[str, Any] = {
        "EndpointConfigName": names.config,
        "ExecutionRoleArn": role_arn,
        "ProductionVariants": [
            {
                "VariantName": names.variant,
                "InstanceType": choice.instance_type,
                # Start with one instance so deployment fails fast if the model can't load;
                # autoscaling scales in to min_copies once idle.
                "InitialInstanceCount": 1,
                "ManagedInstanceScaling": {
                    "Status": "ENABLED",
                    "MinInstanceCount": scaling.min_copies * instances_per_copy,
                    "MaxInstanceCount": scaling.max_copies * instances_per_copy,
                },
                "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
            }
        ],
        "Tags": tags,
    }
    if kms_key_arn:
        req["KmsKeyId"] = kms_key_arn
    return req


def inference_component_request(
    names: EndpointNames, choice: InstanceChoice, *, tags: list[dict[str, str]]
) -> dict[str, Any]:
    return {
        "InferenceComponentName": names.component,
        "EndpointName": names.endpoint,
        "VariantName": names.variant,
        "Specification": {
            "ModelName": names.model,
            "ComputeResourceRequirements": {
                "NumberOfAcceleratorDevicesRequired": choice.gpus,
                "MinMemoryRequiredInMb": choice.memory_mb,
                **({"NumberOfCpuCoresRequired": choice.cpus} if choice.cpus else {}),
            },
            # LLM weights take minutes to download and load.
            "StartupParameters": {
                "ModelDataDownloadTimeoutInSeconds": 1800,
                "ContainerStartupHealthCheckTimeoutInSeconds": 1800,
            },
        },
        "RuntimeConfig": {"CopyCount": 1},
        "Tags": tags,
    }


@dataclass(frozen=True)
class ScalingPlan:
    register_target: dict[str, Any]
    policies: list[dict[str, Any]]
    # Present when scaling to zero: a step policy plus an alarm that must reference the
    # step policy's ARN (filled in by ``scale_from_zero_alarm`` after it's created).
    scale_from_zero_policy: dict[str, Any] | None


def scaling_plan(names: EndpointNames, scaling: Scaling) -> ScalingPlan:
    resource_id = f"inference-component/{names.component}"
    common = {
        "ServiceNamespace": "sagemaker",
        "ResourceId": resource_id,
        "ScalableDimension": SCALABLE_DIMENSION,
    }
    target_tracking = {
        **common,
        "PolicyName": f"{names.component}-invocations",
        "PolicyType": "TargetTrackingScaling",
        "TargetTrackingScalingPolicyConfiguration": {
            "PredefinedMetricSpecification": {"PredefinedMetricType": INVOCATIONS_PER_COPY},
            "TargetValue": scaling.target_invocations_per_copy,
            "ScaleInCooldown": scaling.scale_in_cooldown_s,
            "ScaleOutCooldown": scaling.scale_out_cooldown_s,
        },
    }
    custom = [{**common, **p} for p in scaling.custom_policies]
    step = None
    if scaling.scales_to_zero:
        step = {
            **common,
            "PolicyName": f"{names.component}-from-zero",
            "PolicyType": "StepScaling",
            "StepScalingPolicyConfiguration": {
                "AdjustmentType": "ChangeInCapacity",
                "MetricAggregationType": "Maximum",
                "Cooldown": 60,
                "StepAdjustments": [{"MetricIntervalLowerBound": 0, "ScalingAdjustment": 1}],
            },
        }
    return ScalingPlan(
        register_target={
            **common,
            "MinCapacity": scaling.min_copies,
            "MaxCapacity": scaling.max_copies,
        },
        policies=[target_tracking, *custom],
        scale_from_zero_policy=step,
    )


def scale_from_zero_alarm(names: EndpointNames, policy_arn: str) -> dict[str, Any]:
    return {
        "AlarmName": f"{names.component}-from-zero",
        "AlarmDescription": "skyhammer: request arrived while scaled to zero; add a copy",
        "AlarmActions": [policy_arn],
        "Namespace": "AWS/SageMaker",
        "MetricName": "NoCapacityInvocationFailures",
        "Dimensions": [{"Name": "InferenceComponentName", "Value": names.component}],
        "Statistic": "Sum",
        "Period": 60,
        "EvaluationPeriods": 1,
        "DatapointsToAlarm": 1,
        # AWS's example uses GreaterThan 1; >= 1 reacts to the very first failed request.
        "Threshold": 1,
        "ComparisonOperator": "GreaterThanOrEqualToThreshold",
    }


# LoRA-serving options from Hub hosting configs; merged weights don't need them.
_LORA_ENV = {
    "OPTION_ENABLE_LORA",
    "OPTION_MAX_LORAS",
    "OPTION_MAX_CPU_LORAS",
    "OPTION_MAX_LORA_RANK",
    "SAGEMAKER_MAX_NUMBER_OF_ADAPTERS_IN_MEMORY",
}


def serving_environment(hosting: HostingConfig, *, merged: bool = True) -> dict[str, str]:
    env = dict(hosting.environment)
    if merged:
        for k in _LORA_ENV:
            env.pop(k, None)
    return env


def model_request(
    names: EndpointNames,
    *,
    image_uri: str,
    environment: dict[str, str],
    model_dir: str,
    role_arn: str,
    tags: list[dict[str, str]],
    hub_content_arn: str | None = None,
    accept_eula: bool = False,
    network_isolation: bool = False,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "S3Uri": model_dir.rstrip("/") + "/",
        "S3DataType": "S3Prefix",
        "CompressionType": "None",
    }
    if hub_content_arn:  # private curated hubs only; public-hub models must omit this
        source["HubAccessConfig"] = {"HubContentArn": hub_content_arn}
    if accept_eula:
        source["ModelAccessConfig"] = {"AcceptEula": True}
    req: dict[str, Any] = {
        "ModelName": names.model,
        "ExecutionRoleArn": role_arn,
        "PrimaryContainer": {
            "Image": image_uri,
            "Environment": environment,
            "ModelDataSource": {"S3DataSource": source},
        },
        "Tags": tags,
    }
    if network_isolation:
        req["EnableNetworkIsolation"] = True
    return req


_IC_STATUS = {
    "InService": "Active",
    "Creating": "Creating",
    "Updating": "Creating",
    "Failed": "Failed",
    "Deleting": "Deleting",
}


def _missing(e: ClientError) -> bool:
    msg = str(e)
    return e.response.get("Error", {}).get("Code") in (
        "ValidationException",
        "ResourceNotFound",
    ) and ("Could not find" in msg or "does not exist" in msg or "not found" in msg.lower())


class EndpointDeployer:
    """Creates, scales and deletes one endpoint + inference component. Every step checks
    for existing resources first, so re-running ``deploy`` resumes a partial deployment."""

    def __init__(self, session: Session, region: str) -> None:
        self.session = session
        self.region = region
        self.sm = session.client("sagemaker", region=region)
        self.autoscaling = session.client("application-autoscaling", region=region)
        self.cloudwatch = session.client("cloudwatch", region=region)

    def _exists(self, describe: Callable[[], Any]) -> bool:
        try:
            describe()
            return True
        except ClientError as e:
            if _missing(e):
                return False
            raise

    def deploy(
        self,
        names: EndpointNames,
        *,
        model_dir: str,
        hosting: HostingConfig,
        hub_content_arn: str | None = None,
        accept_eula: bool = False,
        network_isolation: bool = False,
        merged: bool = True,
        scaling: Scaling,
        instance_type: str | None = None,
        tags: dict[str, str] | None = None,
        wait: bool = True,
        poll_seconds: float = 30,
        log: Callable[[str], None] = lambda _: None,
    ) -> str:
        res = self.session.resources_for(self.region)
        aws_tags = [{"Key": k, "Value": v} for k, v in (tags or {}).items()]
        choice = InstanceChoice.from_hosting(hosting, instance_type)
        if not self._exists(lambda: self.sm.describe_model(ModelName=names.model)):
            log(f"creating model {names.model}")
            self.sm.create_model(
                **model_request(
                    names,
                    image_uri=hosting.image_uri,
                    environment=serving_environment(hosting, merged=merged),
                    model_dir=model_dir,
                    role_arn=res.sagemaker_role_arn,
                    tags=aws_tags,
                    hub_content_arn=hub_content_arn,
                    accept_eula=accept_eula,
                    network_isolation=network_isolation,
                )
            )
        if not self._exists(
            lambda: self.sm.describe_endpoint_config(EndpointConfigName=names.config)
        ):
            self.sm.create_endpoint_config(
                **endpoint_config_request(
                    names,
                    choice,
                    scaling,
                    role_arn=res.sagemaker_role_arn,
                    kms_key_arn=res.kms_key_arn,
                    tags=aws_tags,
                )
            )
        if not self._exists(lambda: self.sm.describe_endpoint(EndpointName=names.endpoint)):
            log(f"creating endpoint {names.endpoint} ({choice.instance_type})")
            self.sm.create_endpoint(
                EndpointName=names.endpoint, EndpointConfigName=names.config, Tags=aws_tags
            )
        if not wait:
            return names.endpoint
        self._wait_in_service(names)
        if not self._exists(
            lambda: self.sm.describe_inference_component(InferenceComponentName=names.component)
        ):
            log(f"creating inference component {names.component}")
            self.sm.create_inference_component(
                **inference_component_request(names, choice, tags=aws_tags)
            )
        while (status := self.status(names)) == "Creating":
            time.sleep(poll_seconds)
        if status != "Active":
            ic = self.sm.describe_inference_component(InferenceComponentName=names.component)
            raise SkyhammerError(
                f"Inference component {names.component} {status}: {ic.get('FailureReason', '')}"
            )
        self.apply_scaling(names, scaling)
        return str(self.sm.describe_endpoint(EndpointName=names.endpoint)["EndpointArn"])

    def _wait_in_service(self, names: EndpointNames) -> None:
        try:
            self.sm.get_waiter("endpoint_in_service").wait(
                EndpointName=names.endpoint, WaiterConfig={"Delay": 30, "MaxAttempts": 120}
            )
        except WaiterError:
            # Surface SageMaker's reason (e.g. InsufficientInstanceCapacity), not the waiter's.
            reason = self.sm.describe_endpoint(EndpointName=names.endpoint).get("FailureReason")
            raise JobFailedError(names.endpoint, reason or "endpoint failed to start") from None

    def apply_scaling(self, names: EndpointNames, scaling: Scaling) -> None:
        plan = scaling_plan(names, scaling)
        self.autoscaling.register_scalable_target(**plan.register_target)
        for policy in plan.policies:
            self.autoscaling.put_scaling_policy(**policy)
        if plan.scale_from_zero_policy:
            arn = self.autoscaling.put_scaling_policy(**plan.scale_from_zero_policy)["PolicyARN"]
            self.cloudwatch.put_metric_alarm(**scale_from_zero_alarm(names, arn))

    def status(self, names: EndpointNames) -> str:
        try:
            ic = self.sm.describe_inference_component(InferenceComponentName=names.component)
        except ClientError as e:
            if _missing(e):
                return "Missing"
            raise
        return _IC_STATUS.get(ic["InferenceComponentStatus"], "Creating")

    def copies(self, names: EndpointNames) -> int:
        ic = self.sm.describe_inference_component(InferenceComponentName=names.component)
        return int(ic.get("RuntimeConfig", {}).get("CurrentCopyCount", 0))

    def warm(self, names: EndpointNames, copies: int = 1) -> None:
        """Scale out ahead of traffic so the first request doesn't wait for an instance."""
        self.sm.update_inference_component_runtime_config(
            InferenceComponentName=names.component, DesiredRuntimeConfig={"CopyCount": copies}
        )

    def delete(self, names: EndpointNames) -> None:
        resource = f"inference-component/{names.component}"
        try:
            self.autoscaling.deregister_scalable_target(
                ServiceNamespace="sagemaker",
                ResourceId=resource,
                ScalableDimension=SCALABLE_DIMENSION,
            )  # also removes its scaling policies
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") != "ObjectNotFoundException":
                raise
        self.cloudwatch.delete_alarms(AlarmNames=[f"{names.component}-from-zero"])

        def describe_ic() -> Any:
            return self.sm.describe_inference_component(InferenceComponentName=names.component)

        if self._exists(describe_ic):
            self.sm.delete_inference_component(InferenceComponentName=names.component)
            while self._exists(describe_ic):  # the endpoint can't go until its IC is gone
                time.sleep(10)
        steps: list[tuple[Callable[[], Any], Callable[[], Any]]] = [
            (
                lambda: self.sm.describe_endpoint(EndpointName=names.endpoint),
                lambda: self.sm.delete_endpoint(EndpointName=names.endpoint),
            ),
            (
                lambda: self.sm.describe_endpoint_config(EndpointConfigName=names.config),
                lambda: self.sm.delete_endpoint_config(EndpointConfigName=names.config),
            ),
            (
                lambda: self.sm.describe_model(ModelName=names.model),
                lambda: self.sm.delete_model(ModelName=names.model),
            ),
        ]
        for describe, delete in steps:
            if self._exists(describe):
                delete()
