"""Managed AWQ quantization (SV-14): SageMaker AI inference optimization jobs.

``deploy(quantize="fp8")`` quantizes at load time. An optimization job instead writes
quantized weights once: SageMaker runs AWQ (activation-aware 4-bit weight quantization)
on an instance it chooses, and the result is a new model whose weights take about a
quarter of their bf16 memory, so it fits a smaller (cheaper) instance. vLLM detects AWQ
weights on its own, so the output deploys like any SageMaker fine-tune::

    job = LLM.get("support-qwen").create_optimization_job("support-qwen-awq")
    job.wait()
    LLM.get("support-qwen-awq").deploy(instance_type="ml.g6.2xlarge")

The source is a SageMaker-trained fine-tune (its merged weights in S3) or a base model
(AWS's prepacked weights). Billing is for the instance SageMaker picks, for as long as the
job runs; ``max_hours`` caps it (StoppingCondition). EAGLE speculative-decoding training
isn't wrapped yet.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError

from skyhammer import registry
from skyhammer import tags as tagging
from skyhammer.errors import SkyhammerError, UnsupportedParameterError
from skyhammer.jobs import ModelRef

if TYPE_CHECKING:
    from skyhammer.llm import LLM
    from skyhammer.session import Session

QUANTIZE = ("awq",)
_STATE = {
    "STARTING": "pending",
    "INPROGRESS": "running",
    "STOPPING": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "STOPPED": "stopped",
}


def job_name(output_model: str, model_uri: str, quantize: str, instance_type: str) -> str:
    key = json.dumps([model_uri, quantize, instance_type])
    return f"{output_model[:44]}-{quantize}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def build_job(
    name: str,
    *,
    role_arn: str,
    model_uri: str,
    accept_eula: bool,
    instance_type: str,
    quantize: str,
    output_uri: str,
    max_hours: float,
    kms_key_arn: str | None,
    vpc: dict[str, list[str]] | None,
    tags: list[dict[str, str]],
) -> dict[str, Any]:
    """CreateOptimizationJob kwargs (no AWS calls)."""
    if quantize not in QUANTIZE:
        raise UnsupportedParameterError(
            f"quantize={quantize!r}: optimization jobs here run {', '.join(QUANTIZE)}; for fp8 "
            "use deploy(quantize='fp8'), which quantizes as the model loads"
        )
    if not 0 < max_hours <= 24:
        raise UnsupportedParameterError("max_hours must be between 0 and 24")
    output: dict[str, Any] = {"S3OutputLocation": output_uri}
    if kms_key_arn:
        output["KmsKeyId"] = kms_key_arn
    kwargs: dict[str, Any] = {
        "OptimizationJobName": name,
        "RoleArn": role_arn,
        "ModelSource": {
            "S3": {"S3Uri": model_uri, "ModelAccessConfig": {"AcceptEula": accept_eula}}
        },
        "DeploymentInstanceType": instance_type,
        "OptimizationConfigs": [
            {"ModelQuantizationConfig": {"OverrideEnvironment": {"OPTION_QUANTIZE": quantize}}}
        ],
        "OutputConfig": output,
        "StoppingCondition": {"MaxRuntimeInSeconds": int(max_hours * 3600)},
        "Tags": tags,
    }
    if vpc:
        kwargs["VpcConfig"] = vpc
    return kwargs


@dataclass(frozen=True)
class OptimizationStatus:
    state: str  # pending, running, completed, failed, stopped
    message: str | None
    image: str | None  # the inference image AWS recommends for the output


def find_model_dir(s3: Any, uri: str) -> str:
    """The directory under ``uri`` holding the optimized Hugging Face model (config.json)."""
    bucket, prefix = uri.removeprefix("s3://").split("/", 1)
    keys = [
        obj["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
        if obj["Key"].endswith("/config.json") or obj["Key"] == prefix + "config.json"
    ]
    if not keys:
        raise SkyhammerError(f"No optimized model (config.json) found under {uri}")
    shallowest = min(keys, key=lambda k: (k.count("/"), k))
    return f"s3://{bucket}/{shallowest.removesuffix('config.json')}"


class OptimizationJob:
    """A SageMaker inference optimization job. Create with
    ``LLM.create_optimization_job``; when it completes, ``LLM.get(output_model)`` is the
    optimized model."""

    def __init__(
        self,
        session: Session,
        name: str,
        *,
        region: str,
        output_model: str,
        base_alias: str,
        output_uri: str,
    ) -> None:
        self.session = session
        self.name = name
        self.region = region
        self.output_model = output_model
        self.base_alias = base_alias
        self.output_uri = output_uri

    @property
    def _sm(self) -> Any:
        return self.session.client("sagemaker", region=self.region)

    def status(self) -> OptimizationStatus:
        job = self._sm.describe_optimization_job(OptimizationJobName=self.name)
        return OptimizationStatus(
            _STATE.get(job["OptimizationJobStatus"], "pending"),
            job.get("FailureReason"),
            (job.get("OptimizationOutput") or {}).get("RecommendedInferenceImage"),
        )

    def wait(
        self,
        *,
        poll_seconds: float = 60,
        timeout_seconds: float | None = None,
        on_status: Callable[[OptimizationStatus], None] | None = None,
    ) -> OptimizationJob:
        """Wait for the job, then register the optimized model (``LLM.get(output_model)``)."""
        deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
        while True:
            st = self.status()
            if on_status:
                on_status(st)
            if st.state == "completed":
                self.finalize()
                return self
            if st.state in ("failed", "stopped"):
                raise SkyhammerError(f"Optimization {self.name} {st.state}: {st.message}")
            if deadline and time.monotonic() > deadline:
                raise TimeoutError(f"Optimization {self.name} still {st.state}")
            time.sleep(poll_seconds)

    def finalize(self) -> ModelRef:
        """Register the optimized weights as ``output_model`` (served from SageMaker)."""
        st = self.status()
        if st.state != "completed":
            raise SkyhammerError(f"Optimization {self.name} is {st.state}, not completed")
        s3 = self.session.client("s3", region=self.region)
        ref = ModelRef(
            name=self.output_model,
            model_arn=self._sm.describe_optimization_job(OptimizationJobName=self.name)[
                "OptimizationJobArn"
            ],
            backend="sagemaker",
            base_alias=self.base_alias,
            region=self.region,
            api="openai_chat",
            serving="sagemaker",
            artifacts_uri=find_model_dir(s3, self.output_uri),
        )
        registry.save_model_ref(self.session, ref, optimization_job=self.name)
        return ref

    @property
    def output_llm(self) -> LLM:
        from skyhammer.llm import LLM

        return LLM.from_model_ref(self.finalize(), session=self.session)

    def stop(self) -> None:
        self._sm.stop_optimization_job(OptimizationJobName=self.name)

    def __repr__(self) -> str:
        return f"OptimizationJob(name={self.name!r}, output_model={self.output_model!r})"


def submit(
    llm: LLM,
    output_model: str,
    *,
    quantize: str,
    instance_type: str | None,
    max_hours: float,
    accept_eula: bool,
) -> OptimizationJob:
    """Resolve the source weights, check everything, then create (or reattach to) the job."""
    from skyhammer.backends.sagemaker import SageMakerServerlessBackend, resolve_model_dir
    from skyhammer.catalog import BackendKind

    session = llm.session
    entry = llm._tunable()
    ref = llm.model_ref
    if ref is not None:
        if ref.backend != BackendKind.SAGEMAKER.value or not ref.artifacts_uri:
            raise SkyhammerError(
                f"{ref.name} has no SageMaker-trained weights to optimize (Bedrock-native "
                "fine-tunes keep their weights in Bedrock)"
            )
        region = ref.region
        backend = SageMakerServerlessBackend(session, region)
        model_uri = resolve_model_dir(session.client("s3", region=region), ref.artifacts_uri)
        default_instance = backend.hosting_for(ref.base_alias).instance_type
        gated = entry.requires_eula
    else:
        caps = [c for c in entry.capabilities if c.backend is BackendKind.SAGEMAKER]
        if not caps:
            raise SkyhammerError(f"{entry.display_name} has no SageMaker weights to optimize")
        region = session.settings.region
        backend = SageMakerServerlessBackend(session, region)
        hub_model = backend.hub.model(caps[0].backend_model_id)
        if hub_model.base is None:
            raise SkyhammerError(f"No prepacked weights published for {hub_model.name}")
        model_uri = hub_model.base.artifact_uri
        default_instance = hub_model.base.hosting.instance_type
        gated = hub_model.gated
    if gated and not accept_eula:
        raise SkyhammerError(
            f"{entry.display_name} has a gated license. Review it, then pass accept_eula=True"
        )
    instance = instance_type or default_instance
    sm = session.client("sagemaker", region=region)
    allowed = (
        sm.meta.service_model.operation_model("CreateOptimizationJob")
        .input_shape.members["DeploymentInstanceType"]
        .enum
    )
    if instance not in allowed:
        raise UnsupportedParameterError(
            f"Optimization jobs can't target {instance}; choose one of: {', '.join(allowed)}"
        )
    res = session.resources_for(region)
    name = job_name(output_model, model_uri, quantize, instance)
    output_uri = f"s3://{res.bucket}/optimized/{name}/"
    vpc = session.settings.vpc_for(region)
    kwargs = build_job(
        name,
        role_arn=res.sagemaker_role_arn,
        model_uri=model_uri,
        accept_eula=accept_eula or not gated,
        instance_type=instance,
        quantize=quantize,
        output_uri=output_uri,
        max_hours=max_hours,
        kms_key_arn=res.kms_key_arn,
        vpc={"SecurityGroupIds": vpc.security_group_ids, "Subnets": vpc.subnet_ids}
        if vpc
        else None,
        tags=tagging.as_aws(
            tagging.build_tags(
                {"name": output_model, "base-model": entry.alias, "kind": "optimization"},
                session.settings.tags,
            )
        ),
    )
    job = OptimizationJob(
        session,
        name,
        region=region,
        output_model=output_model,
        base_alias=entry.alias,
        output_uri=output_uri,
    )
    try:
        sm.describe_optimization_job(OptimizationJobName=name)
        return job  # same source, method and instance: reattach
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") not in (
            "ResourceNotFound",
            "ValidationException",
        ):
            raise
    sm.create_optimization_job(**kwargs)
    return job
