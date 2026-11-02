"""Continued pre-training (CPT) of open-weight models as SageMaker training jobs.

SageMaker's serverless customization recipes cover SFT, DPO and RL only, and Bedrock no
longer offers CONTINUED_PRE_TRAINING for any model (Titan Text was the last). Nova CPT
exists only as SageMaker HyperPod recipes on 8 to 24 ml.p5.48xlarge instances. So
Skyhammer runs CPT itself: a SageMaker training job on one GPU instance, using the AWS
Hugging Face training container and Skyhammer's own script (``trainers/cpt_train.py``).
Base weights are the open JumpStart artifacts that SageMaker already hosts, the output is
merged Hugging Face weights, and the job then imports into Bedrock like any SageMaker fine-tune.

Training jobs bill by instance-hour, so every CPT job gets a runtime cap (``max_hours``).
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from importlib.resources import files
from typing import TYPE_CHECKING, Any

from skyhammer import tags as tagging
from skyhammer.catalog import DatasetLimits, TrainingType
from skyhammer.errors import SkyhammerError, UnsupportedParameterError

if TYPE_CHECKING:
    from skyhammer.hub import HubModel
    from skyhammer.jobs import TrainingRequest
    from skyhammer.session import Session

SCRIPT = "cpt_train.py"
# Hugging Face PyTorch training DLC (transformers 4.56, PyTorch 2.8); in every region the
# DLC account serves.
IMAGE = (
    "763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:"
    "2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"
)
# One document per row; the script packs them into max_seq_length blocks, so long documents
# are fine. The bounds only stop accidental uploads.
CPT_LIMITS = DatasetLimits(
    min_samples=1,
    max_samples=1_000_000,
    max_tokens_per_sample=10_000_000,
    source="Skyhammer CPT (documents are packed into fixed-length blocks)",
)
# (instance type, total GPU memory in GB), smallest first. Multi-GPU instances shard the
# model across GPUs (one process), so their memory adds up.
INSTANCES = (
    ("ml.g5.2xlarge", 24),
    ("ml.g6e.2xlarge", 48),
    ("ml.g6e.12xlarge", 192),
    ("ml.p4de.24xlarge", 640),
)
DEFAULTS: dict[str, Any] = {
    "epochs": 1,
    "batch_size": 8,
    "warmup_ratio": 0.03,
    "lora_rank": 16,
    "lora_alpha": 32,
    "max_seq_length": 2048,
}
LEARNING_RATE = {TrainingType.LORA: 1e-4, TrainingType.FULL: 1e-5}
# hyperparameters_extra keys the script understands (plus instance_type, used here).
EXTRA = {"max_seq_length", "per_device_batch_size", "gradient_checkpointing", "instance_type"}


def memory_gb(params_b: float, training_type: TrainingType) -> float:
    """Rough GPU memory to train: bf16 weights plus activations for LoRA; weights, gradients
    and AdamW state (16 bytes/param) for full training."""
    per_param = 2.6 if training_type is TrainingType.LORA else 16.0
    return params_b * per_param + 4


def instance_for(params_b: float | None, training_type: TrainingType) -> str:
    if params_b is None:
        raise SkyhammerError("CPT needs the model size; pass hyperparameters_extra.instance_type")
    need = memory_gb(params_b, training_type)
    for name, gb in INSTANCES:
        if gb >= need:
            return name
    raise UnsupportedParameterError(
        f"{training_type.value} CPT of a {params_b:g}B model needs ~{need:.0f} GB of GPU "
        "memory; use LoRA or pass hyperparameters_extra.instance_type"
    )


def instance_type(request: TrainingRequest) -> str:
    chosen = request.hyperparameters.extra.get("instance_type")
    if chosen:
        return str(chosen)
    return instance_for(request.route.model.params_b, request.training_type)


def hyperparameters(request: TrainingRequest) -> dict[str, str]:
    """Script arguments, JSON-encoded the way the SageMaker training toolkit reads them."""
    hp = request.hyperparameters
    if hp.warmup_steps is not None:
        raise UnsupportedParameterError("CPT doesn't support warmup_steps (use warmup_ratio)")
    if hp.beta is not None:
        raise UnsupportedParameterError("beta is a DPO setting")
    unknown = sorted(set(hp.extra) - EXTRA)
    if unknown:
        raise UnsupportedParameterError(
            f"CPT doesn't support hyperparameters_extra {unknown}; choose from {sorted(EXTRA)}"
        )
    values: dict[str, Any] = {
        **DEFAULTS,
        "learning_rate": LEARNING_RATE[request.training_type],
        **hp.set_fields(),
        **{k: v for k, v in hp.extra.items() if k != "instance_type"},
        "training_type": request.training_type.value,
    }
    for key in ("epochs", "batch_size", "lora_rank", "lora_alpha", "max_seq_length"):
        if not isinstance(values[key], int) or values[key] < 1:
            raise UnsupportedParameterError(f"{key} must be a positive integer")
    if not 0 < float(values["learning_rate"]) < 1:
        raise UnsupportedParameterError("learning_rate must be between 0 and 1")
    if not 0 <= float(values["warmup_ratio"]) < 1:
        raise UnsupportedParameterError("warmup_ratio must be in [0, 1)")
    return {k: json.dumps(v) for k, v in sorted(values.items())}


def check(request: TrainingRequest, hub_model: HubModel) -> dict[str, str]:
    """Everything that can fail before spending: base, weights, size, hyperparameters."""
    if request.source_model:
        raise SkyhammerError("CPT starts from a base model, not from an earlier fine-tune")
    if hub_model.gated:
        raise SkyhammerError(
            f"{hub_model.name} has gated weights ({hub_model.license}); Skyhammer CPT "
            "supports open-weight models"
        )
    if hub_model.base is None:
        raise SkyhammerError(f"{hub_model.name} has no Hugging Face weights to train from")
    instance_type(request)
    return hyperparameters(request)


def script_bundle() -> bytes:
    """sourcedir.tar.gz with the training script; deterministic, so its hash names it."""
    source = files("skyhammer").joinpath("trainers", SCRIPT).read_bytes()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
        info = tarfile.TarInfo(SCRIPT)
        info.size, info.mtime, info.mode = len(source), 0, 0o644
        tar.addfile(info, io.BytesIO(source))
    data = buf.getvalue()
    # gzip stores a timestamp at bytes 4-8; zero it so the bundle is reproducible.
    return data[:4] + b"\0\0\0\0" + data[8:]


def upload_script(session: Session, region: str) -> str:
    res = session.resources_for(region)
    data = script_bundle()
    key = f"code/cpt/{hashlib.sha256(data).hexdigest()[:16]}/sourcedir.tar.gz"
    extra = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": res.kms_key_arn}
    session.client("s3", region=region).put_object(
        Bucket=res.bucket, Key=key, Body=data, **(extra if res.kms_key_arn else {})
    )
    return f"s3://{res.bucket}/{key}"


def build_request(
    session: Session,
    region: str,
    request: TrainingRequest,
    hub_model: HubModel,
    *,
    training_uri: str,
    validation_uri: str | None,
    output_uri: str,
    script_uri: str,
) -> dict[str, Any]:
    """CreateTrainingJob kwargs for a CPT job (no AWS calls)."""
    from skyhammer.backends.sagemaker import _channel

    hp = check(request, hub_model)
    assert hub_model.base is not None
    res = session.resources_for(region)
    model_channel = _channel("model", hub_model.base.artifact_uri)
    channels = [model_channel, _channel("train", training_uri)]
    if validation_uri:
        channels.append(_channel("validation", validation_uri))
    output: dict[str, Any] = {"S3OutputPath": output_uri, "CompressionType": "NONE"}
    if res.kms_key_arn:
        output["KmsKeyId"] = res.kms_key_arn
    vpc = session.settings.vpc_for(region)
    max_hours = request.max_hours or 4.0
    return {
        "TrainingJobName": request.job_name,
        "RoleArn": res.sagemaker_role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": IMAGE.format(region=region),
            "TrainingInputMode": "File",
        },
        "HyperParameters": {
            **hp,
            "sagemaker_program": json.dumps(SCRIPT),
            "sagemaker_submit_directory": json.dumps(script_uri),
        },
        "InputDataConfig": channels,
        "OutputDataConfig": output,
        "ResourceConfig": {
            "InstanceType": instance_type(request),
            "InstanceCount": 1,
            "VolumeSizeInGB": max(50, int((request.route.model.params_b or 8) * 8)),
            **({"VolumeKmsKeyId": res.kms_key_arn} if res.kms_key_arn else {}),
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": int(max_hours * 3600)},
        **(
            {"VpcConfig": {"SecurityGroupIds": vpc.security_group_ids, "Subnets": vpc.subnet_ids}}
            if vpc
            else {}
        ),
        "Tags": tagging.as_aws(request.tags),
    }
