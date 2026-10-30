"""Bring weights into Bedrock: your own Hugging Face weights, or a SageMaker-served model.

    LLM.import_weights("s3://bucket/my-llama/", name="my-llama")   # BYO weights
    ft.deploy(serving="bedrock")                                    # SageMaker -> Bedrock

Custom Model Import constraints are checked before the (slow) import job starts.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError

from skyhammer import tags as tagging
from skyhammer.errors import JobFailedError, SkyhammerError

if TYPE_CHECKING:
    from skyhammer.session import Session

# Custom Model Import architectures (Bedrock user guide, 2026-10-29).
SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM",
    "MllamaForConditionalGeneration",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "T5ForConditionalGeneration",
    "GPTBigCodeForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "GptOssForCausalLM",
}
MAX_CONTEXT = 128 * 1024
MAX_TEXT_BYTES = 200 * 2**30
MAX_MULTIMODAL_BYTES = 100 * 2**30


def preflight(s3: Any, model_dir: str) -> dict[str, Any]:
    """Validate an HF model directory in S3 against Custom Model Import limits."""
    bucket, prefix = model_dir.removeprefix("s3://").split("/", 1)
    prefix = prefix.rstrip("/") + "/"
    keys: dict[str, int] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys[obj["Key"][len(prefix) :]] = obj["Size"]
    if "config.json" not in keys:
        raise SkyhammerError(f"{model_dir} has no config.json (expected Hugging Face format)")
    if not any(k.endswith(".safetensors") for k in keys):
        raise SkyhammerError(f"{model_dir} has no .safetensors weights")
    if not any(k.startswith("tokenizer") for k in keys):
        raise SkyhammerError(f"{model_dir} has no tokenizer files")
    config = json.loads(s3.get_object(Bucket=bucket, Key=prefix + "config.json")["Body"].read())
    archs = set(config.get("architectures") or [])
    if not archs & SUPPORTED_ARCHITECTURES:
        raise SkyhammerError(
            f"Architecture {sorted(archs) or 'unknown'} isn't supported by Bedrock Custom Model "
            "Import; serve it from a SageMaker endpoint instead (serving='sagemaker')."
        )
    ctx = config.get("max_position_embeddings") or 0
    if ctx > MAX_CONTEXT:
        raise SkyhammerError(f"max_position_embeddings={ctx:,} exceeds the 128K import limit")
    total = sum(keys.values())
    multimodal = any("VL" in a or "Mllama" in a for a in archs)
    limit = MAX_MULTIMODAL_BYTES if multimodal else MAX_TEXT_BYTES
    if total > limit:
        raise SkyhammerError(
            f"Weights total {total / 2**30:.0f} GB; import allows "
            f"{limit // 2**30} GB for this model type"
        )
    return {"architectures": sorted(archs), "bytes": total, "context": ctx}


def import_to_bedrock(
    session: Session,
    region: str,
    model_dir: str,
    name: str,
    *,
    tags: dict[str, str] | None = None,
    wait: bool = True,
    poll_seconds: float = 30,
    log: Callable[[str], None] = lambda _: None,
) -> str:
    """Import (idempotently) and return the imported model ARN."""
    bedrock = session.client("bedrock", region=region)
    try:
        return str(bedrock.get_imported_model(modelIdentifier=name)["modelArn"])
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") not in (
            "ResourceNotFoundException",
            "ValidationException",
        ):
            raise
    info = preflight(session.client("s3", region=region), model_dir)
    log(f"importing {info['architectures']} ({info['bytes'] / 2**30:.1f} GB) as {name}")
    job_name = f"{name[:55]}-import"
    existing = [
        j
        for j in bedrock.list_model_import_jobs(nameContains=job_name).get(
            "modelImportJobSummaries", []
        )
        if j["jobName"] == job_name
    ]
    if existing:
        job_arn = existing[0]["jobArn"]
    else:
        res = session.resources_for(region)
        bedrock_tags = tagging.as_bedrock(tags or {})
        kwargs: dict[str, Any] = {
            "jobName": job_name,
            "importedModelName": name,
            "roleArn": res.import_role_arn,
            "modelDataSource": {"s3DataSource": {"s3Uri": model_dir.rstrip("/") + "/"}},
            "jobTags": bedrock_tags,
            "importedModelTags": bedrock_tags,
        }
        if res.kms_key_arn:
            kwargs["importedModelKmsKeyId"] = res.kms_key_arn
        vpc = session.settings.vpc_for(region)
        if vpc:
            kwargs["vpcConfig"] = {
                "subnetIds": vpc.subnet_ids,
                "securityGroupIds": vpc.security_group_ids,
            }
        while True:
            try:
                job_arn = bedrock.create_model_import_job(**kwargs)["jobArn"]
                break
            except ClientError as e:
                if e.response["Error"]["Code"] != "ServiceQuotaExceededException":
                    raise
                log("waiting for a free Bedrock import slot (concurrent import quota)")
                time.sleep(poll_seconds)
    if not wait:
        return str(job_arn)
    while True:
        job = bedrock.get_model_import_job(jobIdentifier=job_arn)
        if job["status"] == "Completed":
            return str(job["importedModelArn"])
        if job["status"] == "Failed":
            raise JobFailedError(job_name, job.get("failureMessage", "import failed"))
        time.sleep(poll_seconds)
