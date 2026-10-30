"""Bedrock-native customization: CreateModelCustomizationJob + CreateCustomModelDeployment."""

from __future__ import annotations

import csv
import io
from typing import Any

from skyhammer import tags as tagging
from skyhammer.backends.base import (
    DeploymentRef,
    DeploymentType,
    DeployOptions,
    map_hyperparameters,
)
from skyhammer.catalog import DatasetLimits, ModelEntry, Technique, TrainingType
from skyhammer.errors import UnsupportedParameterError
from skyhammer.formats import message_text, to_bedrock_conversation
from skyhammer.jobs import (
    JobHandle,
    JobRecord,
    JobState,
    JobStatus,
    ModelRef,
    TrainingRequest,
)
from skyhammer.session import Session

_CUSTOMIZATION_TYPE = {
    Technique.SFT: "FINE_TUNING",
    Technique.DISTILLATION: "DISTILLATION",
    Technique.CPT: "CONTINUED_PRE_TRAINING",
    Technique.RFT: "REINFORCEMENT_FINE_TUNING",
}

_STATE = {
    "InProgress": JobState.TRAINING,
    "Completed": JobState.COMPLETED,
    "Failed": JobState.FAILED,
    "Stopping": JobState.TRAINING,
    "Stopped": JobState.STOPPED,
}

# SDK hyperparameter -> Bedrock hyperParameters key. Others are rejected, not dropped.
_HP_KEYS = {
    "epochs": "epochCount",
    "learning_rate": "learningRate",
    "batch_size": "batchSize",
    "warmup_steps": "learningRateWarmupSteps",
}

# RFT hyperparameters live in rftConfig, typed (not the string map other techniques use).
_RFT_HP_KEYS = {"epochs": "epochCount", "learning_rate": "learningRate", "batch_size": "batchSize"}
_RFT_INT_KEYS = {
    "epochCount",
    "batchSize",
    "maxPromptLength",
    "trainingSamplePerPrompt",
    "inferenceMaxTokens",
    "evalInterval",
}
RFT_MIN_PROMPTS, RFT_MAX_PROMPTS = 100, 20_000
_REFERENCE_KEYS = ("answer", "reference", "expected", "reference_answer")

DEFAULT_TEACHER_MAX_RESPONSE_TOKENS = 1000


_STAGES = (  # Bedrock's job phases, in order (statusDetails)
    ("validationDetails", "validating data"),
    ("dataProcessingDetails", "processing data (distillation: teacher answers)"),
    ("trainingDetails", "training"),
)


def stage(details: dict[str, Any]) -> str | None:
    """The phase in progress, e.g. ``processing data (...)``, from ``statusDetails``."""
    for key, label in _STAGES:
        if (details.get(key) or {}).get("status") == "InProgress":
            return label
    return None


class BedrockNativeBackend:
    name = "bedrock"

    def __init__(self, session: Session, region: str | None = None) -> None:
        self.session = session
        self.region = region or session.region
        self.bedrock = session.client("bedrock", region=self.region)

    # ---- data ----

    def dataset_limits(
        self, entry: ModelEntry, technique: Technique, training_type: TrainingType
    ) -> DatasetLimits:
        if technique is Technique.RFT:
            return entry.limits.model_copy(
                update={"min_samples": RFT_MIN_PROMPTS, "max_samples": RFT_MAX_PROMPTS}
            )
        return entry.limits

    def format_record(self, technique: Technique, record: dict[str, Any]) -> dict[str, Any]:
        if technique is Technique.CPT:
            return {"input": record["text"]}
        if technique is Technique.RFT:
            return to_bedrock_rft(record)
        return to_bedrock_conversation(record)

    # ---- training ----

    def build_request(
        self,
        request: TrainingRequest,
        training_uri: str,
        validation_uri: str | None,
        output_uri: str,
        *,
        grader_arn: str | None = None,
    ) -> dict[str, Any]:
        """Pure mapping from TrainingRequest to CreateModelCustomizationJob kwargs.

        RFT needs the reward Lambda's ARN: ``grader_arn`` (``submit`` deploys Python graders
        first) or a Lambda grader's own ARN.
        """
        res = self.session.resources_for(self.region)
        technique = request.technique
        tags = tagging.as_bedrock(request.tags)
        kwargs: dict[str, Any] = {
            "jobName": request.job_name,
            "customModelName": request.output_model,
            "roleArn": res.bedrock_role_arn,
            "baseModelIdentifier": request.route.capability.backend_model_id,
            "customizationType": _CUSTOMIZATION_TYPE[technique],
            "trainingDataConfig": {"s3Uri": training_uri},
            "outputDataConfig": {"s3Uri": output_uri},
            "jobTags": tags,
            "customModelTags": tags,
            "clientRequestToken": request.fingerprint()[:64],
        }
        if technique is Technique.DISTILLATION:
            if not request.teacher_model:
                raise UnsupportedParameterError("distillation requires teacher_model")
            kwargs["customizationConfig"] = {
                "distillationConfig": {
                    "teacherModelConfig": {
                        "teacherModelIdentifier": request.teacher_model,
                        "maxResponseLengthForInference": DEFAULT_TEACHER_MAX_RESPONSE_TOKENS,
                    }
                }
            }
        elif technique is Technique.RFT:
            kwargs["customizationConfig"] = {"rftConfig": _rft_config(request, grader_arn)}
        else:
            kwargs["hyperParameters"] = map_hyperparameters(
                request, _HP_KEYS, "Remove them or pass backend='sagemaker'."
            )
        if validation_uri:
            kwargs["validationDataConfig"] = {"validators": [{"s3Uri": validation_uri}]}
        if res.kms_key_arn:
            kwargs["customModelKmsKeyId"] = res.kms_key_arn
        return kwargs

    def find(self, request: TrainingRequest) -> JobHandle | None:
        resp = self.bedrock.list_model_customization_jobs(nameContains=request.job_name)
        for job in resp.get("modelCustomizationJobSummaries", []):
            if job["jobName"] == request.job_name:
                return JobHandle(self.name, job["jobName"], job["jobArn"], self.region)
        return None

    def submit(self, request: TrainingRequest) -> JobHandle:
        res = self.session.resources_for(self.region)
        prefix = f"s3://{res.bucket}/jobs/{request.job_name}"
        training_uri = request.dataset.upload(self, request.technique)
        validation_uri = (
            request.validation_dataset.upload(self, request.technique)
            if request.validation_dataset
            else None
        )
        grader_arn = None
        if request.technique is Technique.RFT and request.grader:
            # Build (and so validate) the request before creating the Lambda.
            self.build_request(
                request, training_uri, validation_uri, "s3://x/", grader_arn="pending"
            )
            # The reward Lambda runs in the training region (Bedrock calls it from there).
            deployed = request.grader.deploy(self.session, region=self.region, contract="bedrock")
            grader_arn = deployed.lambda_arn
        kwargs = self.build_request(
            request, training_uri, validation_uri, f"{prefix}/output/", grader_arn=grader_arn
        )
        resp = self.bedrock.create_model_customization_job(**kwargs)
        return JobHandle(self.name, request.job_name, resp["jobArn"], self.region)

    def describe(self, handle: JobHandle) -> JobStatus:
        job = self.bedrock.get_model_customization_job(jobIdentifier=handle.job_id)
        return JobStatus(
            state=_STATE.get(job["status"], JobState.PENDING),
            message=job.get("failureMessage") or stage(job.get("statusDetails") or {}),
            output_model_arn=job.get("outputModelArn"),
        )

    def poll(self, record: JobRecord) -> JobStatus:
        return self.describe(record.handle)  # single phase

    def logs(self, record: JobRecord, limit: int = 100) -> list[str]:
        """Bedrock doesn't stream logs; it writes step metrics CSVs to the output prefix."""
        res = self.session.resources_for(self.region)
        s3 = self.session.client("s3", region=self.region)
        prefix = f"jobs/{record.job_name}/output/"
        lines: list[str] = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=res.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("_metrics.csv"):
                    continue
                body = s3.get_object(Bucket=res.bucket, Key=key)["Body"].read().decode()
                lines += format_metrics_csv(key.rsplit("/", 1)[-1], body)
        if not lines:
            st = self.describe(record.handle)
            return [f"no metrics yet (status={st.state.value})"]
        return lines[-limit:]

    def cancel(self, handle: JobHandle) -> None:
        self.bedrock.stop_model_customization_job(jobIdentifier=handle.job_id)

    def finalize(self, record: JobRecord) -> ModelRef:
        st = self.describe(record.handle)
        if st.state is not JobState.COMPLETED or not st.output_model_arn:
            raise RuntimeError(
                f"Job {record.job_name!r} has not completed (state={st.state.value})"
            )
        return ModelRef(
            name=record.output_model,
            model_arn=st.output_model_arn,
            backend=self.name,
            base_alias=record.base_model,
            region=self.region,
        )

    # ---- serving ----

    def deploy(self, model: ModelRef, opts: DeployOptions) -> DeploymentRef:
        if opts.type is DeploymentType.PROVISIONED:
            raise NotImplementedError("Provisioned throughput is planned for M4")
        existing = self._find_deployment(opts.name)
        if existing and existing.status != "Failed":
            return existing  # idempotent: re-running deploy/apply reuses it
        resp = self.bedrock.create_custom_model_deployment(
            modelDeploymentName=opts.name,
            modelArn=model.model_arn,
            tags=tagging.as_bedrock(opts.tags or {}),
        )
        return DeploymentRef(opts.name, resp["customModelDeploymentArn"], "Creating")

    def _find_deployment(self, name: str) -> DeploymentRef | None:
        # GetCustomModelDeployment rejects unknown *names* with ValidationException (seen
        # live), so look names up via the list API and match exactly.
        resp = self.bedrock.list_custom_model_deployments(nameContains=name)
        for d in resp.get("modelDeploymentSummaries", []):
            if d["customModelDeploymentName"] == name:
                return DeploymentRef(name, d["customModelDeploymentArn"], d["status"])
        return None

    def describe_deployment(self, arn: str) -> DeploymentRef:
        d = self.bedrock.get_custom_model_deployment(customModelDeploymentIdentifier=arn)
        return DeploymentRef(d["modelDeploymentName"], d["customModelDeploymentArn"], d["status"])

    def delete_deployment(self, arn: str) -> None:
        self.bedrock.delete_custom_model_deployment(customModelDeploymentIdentifier=arn)

    def delete_model(self, model: ModelRef) -> None:
        self.bedrock.delete_custom_model(modelIdentifier=model.model_arn)


def format_metrics_csv(name: str, body: str) -> list[str]:
    """``step_number,epoch_number,training_loss`` rows -> "step 6 epoch 0 training_loss=0.54"."""
    rows = list(csv.DictReader(io.StringIO(body)))
    label = name.removesuffix(".csv")
    out = []
    for row in rows:
        step = row.pop("step_number", "?")
        epoch = row.pop("epoch_number", "?")
        values = " ".join(
            f"{k}={float(v):.4f}" if _is_float(v) else f"{k}={v}" for k, v in row.items()
        )
        out.append(f"[{label}] step {step} epoch {epoch} {values}")
    return out


def _is_float(v: str) -> bool:
    try:
        float(v)
    except (TypeError, ValueError):
        return False
    return True


def to_bedrock_rft(record: dict[str, Any]) -> dict[str, Any]:
    """Prompt + reference -> Bedrock RFT row: OpenAI-style ``messages`` (ending with the
    prompt), ``reference_answer``, and any other fields, which reach the reward Lambda in
    ``metadata``. A trailing assistant turn is treated as the reference, not the prompt."""
    from skyhammer.graders import reference_answer

    messages = record["messages"]
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    reference = record.get("reference_answer") or reference_answer(record)
    extra = {k: v for k, v in record.items() if k not in ("messages", *_REFERENCE_KEYS)}
    return {
        **extra,
        "messages": [{"role": m["role"], "content": message_text(m["content"])} for m in messages],
        "reference_answer": reference,
    }


def _rft_config(request: TrainingRequest, grader_arn: str | None) -> dict[str, Any]:
    from skyhammer.graders import GraderKind

    g = request.grader
    if g is None:
        raise UnsupportedParameterError("rft requires a grader (a Python function or Lambda)")
    if g.kind is GraderKind.LLM_JUDGE:
        raise UnsupportedParameterError(
            "Bedrock RFT takes Lambda rewards; write the judge call in a @grader function"
        )
    arn = grader_arn if grader_arn is not None else g.lambda_arn
    if arn is None:
        raise UnsupportedParameterError("deploy the grader first (submit does this)")
    hp: dict[str, Any] = {
        k: (int(v) if k in _RFT_INT_KEYS else v)
        for k, v in map_hyperparameters(
            request, _RFT_HP_KEYS, "Pass RFT-specific keys with hyperparameters_extra."
        ).items()
    }
    for key in ("learningRate",):
        if key in hp:
            hp[key] = float(hp[key])
    config: dict[str, Any] = {"graderConfig": {"lambdaGrader": {"lambdaArn": arn}}}
    if hp:
        config["hyperParameters"] = hp
    return config
