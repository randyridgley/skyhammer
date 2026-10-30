"""SageMaker AI serverless customization, called directly through boto3.

Training is ``CreateTrainingJob`` with a ``ServerlessJobConfig`` (the same API the
SageMaker Python SDK's ``SFTTrainer``/``DPOTrainer``/``RLVRTrainer`` call), so we don't
depend on the ~530 MB ``sagemaker`` package.

The user sees one ``FineTuningJob``; internally it is a state machine whose phases are
separate AWS jobs, recorded in the job manifest so a failed phase can be retried without
retraining:

    TRAINING   create_training_job(ServerlessJobConfig=...)
    IMPORTING  bedrock.create_model_import_job -> imported model ARN (Bedrock serving only).
               Recipes merge LoRA adapters by default (merge_weights=true), and we write
               uncompressed output (CompressionType=NONE), so weights import as-is.
    COMPLETED  ModelRef; Bedrock: InvokeModel (OpenAI chat body), SageMaker: endpoint

Recipes, hyperparameter specs and gating come from the SageMaker Public Hub (``hub.py``),
so hyperparameters are validated locally against AWS's own bounds before any spend.

Trained weights stay in S3, so a model served from a SageMaker endpoint today can be
imported into Bedrock later (``llm.deploy(serving="bedrock")``) without retraining once
Bedrock supports its architecture.

Import constraints to enforce in ``finalize``: HF safetensors + config/tokenizer files,
<=200 GB (text) / <=100 GB (multimodal), context <128K, transformers 4.51.3.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from botocore.exceptions import ClientError

from skyhammer import tags as tagging
from skyhammer.backends.base import DeploymentRef, DeployOptions
from skyhammer.catalog import (
    SAGEMAKER_HUB_MODEL_ARN,
    DatasetLimits,
    ModelEntry,
    ServingTarget,
    Technique,
    TrainingType,
)
from skyhammer.errors import SkyhammerError, UnsupportedParameterError
from skyhammer.formats import to_sagemaker_dpo, to_sagemaker_rlvr, to_sagemaker_sft
from skyhammer.hub import HostingConfig, HubClient, ParamSpec, Recipe, hub_technique
from skyhammer.jobs import (
    Checkpoint,
    JobHandle,
    JobRecord,
    JobState,
    JobStatus,
    ModelRef,
    TrainingRequest,
)
from skyhammer.serving import EndpointDeployer, EndpointNames, Scaling
from skyhammer.session import Session

log = logging.getLogger(__name__)

_STATE = {
    "InProgress": JobState.TRAINING,
    "Completed": JobState.COMPLETED,
    "Failed": JobState.FAILED,
    "Stopping": JobState.TRAINING,
    "Stopped": JobState.STOPPED,
    "Deleting": JobState.STOPPED,
}

# SDK hyperparameter -> recipe override-param key (from the Hub override params).
_SDK_TO_RECIPE = {
    "epochs": "max_epochs",
    "learning_rate": "learning_rate",
    "batch_size": "global_batch_size",
    "lora_rank": "lora_rank",
    "lora_alpha": "lora_alpha",
    "warmup_ratio": "lr_warmup_steps_ratio",
    "beta": "adam_beta",  # DPO inverse temperature (the recipe's name for it)
}

_LOG_URL = (
    "https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
    "#logStream:group=/aws/sagemaker/TrainingJobs;prefix={name}"
)


EVALUATOR_HUB = "skyhammer"
HUB_DESCRIPTION = "Skyhammer reward evaluators for RLVR"


def is_import_quota_error(e: ClientError) -> bool:
    """Bedrock's "Concurrent model import jobs" quota (seen live with 2 jobs at once)."""
    return e.response["Error"]["Code"] == "ServiceQuotaExceededException"


def _not_found(e: ClientError) -> bool:
    return e.response["Error"]["Code"] in ("ResourceNotFound", "ResourceNotFoundException")


def evaluator_hub_content(name: str, lambda_arn: str) -> dict[str, Any]:
    """ImportHubContent kwargs for a reward-function evaluator, matching the SageMaker
    Python SDK's ``Evaluator.create`` (sagemaker-train 3.x ai_registry)."""
    json_content = {"Reference": lambda_arn, "EvaluatorType": "RewardFunction"}
    return {
        "HubName": EVALUATOR_HUB,
        "HubContentType": "JsonDoc",
        "HubContentName": name,
        "HubContentVersion": "1.0.0",
        "DocumentSchemaVersion": "2.0.0",
        "HubContentDocument": json.dumps(
            {"SubType": "AWS/Evaluator", "JsonContent": json.dumps(json_content)}
        ),
        "HubContentSearchKeywords": [
            "method:lambda",
            "@subtype:aws/evaluator",
            "@evaluatortype:rewardfunction",
            "@contenttype:byolambda",
        ],
    }


def _technique(
    request: TrainingRequest, evaluator_arn: str | None = None
) -> tuple[str, str | None]:
    """Return (Hub CustomizationTechnique, EvaluatorArn). ``evaluator_arn`` is the Hub
    evaluator (JsonDoc) wrapping the reward Lambda, registered by ``submit``."""
    t = request.technique
    if t in (Technique.SFT, Technique.DPO):
        return hub_technique(t), None
    if t is Technique.RFT:
        g = request.grader
        if g is None:
            raise UnsupportedParameterError("rft requires a grader")
        # LLM judges (RLAIF) deploy as a reward Lambda that calls the judge model, so every
        # grader kind trains through the same Lambda evaluator.
        if not evaluator_arn:
            raise UnsupportedParameterError("register the reward Lambda first (submit does)")
        return "RLVR", evaluator_arn
    if t is Technique.DISTILLATION:  # LLM.create_distillation_job runs teacher answers + SFT
        raise UnsupportedParameterError("use LLM.create_distillation_job for open-weight students")
    raise NotImplementedError("Continued pre-training on SageMaker is planned for M4")


def recipe_hyperparameters(
    request: TrainingRequest, recipe: Recipe, params: dict[str, ParamSpec]
) -> dict[str, str]:
    """Recipe defaults overlaid with the user's values, each validated against the spec."""
    hp = {
        name: spec.check(spec.default) for name, spec in params.items() if spec.default is not None
    }
    set_fields = request.hyperparameters.set_fields()
    unknown = sorted(k for k in set_fields if k not in _SDK_TO_RECIPE)
    if unknown:
        hint = " (use warmup_ratio)" if "warmup_steps" in unknown else ""
        raise UnsupportedParameterError(
            f"SageMaker recipes don't support: {', '.join(unknown)}{hint}. "
            "Pass recipe keys with hyperparameters_extra={...}."
        )
    overrides = {_SDK_TO_RECIPE[k]: v for k, v in set_fields.items()}
    overrides.update(request.hyperparameters.extra)
    for key, value in overrides.items():
        if key not in params:
            raise UnsupportedParameterError(
                f"{key!r} isn't tunable in recipe {recipe.name}; tunable: "
                + ", ".join(sorted(params))
            )
        hp[key] = params[key].check(value)
    if request.route.serving is ServingTarget.BEDROCK and hp.get("merge_weights") == "false":
        raise UnsupportedParameterError(
            "merge_weights=false keeps LoRA adapters separate, but Bedrock import needs "
            "merged weights. Remove it or deploy with serving='sagemaker'."
        )
    _check_batches(request, hp, ratio_set_by_user="train_val_split_ratio" in overrides)
    return hp


def _check_batches(
    request: TrainingRequest, hp: dict[str, str], *, ratio_set_by_user: bool
) -> None:
    """Recipes fail at runtime (after minutes of setup) if the train or validation split
    holds less than one batch. Check locally, and fix the split ratio when it's ours to set."""
    if "global_batch_size" not in hp:
        return
    batch = int(hp["global_batch_size"])
    n = request.dataset.num_records
    if request.validation_dataset is not None:
        n_val, n_train = request.validation_dataset.num_records, n
    else:
        ratio = float(hp.get("train_val_split_ratio", "1.0"))
        n_val = n - int(n * ratio)
        n_train = n - n_val
        if n_val < batch and not ratio_set_by_user and n - batch >= batch:
            new_ratio = round(1 - batch / n, 4)
            log.warning(
                "Validation split of %d rows is smaller than batch size %d; using "
                "train_val_split_ratio=%s (%d validation rows)",
                n_val,
                batch,
                new_ratio,
                batch,
            )
            hp["train_val_split_ratio"] = str(new_ratio)
            n_val, n_train = batch, n - batch
    if n_train < batch or n_val < batch:
        raise UnsupportedParameterError(
            f"Batch size {batch} needs at least {batch} training and {batch} validation rows, "
            f"but the data gives {n_train} training / {n_val} validation. Add rows (>= "
            f"{2 * batch} total), pass a larger validation_dataset, or lower batch_size."
        )


def import_job_name(job_name: str) -> str:
    return f"{job_name[:59]}-imp"  # Bedrock job names are <= 63 chars


class SageMakerServerlessBackend:
    name = "sagemaker"

    def __init__(self, session: Session, region: str | None = None) -> None:
        self.session = session
        self.region = region or session.region
        self.sagemaker = session.client("sagemaker", region=self.region)
        self.hub = HubClient(session, self.region)

    @property
    def _bedrock(self) -> Any:
        return self.session.client("bedrock", region=self.region)

    # ---- data ----

    def dataset_limits(
        self, entry: ModelEntry, technique: Technique, training_type: TrainingType
    ) -> DatasetLimits:
        """Max tokens per sample = the longest recipe AWS offers for this model/technique."""
        hub_tech = {Technique.SFT: "SFT", Technique.DPO: "DPO", Technique.RFT: "RLVR"}.get(
            technique
        )
        if hub_tech is None:
            return entry.limits
        caps = [c for c in entry.capabilities_for(technique) if c.backend.value == self.name]
        if not caps:
            return entry.limits
        try:
            model = self.hub.model(caps[0].backend_model_id)
        except Exception as e:  # offline/no credentials: plan still works on catalog limits
            log.debug("hub lookup for %s failed: %s", entry.alias, e)
            return entry.limits
        lengths = [
            r.max_tokens
            for r in model.recipes
            if r.technique == hub_tech and r.training_type is training_type
        ]
        if not lengths:
            return entry.limits
        return entry.limits.model_copy(
            update={
                "max_tokens_per_sample": max(lengths),
                "source": f"SageMaker {hub_tech} recipes for {model.name} v{model.version}",
            }
        )

    def format_record(self, technique: Technique, record: dict[str, Any]) -> dict[str, Any]:
        if technique is Technique.DPO:
            return to_sagemaker_dpo(record)
        if technique is Technique.RFT:
            return to_sagemaker_rlvr(record)
        return to_sagemaker_sft(record)

    # ---- training ----

    def check(
        self, request: TrainingRequest, evaluator_arn: str | None = "pending"
    ) -> tuple[Recipe, dict[str, str]]:
        """Everything that can fail before spending: EULA gating, recipe choice, and
        hyperparameters/batch sizes against the live recipe. No bootstrap resources needed."""
        technique, _ = _technique(request, evaluator_arn)
        hub_model = self.hub.model(request.route.capability.backend_model_id)
        if hub_model.gated and not request.accept_eula:
            raise SkyhammerError(
                f"{request.route.model.display_name} has a gated license "
                f"({hub_model.license}). Review it, then pass accept_eula=True."
            )
        report = request.dataset.report
        recipe = hub_model.recipe(
            technique,
            request.training_type,
            min_tokens=report.max_sample_tokens if report else None,
        )
        return recipe, recipe_hyperparameters(request, recipe, self.hub.params(recipe))

    def build_request(
        self,
        request: TrainingRequest,
        training_uri: str,
        validation_uri: str | None,
        output_uri: str,
        *,
        model_package_group_arn: str,
        evaluator_arn: str | None = None,
    ) -> dict[str, Any]:
        """Map a TrainingRequest to CreateTrainingJob kwargs. Reads Hub metadata (no spend)."""
        res = self.session.resources_for(self.region)
        technique, evaluator_arn = _technique(request, evaluator_arn)
        model_id = request.route.capability.backend_model_id
        recipe, hyperparameters = self.check(request, evaluator_arn)

        serverless: dict[str, Any] = {
            "BaseModelArn": SAGEMAKER_HUB_MODEL_ARN.format(region=self.region, model_id=model_id),
            "JobType": "FineTuning",
            "CustomizationTechnique": technique,
            "AcceptEula": request.accept_eula,
            "SequenceLength": recipe.sequence_length,
        }
        if recipe.training_type.value == "lora":
            serverless["Peft"] = "LORA"
        if evaluator_arn:
            serverless["EvaluatorArn"] = evaluator_arn

        channels = [_channel("train", training_uri)]
        if validation_uri:
            channels.append(_channel("validation", validation_uri))
        # Uncompressed output: Bedrock Custom Model Import reads HF files directly.
        output: dict[str, Any] = {"S3OutputPath": output_uri, "CompressionType": "NONE"}
        if res.kms_key_arn:
            output["KmsKeyId"] = res.kms_key_arn

        vpc = self.session.settings.vpc_for(self.region)
        return {
            **(
                {
                    "VpcConfig": {
                        "SecurityGroupIds": vpc.security_group_ids,
                        "Subnets": vpc.subnet_ids,
                    }
                }
                if vpc
                else {}
            ),
            "TrainingJobName": request.job_name,
            "RoleArn": res.sagemaker_role_arn,
            "ServerlessJobConfig": serverless,
            "InputDataConfig": channels,
            "OutputDataConfig": output,
            "HyperParameters": hyperparameters,
            # Required for serverless fine-tuning: each run registers a model package version.
            "ModelPackageConfig": {
                "ModelPackageGroupArn": model_package_group_arn,
                # Continued fine-tuning starts from the earlier run's registered model.
                **({"SourceModelPackageArn": request.source_model} if request.source_model else {}),
            },
            "Tags": tagging.as_aws(request.tags),
            **self._mlflow(request),
            **(
                {"StoppingCondition": {"MaxRuntimeInSeconds": int(request.max_hours * 3600)}}
                if request.max_hours
                else {}
            ),
        }

    def _mlflow(self, request: TrainingRequest) -> dict[str, Any]:
        arn = self.session.settings.mlflow_arn
        if not arn:
            return {}
        return {
            "MlflowConfig": {
                "MlflowResourceArn": arn,
                "MlflowExperimentName": request.output_model,
                "MlflowRunName": request.job_name,
            }
        }

    def find(self, request: TrainingRequest) -> JobHandle | None:
        resp = self.sagemaker.list_training_jobs(NameContains=request.job_name)
        for job in resp.get("TrainingJobSummaries", []):
            if job["TrainingJobName"] == request.job_name:
                return JobHandle(
                    self.name, job["TrainingJobName"], job["TrainingJobArn"], self.region
                )
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
        needs_evaluator = bool(request.grader and request.technique is Technique.RFT)
        # Build (and so validate) the request before creating anything for it.
        kwargs = self.build_request(
            request,
            training_uri,
            validation_uri,
            f"{prefix}/output/",
            model_package_group_arn=self.ensure_model_package_group(request),
            evaluator_arn="pending" if needs_evaluator else None,
        )
        if needs_evaluator:
            assert request.grader is not None
            g = request.grader.deploy(self.session, region=self.region)  # no-op for Lambdas
            assert g.lambda_arn
            kwargs["ServerlessJobConfig"]["EvaluatorArn"] = self.ensure_evaluator(
                g.lambda_arn, request.tags
            )
        resp = self.sagemaker.create_training_job(**kwargs)
        return JobHandle(self.name, request.job_name, resp["TrainingJobArn"], self.region)

    def ensure_evaluator(self, lambda_arn: str, tags: dict[str, str] | None = None) -> str:
        """Register a reward Lambda as a SageMaker Hub evaluator (JsonDoc) and return its
        hub-content ARN: RLVR's EvaluatorArn must be one, not a Lambda ARN (seen live,
        2026-10-29). Lives in Skyhammer's own tagged private hub; idempotent per Lambda."""
        self._ensure_hub(tags)
        name = lambda_arn.rsplit(":", 1)[-1][:63]
        try:
            return str(self._describe_evaluator(name)["HubContentArn"])
        except ClientError as e:
            if not _not_found(e):
                raise
        self.sagemaker.import_hub_content(**evaluator_hub_content(name, lambda_arn))
        for _ in range(60):
            d = self._describe_evaluator(name)
            if d["HubContentStatus"] == "Available":
                return str(d["HubContentArn"])
            if d["HubContentStatus"] == "ImportFailed":
                raise SkyhammerError(f"Evaluator import failed: {d.get('FailureReason')}")
            time.sleep(2)
        raise TimeoutError(f"Evaluator {name} still importing")

    def _describe_evaluator(self, name: str) -> dict[str, Any]:
        return dict(
            self.sagemaker.describe_hub_content(
                HubName=EVALUATOR_HUB, HubContentType="JsonDoc", HubContentName=name
            )
        )

    def delete_evaluator(self, lambda_arn: str) -> None:
        """Remove a reward Lambda's Hub evaluator (all versions)."""
        name = lambda_arn.rsplit(":", 1)[-1][:63]
        try:
            versions = self.sagemaker.list_hub_content_versions(
                HubName=EVALUATOR_HUB, HubContentType="JsonDoc", HubContentName=name
            )["HubContentSummaries"]
        except ClientError as e:
            if _not_found(e):
                return
            raise
        for v in versions:
            self.sagemaker.delete_hub_content(
                HubName=EVALUATOR_HUB,
                HubContentType="JsonDoc",
                HubContentName=name,
                HubContentVersion=v["HubContentVersion"],
            )

    def _ensure_hub(self, tags: dict[str, str] | None) -> None:
        try:
            self.sagemaker.describe_hub(HubName=EVALUATOR_HUB)
            return
        except ClientError as e:
            if not _not_found(e):
                raise
        self.sagemaker.create_hub(
            HubName=EVALUATOR_HUB,
            HubDescription=HUB_DESCRIPTION,
            Tags=tagging.as_aws(
                tagging.build_tags({"name": EVALUATOR_HUB, "kind": "hub"}, tags or {})
            ),
        )
        for _ in range(60):
            if self.sagemaker.describe_hub(HubName=EVALUATOR_HUB)["HubStatus"] == "InService":
                return
            time.sleep(2)

    def ensure_model_package_group(self, request: TrainingRequest) -> str:
        """One model package group per output model (free); retrains add versions."""
        name = request.output_model
        try:
            resp = self.sagemaker.describe_model_package_group(ModelPackageGroupName=name)
        except ClientError as e:
            if "does not exist" not in str(e):
                raise
            resp = self.sagemaker.create_model_package_group(
                ModelPackageGroupName=name,
                ModelPackageGroupDescription=f"skyhammer fine-tunes of {request.route.model.alias}",
                Tags=tagging.as_aws(request.tags),
            )
        arn: str = resp["ModelPackageGroupArn"]
        return arn

    def describe(self, handle: JobHandle) -> JobStatus:
        job = self.sagemaker.describe_training_job(TrainingJobName=handle.name)
        state = _STATE.get(job["TrainingJobStatus"], JobState.PENDING)
        return JobStatus(
            state=state,
            message=job.get("FailureReason") or job.get("SecondaryStatus"),
            output_artifacts=self._artifacts(job) if state is JobState.COMPLETED else None,
            log_url=_LOG_URL.format(region=self.region, name=handle.name),
        )

    def _artifacts(self, job: dict[str, Any]) -> str | None:
        uri: str | None = job.get("ModelArtifacts", {}).get("S3ModelArtifacts")
        if uri or not job.get("OutputModelPackageArn"):
            return uri
        # Serverless jobs register a model package; its container points at the weights.
        pkg = self.sagemaker.describe_model_package(ModelPackageName=job["OutputModelPackageArn"])
        container = pkg.get("InferenceSpecification", {}).get("Containers", [{}])[0]
        source = container.get("ModelDataSource", {}).get("S3DataSource", {}).get("S3Uri")
        result: str | None = source or container.get("ModelDataUrl")
        return result

    def poll(self, record: JobRecord) -> JobStatus:
        """Training status; once training completes (Bedrock serving), drive the import."""
        st = self.describe(record.handle)
        if st.state is not JobState.COMPLETED or record.serving != ServingTarget.BEDROCK.value:
            return st
        common = {"output_artifacts": st.output_artifacts, "log_url": st.log_url}
        imp = self._find_import(record.job_name)
        if imp is None:
            if not st.output_artifacts:
                return JobStatus(
                    JobState.FAILED, "training produced no model artifacts", log_url=st.log_url
                )
            try:
                self._start_import(record, st.output_artifacts)
            except ClientError as e:
                if not is_import_quota_error(e):
                    raise
                # Accounts allow few concurrent imports (1 here); retry on the next poll.
                return JobStatus(
                    JobState.IMPORTING, "waiting for a free Bedrock import slot", **common
                )
            return JobStatus(JobState.IMPORTING, "importing into Bedrock", **common)
        status = imp["status"]
        if status == "Completed":
            return JobStatus(JobState.COMPLETED, output_model_arn=imp["importedModelArn"], **common)
        if status == "Failed":
            return JobStatus(
                JobState.FAILED, f"Bedrock import failed: {imp.get('failureMessage')}", **common
            )
        return JobStatus(JobState.IMPORTING, "importing into Bedrock", **common)

    def logs(self, record: JobRecord, limit: int = 100) -> list[str]:
        """Serverless jobs write a structured ``.../concise`` stream; show it as text."""
        logs = self.session.client("logs", region=self.region)
        group = "/aws/sagemaker/TrainingJobs"
        try:
            streams = logs.describe_log_streams(
                logGroupName=group, logStreamNamePrefix=record.job_name
            )["logStreams"]
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                return ["no logs yet"]
            raise
        names = [s["logStreamName"] for s in streams]
        concise = [n for n in names if n.endswith("/concise")]
        lines: list[str] = []
        for name in concise or names:
            for event in logs.get_log_events(
                logGroupName=group, logStreamName=name, startFromHead=False, limit=limit
            )["events"]:
                lines.append(format_log_event(event["message"]))
        return lines[-limit:] or ["no logs yet"]

    def checkpoints(self, record: JobRecord) -> list[Checkpoint]:
        """``checkpoints/step_NNNN`` folders in the job output, with the training metrics
        logged at that step. They hold FSDP training state (for resuming or offline
        conversion), so they aren't directly servable; the final weights are."""
        res = self.session.resources_for(self.region)
        s3 = self.session.client("s3", region=self.region)
        prefix = f"jobs/{record.job_name}/output/{record.job_name}/output/model/"
        steps: dict[int, str] = {}
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=res.bucket, Prefix=prefix + "checkpoints/step_", Delimiter="/"
        ):
            for p in page.get("CommonPrefixes", []):
                folder = p["Prefix"].rstrip("/").rsplit("/", 1)[-1]
                steps[int(folder.removeprefix("step_"))] = f"s3://{res.bucket}/{p['Prefix']}"
        metrics = {int(row.pop("step")): row for row in self.metrics(record)}
        return [
            Checkpoint(step, metrics.get(step, {}), uri=uri) for step, uri in sorted(steps.items())
        ]

    def metrics(self, record: JobRecord) -> list[dict[str, float]]:
        """Per-step training metrics (``train_metrics.jsonl``) merged with evaluation metrics
        (``metrics.jsonl``, ``eval_`` keys), keyed by ``step``. Empty until written."""
        res = self.session.resources_for(self.region)
        s3 = self.session.client("s3", region=self.region)
        prefix = f"jobs/{record.job_name}/output/{record.job_name}/output/model/"
        rows: dict[int, dict[str, float]] = {}
        for name, kind in (("train_metrics.jsonl", "train/"), ("metrics.jsonl", "eval/")):
            try:
                body = s3.get_object(Bucket=res.bucket, Key=prefix + name)["Body"].read()
            except ClientError:
                continue  # not written yet
            for line in body.decode().splitlines():
                data = json.loads(line)
                step_key = f"{kind}global_step"
                if step_key not in data:
                    continue
                step = int(float(data.pop(step_key)))
                row = rows.setdefault(step, {"step": float(step)})
                for k, v in data.items():
                    if k.startswith(kind):
                        row[k.removeprefix(kind)] = float(v)
        return [rows[s] for s in sorted(rows)]

    def _find_import(self, job_name: str) -> dict[str, Any] | None:
        name = import_job_name(job_name)
        resp = self._bedrock.list_model_import_jobs(nameContains=name)
        for job in resp.get("modelImportJobSummaries", []):
            if job["jobName"] == name:
                result: dict[str, Any] = self._bedrock.get_model_import_job(
                    jobIdentifier=job["jobArn"]
                )
                return result
        return None

    def import_request(self, record: JobRecord, model_dir: str) -> dict[str, Any]:
        res = self.session.resources_for(self.region)
        kwargs: dict[str, Any] = {
            "jobName": import_job_name(record.job_name),
            "importedModelName": record.output_model,
            "roleArn": res.import_role_arn,
            "modelDataSource": {"s3DataSource": {"s3Uri": model_dir}},
            "jobTags": tagging.as_bedrock(record.tags),
            "importedModelTags": tagging.as_bedrock(record.tags),
            "clientRequestToken": record.fingerprint[:64],
        }
        if res.kms_key_arn:
            kwargs["importedModelKmsKeyId"] = res.kms_key_arn
        return kwargs

    def _start_import(self, record: JobRecord, artifacts_uri: str) -> None:
        model_dir = resolve_model_dir(self.session.client("s3", region=self.region), artifacts_uri)
        self._bedrock.create_model_import_job(**self.import_request(record, model_dir))

    def cancel(self, handle: JobHandle) -> None:
        self.sagemaker.stop_training_job(TrainingJobName=handle.name)

    def finalize(self, record: JobRecord) -> ModelRef:
        st = self.poll(record)
        if st.state is not JobState.COMPLETED or not st.output_artifacts:
            raise RuntimeError(
                f"Job {record.job_name!r} has not completed (state={st.state.value})"
            )
        bedrock = record.serving == ServingTarget.BEDROCK.value
        return ModelRef(
            name=record.output_model,
            model_arn=st.output_model_arn
            if bedrock and st.output_model_arn
            else st.output_artifacts,
            backend=self.name,
            base_alias=record.base_model,
            region=self.region,
            api="openai_chat",
            serving=record.serving,
            artifacts_uri=st.output_artifacts,
        )

    # ---- serving ----

    def deploy(self, model: ModelRef, opts: DeployOptions) -> DeploymentRef:
        if model.serving == ServingTarget.BEDROCK.value:
            # Imported models are on-demand as soon as the import completes; no deployment
            # resource is needed. The imported model ARN is the invocation target.
            return DeploymentRef(opts.name, model.model_arn, "Active")
        if not model.artifacts_uri:
            raise SkyhammerError(f"{model.name} has no trained weights to serve")
        if opts.on:
            names = EndpointNames.for_adapter(opts.on, opts.name)
            adapter_dir = resolve_adapter_dir(
                self.session.client("s3", region=self.region), model.artifacts_uri
            )
            arn = EndpointDeployer(self.session, self.region).deploy_adapter(
                names, adapter_dir, tags=opts.tags
            )
            return DeploymentRef(names.deployment_name, arn, "Active")
        names = EndpointNames.for_deployment(opts.name)
        model_dir = resolve_model_dir(
            self.session.client("s3", region=self.region), model.artifacts_uri
        )
        arn = EndpointDeployer(self.session, self.region).deploy(
            names,
            model_dir=model_dir,
            hosting=self.hosting_for(model.base_alias),
            scaling=opts.scaling or Scaling(),
            instance_type=opts.instance_type,
            params_b=_params_b(model.base_alias),
            tags=opts.tags,
            log=lambda msg: log.info("%s: %s", opts.name, msg),
        )
        return DeploymentRef(opts.name, arn, "Active")

    def deploy_base(
        self, alias: str, opts: DeployOptions, *, accept_eula: bool = False
    ) -> tuple[DeploymentRef, ModelRef]:
        """Dedicated endpoint for a *base* model from AWS's prepacked Hub artifacts."""
        from skyhammer.catalog import BackendKind, get_model

        entry = get_model(alias)
        caps = [c for c in entry.capabilities if c.backend is BackendKind.SAGEMAKER]
        if not caps:
            raise SkyhammerError(f"{entry.display_name} has no SageMaker serving package")
        hub_model = self.hub.model(caps[0].backend_model_id)
        if hub_model.base is None:
            raise SkyhammerError(f"No prepacked serving artifacts published for {hub_model.name}")
        if hub_model.gated and not accept_eula:
            raise SkyhammerError(
                f"{entry.display_name} has a gated license "
                f"({hub_model.license}); pass accept_eula=True"
            )
        names = EndpointNames.for_deployment(opts.name)
        hosting = hub_model.base.hosting
        scaling = opts.scaling or Scaling()
        if opts.adapters:
            # Serve the base weights with the fine-tuning recipe's LoRA-enabled vLLM container
            # and settings. The prepacked base container ignores LoRA options (live
            # 2026-10-30: adapter answers matched the base, then the container failed to
            # start); with the recipe's container the adapter's trained style came through.
            # Adapters need a running base copy, so keep one warm.
            hosting = self.hosting_for(alias)
            if opts.scaling is None:
                scaling = Scaling(min_copies=1)
        arn = EndpointDeployer(self.session, self.region).deploy(
            names,
            model_dir=hub_model.base.artifact_uri,
            hosting=hosting,
            # No HubAccessConfig: it's for private curated hubs; with SageMakerPublicHub
            # content CreateModel rejects it ("Invalid HubContentArn", seen live 2026-10-29).
            accept_eula=hub_model.gated,
            network_isolation=hub_model.base.network_isolation,
            merged=False,
            scaling=scaling,
            instance_type=opts.instance_type,
            params_b=entry.params_b,
            tags=opts.tags,
            log=lambda msg: log.info("%s: %s", opts.name, msg),
        )
        ref = ModelRef(
            name=opts.name,
            model_arn=hub_model.arn,
            backend=self.name,
            base_alias=entry.alias,
            region=self.region,
            api="openai_chat",
            serving=ServingTarget.SAGEMAKER.value,
            artifacts_uri=hub_model.base.artifact_uri,
        )
        return DeploymentRef(opts.name, arn, "Active"), ref

    def hosting_for(self, base_alias: str) -> HostingConfig:
        """AWS's validated serving profile (LMI image, env, instance, compute) for the model."""
        from skyhammer.catalog import BackendKind, get_model

        entry = get_model(base_alias)
        cap = next(c for c in entry.capabilities if c.backend is BackendKind.SAGEMAKER)
        model = self.hub.model(cap.backend_model_id)
        for recipe in model.recipes:
            if recipe.technique == "SFT" and recipe.hosting:
                return recipe.hosting
        hosted = [r.hosting for r in model.recipes if r.hosting]
        if not hosted:
            raise SkyhammerError(
                f"No SageMaker hosting profile published for {cap.backend_model_id}"
            )
        return hosted[0]

    def describe_deployment(self, arn: str) -> DeploymentRef:
        name = arn.rsplit("/", 1)[-1]
        if ":inference-component/" in arn:  # a LoRA adapter on a shared endpoint
            ic = self.sagemaker.describe_inference_component(InferenceComponentName=name)
            names = EndpointNames.for_deployment(f"{ic['EndpointName']}/{name}")
            status = EndpointDeployer(self.session, self.region).status(names)
            return DeploymentRef(names.deployment_name, arn, status)
        if ":endpoint/" not in arn:
            return DeploymentRef(name, arn, "Active")  # Bedrock imported model
        status = EndpointDeployer(self.session, self.region).status(
            EndpointNames.for_deployment(name)
        )
        return DeploymentRef(name, arn, status)

    def delete_deployment(self, arn: str) -> None:
        if ":inference-component/" in arn:
            name = arn.rsplit("/", 1)[-1]
            EndpointDeployer(self.session, self.region).delete(
                EndpointNames.for_deployment(f"endpoint/{name}")  # adapter: its component only
            )
            return
        if ":endpoint/" in arn:
            EndpointDeployer(self.session, self.region).delete(
                EndpointNames.for_deployment(arn.rsplit("/", 1)[-1])
            )
        # Bedrock imported models have no deployment resource; the model itself is kept.

    def delete_model(self, model: ModelRef) -> None:
        if model.serving == ServingTarget.BEDROCK.value:
            self._bedrock.delete_imported_model(modelIdentifier=model.model_arn)
        # SageMaker-served: weights stay in S3 (the bootstrap bucket is versioned/retained).


def resolve_adapter_dir(s3: Any, artifacts_uri: str) -> str:
    """The LoRA adapter directory (``checkpoints/hf/``) in training output. Train with
    ``hyperparameters_extra={"merge_weights": False}`` to keep the adapter separate."""
    base = artifacts_uri.rstrip("/") + "/"
    bucket, prefix = base.removeprefix("s3://").split("/", 1)
    for candidate in (prefix + "checkpoints/hf/", prefix):
        try:
            s3.head_object(Bucket=bucket, Key=candidate + "adapter_config.json")
            return f"s3://{bucket}/{candidate}"
        except s3.exceptions.ClientError:
            continue
    raise SkyhammerError(
        f"No LoRA adapter (adapter_config.json) under {artifacts_uri}. Adapters come from "
        'LoRA jobs with serving="sagemaker" and hyperparameters_extra={"merge_weights": '
        "False}; merged weights deploy as their own endpoint instead."
    )


def resolve_model_dir(s3: Any, artifacts_uri: str) -> str:
    """Find the merged HF model directory inside training output.

    Recipes write merged weights to ``checkpoints/hf_merged/`` and adapter-only output to
    ``checkpoints/hf/``; some outputs put ``config.json`` at the root.
    """
    base = artifacts_uri.rstrip("/") + "/"
    bucket, prefix = base.removeprefix("s3://").split("/", 1)

    def exists(key: str) -> bool:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except s3.exceptions.ClientError:
            return False

    for candidate in (prefix, prefix + "checkpoints/hf_merged/"):
        if exists(candidate + "config.json"):
            return f"s3://{bucket}/{candidate}"
    if exists(prefix + "checkpoints/hf/adapter_config.json"):
        raise SkyhammerError(
            f"{artifacts_uri} contains only a LoRA adapter; Bedrock import needs merged "
            "weights (recipe merge_weights=true)."
        )
    raise SkyhammerError(f"No Hugging Face model (config.json) found under {artifacts_uri}")


def _channel(name: str, uri: str) -> dict[str, Any]:
    return {
        "ChannelName": name,
        "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": uri}},
    }


def format_log_event(message: str) -> str:
    """Concise-stream JSON -> "19:42:37 TRAINING_COMPLETE Training completed"; raw otherwise."""
    try:
        event = json.loads(message)
    except ValueError:
        return message.rstrip()
    if not isinstance(event, dict) or "message" not in event:
        return message.rstrip()
    ts = str(event.get("timestamp", ""))[11:19]
    return f"{ts} {event.get('category', '')} {event['message']}".strip()


def _params_b(alias: str) -> float | None:
    from skyhammer.catalog import get_model

    return get_model(alias).params_b
