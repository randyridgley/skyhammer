"""Bedrock's OpenAI-compatible fine-tuning API (``bedrock-mantle``).

Reinforcement fine-tuning of open-weight models (Qwen3-32B, GPT-OSS-20B; us-west-2) runs
only here, not through ``CreateModelCustomizationJob``. The flow mirrors OpenAI's:
upload the prompts with the Files API, create a fine-tuning job with a Lambda grader,
then call the resulting ``fine_tuned_model`` on demand (no deployment step).

Reward Lambdas run as the caller, so the identity submitting the job needs
``lambda:InvokeFunction`` on them (Skyhammer deploys Python graders as
``skyhammer-grader-*``).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from skyhammer.backends.base import (
    DeploymentRef,
    DeployOptions,
    map_hyperparameters,
)
from skyhammer.backends.bedrock import RFT_MAX_PROMPTS, RFT_MIN_PROMPTS, to_bedrock_rft
from skyhammer.catalog import DatasetLimits, ModelEntry, Technique, TrainingType
from skyhammer.errors import UnsupportedParameterError
from skyhammer.graders import GraderKind
from skyhammer.jobs import JobHandle, JobRecord, JobState, JobStatus, ModelRef, TrainingRequest
from skyhammer.mantle import MantleClient, MantleError
from skyhammer.session import Session

log = logging.getLogger(__name__)

API = "mantle"  # ModelRef.api / Chat api for models called through bedrock-mantle

_STATE = {
    "validating_files": JobState.PENDING,
    "queued": JobState.PENDING,
    "running": JobState.TRAINING,
    "succeeded": JobState.COMPLETED,
    "failed": JobState.FAILED,
    "cancelled": JobState.STOPPED,
}

# SDK hyperparameter -> OpenAI fine-tuning hyperparameter. Others are rejected, not dropped;
# pass e.g. learning_rate_multiplier through hyperparameters_extra.
_HP_KEYS = {"epochs": "n_epochs", "batch_size": "batch_size"}
_INT_KEYS = {"n_epochs", "batch_size"}
# The service holds out ~10% of rows for validation, and its default batch is 128 prompts
# ("got 90 examples but train_batch_size requires at least 128", live 2026-10-30).
DEFAULT_BATCH_SIZE = 128
TRAIN_FRACTION = 0.9


class BedrockMantleBackend:
    name = "mantle"

    def __init__(self, session: Session, region: str | None = None) -> None:
        self.session = session
        self.region = region or session.region
        # Jobs are created with a short-term Bedrock API key, as AWS documents: a SigV4-
        # created job trained but never called its reward Lambda ("internal error", live
        # 2026-10-30).
        self.client = MantleClient(session, self.region, auth="api_key")

    # ---- data ----

    def dataset_limits(
        self, entry: ModelEntry, technique: Technique, training_type: TrainingType
    ) -> DatasetLimits:
        if technique is not Technique.RFT:
            raise UnsupportedParameterError("bedrock-mantle fine-tuning supports rft only")
        return entry.limits.model_copy(
            update={"min_samples": RFT_MIN_PROMPTS, "max_samples": RFT_MAX_PROMPTS}
        )

    def format_record(self, technique: Technique, record: dict[str, Any]) -> dict[str, Any]:
        return to_bedrock_rft(record)

    # ---- training ----

    def build_request(
        self, request: TrainingRequest, training_file: str, *, grader_arn: str | None
    ) -> dict[str, Any]:
        """Pure mapping from TrainingRequest to a ``POST /v1/fine_tuning/jobs`` body."""
        if request.technique is not Technique.RFT:
            raise UnsupportedParameterError("bedrock-mantle fine-tuning supports rft only")
        g = request.grader
        if g is None:
            raise UnsupportedParameterError("rft requires a grader (a Python function or Lambda)")
        if g.kind is GraderKind.LLM_JUDGE:
            raise UnsupportedParameterError(
                "bedrock-mantle RFT takes Lambda rewards; write the judge call in a @grader"
            )
        arn = grader_arn if grader_arn is not None else g.lambda_arn
        if arn is None:
            raise UnsupportedParameterError("deploy the grader first (submit does this)")
        hp: dict[str, Any] = {}
        for k, v in map_hyperparameters(
            request, _HP_KEYS, "Pass other keys with hyperparameters_extra."
        ).items():
            hp[k] = int(v) if k in _INT_KEYS else float(v)
        _check_batch(request.dataset.num_records, hp.get("batch_size"))
        reinforcement: dict[str, Any] = {"grader": {"type": "lambda", "lambda": {"function": arn}}}
        if hp:
            reinforcement["hyperparameters"] = hp
        return {
            "model": request.route.capability.backend_model_id,
            "training_file": training_file,
            "method": {"type": "reinforcement", "reinforcement": reinforcement},
        }

    def find(self, request: TrainingRequest) -> JobHandle | None:
        # The API takes no job name; match our training file, which is named after the job.
        try:
            jobs = self.client.request("GET", "/v1/fine_tuning/jobs?limit=100").get("data", [])
        except MantleError:
            return None
        for job in jobs:
            if job.get("status") in ("failed", "cancelled"):
                continue
            if job.get("training_file") in self._file_ids(request.job_name):
                return JobHandle(self.name, request.job_name, job["id"], self.region)
        return None

    def _file_ids(self, job_name: str) -> set[str]:
        files = self.client.request("GET", "/v1/files?purpose=fine-tune").get("data", [])
        return {f["id"] for f in files if f.get("filename") == f"{job_name}.jsonl"}

    def submit(self, request: TrainingRequest) -> JobHandle:
        body = request.dataset._encode(self, request.technique)  # validated and auto-fixed rows
        self.build_request(request, "file-pending", grader_arn="pending")  # validate first
        assert request.grader is not None
        # The reward Lambda runs in the training region, invoked as the caller.
        grader_arn = request.grader.deploy(
            self.session, region=self.region, contract="bedrock"
        ).lambda_arn
        uploaded = self.client.request(
            "POST",
            "/v1/files",
            fields={"purpose": "fine-tune"},
            files={"file": (f"{request.job_name}.jsonl", body)},
        )
        job = self.client.request(
            "POST",
            "/v1/fine_tuning/jobs",
            json_body=self.build_request(request, uploaded["id"], grader_arn=grader_arn),
        )
        return JobHandle(self.name, request.job_name, job["id"], self.region)

    def describe(self, handle: JobHandle) -> JobStatus:
        job = self.client.request("GET", f"/v1/fine_tuning/jobs/{handle.job_id}")
        error = job.get("error") or {}
        return JobStatus(
            state=_STATE.get(job.get("status", ""), JobState.PENDING),
            message=error.get("message") or job.get("status"),
            output_model_arn=job.get("fine_tuned_model"),
        )

    def poll(self, record: JobRecord) -> JobStatus:
        return self.describe(record.handle)

    def logs(self, record: JobRecord, limit: int = 100) -> list[str]:
        """Job events: progress, training metrics and checkpoints, oldest first."""
        events = self.client.request(
            "GET", f"/v1/fine_tuning/jobs/{record.job_id}/events?limit={min(limit, 100)}"
        ).get("data", [])
        lines = []
        for e in reversed(events):
            data = f" {json.dumps(e['data'])}" if e.get("data") else ""
            lines.append(f"{e.get('level', 'info')}: {e.get('message', '')}{data}")
        return lines or ["no events yet"]

    def cancel(self, handle: JobHandle) -> None:
        self.client.request("POST", f"/v1/fine_tuning/jobs/{handle.job_id}/cancel")

    def finalize(self, record: JobRecord) -> ModelRef:
        st = self.describe(record.handle)
        if st.state is not JobState.COMPLETED or not st.output_model_arn:
            raise RuntimeError(
                f"Job {record.job_name!r} has not completed (state={st.state.value})"
            )
        return ModelRef(
            name=record.output_model,
            model_arn=st.output_model_arn,  # the fine_tuned_model ID, called on demand
            backend=self.name,
            base_alias=record.base_model,
            region=self.region,
            api=API,
        )

    # ---- serving: fine-tuned models are on demand as soon as the job succeeds ----

    def deploy(self, model: ModelRef, opts: DeployOptions) -> DeploymentRef:
        return DeploymentRef(opts.name, model.model_arn, "Active")

    def describe_deployment(self, arn: str) -> DeploymentRef:
        return DeploymentRef(arn, arn, "Active")

    def delete_deployment(self, arn: str) -> None:
        return None  # nothing deployed

    def delete_model(self, model: ModelRef) -> None:
        try:
            self.client.request("DELETE", f"/v1/models/{model.model_arn}")
        except MantleError as e:
            if e.status not in (404, 405):
                raise
            log.warning("bedrock-mantle can't delete %s (%s)", model.model_arn, e.status)


def _check_batch(rows: int, batch_size: int | None) -> None:
    train_rows = int(rows * TRAIN_FRACTION)
    batch = batch_size or DEFAULT_BATCH_SIZE
    if train_rows < batch:
        raise UnsupportedParameterError(
            f"~{train_rows} of {rows} prompts are used for training (the rest validate), but "
            f"the batch size is {batch}"
            + ("" if batch_size else " by default")
            + f". Add prompts (>= {int(batch / TRAIN_FRACTION) + 1}) or pass a smaller "
            "batch_size."
        )
