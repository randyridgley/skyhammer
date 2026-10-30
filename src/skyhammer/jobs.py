"""Backend-agnostic job types and the user-facing ``FineTuningJob`` resource."""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from skyhammer.catalog import Technique, TrainingType
from skyhammer.errors import JobFailedError, SkyhammerError
from skyhammer.router import Route

if TYPE_CHECKING:
    from skyhammer.backends.base import Backend
    from skyhammer.dataset import Dataset
    from skyhammer.graders import Grader
    from skyhammer.llm import LLM
    from skyhammer.session import Session


class Hyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # None means the backend's default (Bedrock's, or the SageMaker recipe's).
    epochs: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    warmup_steps: int | None = None  # Bedrock
    warmup_ratio: float | None = None  # SageMaker recipes
    lora_rank: int | None = None
    lora_alpha: int | None = None
    beta: float | None = None  # DPO: how strongly to prefer chosen over rejected
    # Backend-specific passthrough for platform users; validated by the backend.
    extra: dict[str, Any] = {}

    def set_fields(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, exclude_defaults=False, exclude={"extra"})


@dataclass
class TrainingRequest:
    technique: Technique
    route: Route
    dataset: Dataset
    output_model: str
    hyperparameters: Hyperparameters
    training_type: TrainingType = TrainingType.LORA
    validation_dataset: Dataset | None = None
    teacher_model: str | None = None
    grader: Grader | None = None
    accept_eula: bool = False
    tags: dict[str, str] = field(default_factory=dict)
    # Runtime cap for time-billed (RL) training; not in the fingerprint, so changing it
    # reattaches instead of starting a new job.
    max_hours: float | None = None
    # Continued customization: the fine-tune to train further (Bedrock custom model ARN or
    # SageMaker model package ARN) instead of the base model.
    source_model: str | None = None

    def fingerprint(self) -> str:
        """Stable hash of everything that affects training. Same request -> same job."""
        payload = {
            "technique": self.technique.value,
            "base": self.route.capability.backend_model_id,
            "backend": self.route.backend.value,
            "region": self.route.region,
            "serving": self.route.serving.value,
            "dataset": self.dataset.sha256,
            "validation": self.validation_dataset.sha256 if self.validation_dataset else None,
            "output": self.output_model,
            "hp": self.hyperparameters.model_dump(),
            "training_type": self.training_type.value,
            "teacher": self.teacher_model,
            "grader": self.grader.fingerprint() if self.grader else None,
        }
        if self.source_model:  # only when set, so existing jobs keep their names
            payload["source"] = self.source_model
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def job_name(self) -> str:
        return f"{self.output_model}-{self.fingerprint()[:8]}"


class JobState(StrEnum):
    PENDING = "pending"
    TRAINING = "training"
    MERGING = "merging"  # SageMaker: LoRA merge before import
    IMPORTING = "importing"  # SageMaker: Bedrock Custom Model Import
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

    @property
    def terminal(self) -> bool:
        return self in (JobState.COMPLETED, JobState.FAILED, JobState.STOPPED)


@dataclass(frozen=True)
class JobHandle:
    backend: str
    name: str
    job_id: str  # backend identifier (ARN)
    region: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobStatus:
    state: JobState
    message: str | None = None
    output_model_arn: str | None = None
    output_artifacts: str | None = None  # S3 URI of trained weights (SageMaker)
    log_url: str | None = None


@dataclass(frozen=True)
class JobRecord:
    """Everything needed to follow a job after the process that started it has exited.

    Persisted as ``manifests/jobs/<job_name>.json`` in the home-region bucket; backends
    poll, import and finalize from this record alone (not from the in-memory request).
    """

    job_name: str
    job_id: str
    backend: str
    region: str
    technique: str
    base_model: str  # catalog alias
    backend_model_id: str
    serving: str
    output_model: str
    training_type: str
    fingerprint: str
    tags: dict[str, str] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    estimate: str | None = None
    created_at: float = 0.0
    schema_version: int = 1  # bump when the record format changes incompatibly

    @classmethod
    def from_request(
        cls, request: TrainingRequest, handle: JobHandle, estimate: str | None = None
    ) -> JobRecord:
        return cls(
            job_name=handle.name,
            job_id=handle.job_id,
            backend=handle.backend,
            region=handle.region,
            technique=request.technique.value,
            base_model=request.route.model.alias,
            backend_model_id=request.route.capability.backend_model_id,
            serving=request.route.serving.value,
            output_model=request.output_model,
            training_type=request.training_type.value,
            fingerprint=request.fingerprint(),
            tags=dict(request.tags),
            hyperparameters=request.hyperparameters.model_dump(),
            dataset={
                "name": request.dataset.name,
                "sha256": request.dataset.sha256,
                "rows": request.dataset.num_records,
            },
            estimate=estimate,
            created_at=time.time(),
        )

    @property
    def handle(self) -> JobHandle:
        return JobHandle(self.backend, self.job_name, self.job_id, self.region)

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> JobRecord:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class ModelRef:
    """A trained model that Bedrock can serve (custom model ARN or imported model ARN)."""

    name: str
    model_arn: str
    backend: str
    base_alias: str
    region: str
    # "converse" for Bedrock-native custom models; "openai_chat" for imported models, which
    # accept OpenAI ChatCompletion bodies via InvokeModel (Converse is not universal there).
    api: str = "converse"
    serving: str = "bedrock"  # ServingTarget value
    artifacts_uri: str | None = None  # trained weights in S3, kept for later re-import


@dataclass(frozen=True)
class Checkpoint:
    """An intermediate snapshot of a training job.

    ``model_id`` is set when the checkpoint can be called as a model (bedrock-mantle);
    ``uri`` points at stored training state otherwise (SageMaker: FSDP shards in S3, kept
    for resuming or offline conversion, not directly servable).
    """

    step: int
    metrics: dict[str, float] = field(default_factory=dict)
    model_id: str | None = None
    uri: str | None = None
    _ref: ModelRef | None = field(default=None, repr=False, compare=False)

    def llm(self, session: Any = None) -> LLM:
        """The checkpoint as a model you can chat with (callable checkpoints only)."""
        from skyhammer.llm import LLM

        if self._ref is None:
            raise SkyhammerError(
                f"Checkpoint at step {self.step} is training state ({self.uri}), not a model"
            )
        return LLM.from_model_ref(self._ref, session=session)


class FineTuningJob:
    """A training job, possibly multi-phase (train -> import). Reattach from anywhere with
    ``FineTuningJob.get("support-qwen-v1")``."""

    def __init__(
        self, record: JobRecord, backend: Backend, request: TrainingRequest | None = None
    ) -> None:
        self.record = record
        self.request = request  # only set in the process that submitted the job
        self._backend = backend
        self._output: ModelRef | None = None

    @classmethod
    def get(cls, name: str, *, session: Session | None = None) -> FineTuningJob:
        """Reattach by output model name (latest job) or by exact job name."""
        from skyhammer import registry
        from skyhammer.backends import get_backend
        from skyhammer.catalog import BackendKind
        from skyhammer.session import get_session

        session = session or get_session()
        record = registry.find_job(session, name)
        backend = get_backend(BackendKind(record.backend), session, record.region)
        return cls(record, backend)

    @classmethod
    def list(cls, *, session: Session | None = None) -> builtins.list[FineTuningJob]:
        """Latest job for every fine-tuned model in this account/home region."""
        from skyhammer import registry
        from skyhammer.backends import get_backend
        from skyhammer.catalog import BackendKind
        from skyhammer.session import get_session

        session = session or get_session()
        return [
            cls(r, get_backend(BackendKind(r.backend), session, r.region))
            for r in registry.list_jobs(session)
        ]

    @property
    def handle(self) -> JobHandle:
        return self.record.handle

    @property
    def name(self) -> str:
        return self.record.job_name

    def status(self) -> JobStatus:
        """Current status. May advance multi-phase jobs (e.g. start the Bedrock import once
        SageMaker training completes), so ``wait()`` drives the whole pipeline."""
        return self._backend.poll(self.record)

    def logs(self, limit: int = 100) -> builtins.list[str]:
        """Recent training log lines (SageMaker) or step metrics (Bedrock)."""
        return self._backend.logs(self.record, limit)

    def cancel(self) -> None:
        self._backend.cancel(self.handle)

    def _progress_line(self, st: JobStatus) -> str:
        from skyhammer.progress import loss_series, sparkline

        line = f"{st.state.value}" + (f" ({st.message})" if st.message else "")
        if st.state in (
            JobState.TRAINING,
            JobState.MERGING,
            JobState.IMPORTING,
            JobState.COMPLETED,
        ):
            try:
                losses = loss_series(self.metrics())
            except Exception:  # metrics are best-effort; never break a wait
                losses = []
            if losses:
                line += f"  loss {losses[-1]:.4f} {sparkline(losses)}"
        return line

    def _repr_html_(self) -> str:
        """Notebook view: status, route, estimate and the loss curve."""
        from skyhammer.progress import html_table, loss_series, svg_sparkline

        st = self.status()
        losses = loss_series(self.metrics())
        rows = [
            ("job", self.name),
            ("model", f"{self.record.output_model} (from {self.record.base_model})"),
            ("backend", f"{self.record.backend} in {self.record.region}"),
            ("state", st.state.value + (f": {st.message}" if st.message else "")),
        ]
        if self.record.estimate:
            rows.append(("estimate", self.record.estimate))
        if losses:
            rows.append(("loss", f"{losses[-1]:.4f} after {len(losses)} steps"))
            rows.append(("curve", svg_sparkline(losses)))
        return html_table(rows, raw={"curve"})

    def metrics(self) -> builtins.list[dict[str, float]]:
        """Per-step training (and validation) metrics so far, oldest first, each with a
        ``step``: loss, learning rate, gradient norm... whatever the backend reports."""
        read = getattr(self._backend, "metrics", None)
        return read(self.record) if read else []

    def checkpoints(self) -> builtins.list[Checkpoint]:
        """Intermediate checkpoints with their metrics, oldest first (empty where the
        backend keeps none: Bedrock model customization)."""
        list_checkpoints = getattr(self._backend, "checkpoints", None)
        return list_checkpoints(self.record) if list_checkpoints else []

    def wait(
        self,
        *,
        poll_seconds: float = 60,
        timeout_seconds: float | None = None,
        on_status: Any = None,
        progress: bool = False,
    ) -> FineTuningJob:
        """Block until the job is terminal. Raises ``JobFailedError`` on failure.

        ``progress=True`` prints a line whenever the stage or the training loss changes,
        with the loss curve so far as a sparkline."""
        deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
        start, last = time.monotonic(), ""
        while True:
            st = self.status()
            if on_status:
                on_status(st)
            if progress:
                line = self._progress_line(st)
                if line != last:
                    print(f"{(time.monotonic() - start) / 60:5.1f} min  {line}", flush=True)
                    last = line
            if st.state is JobState.COMPLETED:
                return self
            if st.state.terminal:
                raise JobFailedError(self.name, st.message or st.state.value, st.log_url)
            if deadline and time.monotonic() > deadline:
                raise TimeoutError(f"Job {self.name!r} still {st.state.value} after timeout")
            time.sleep(poll_seconds)

    @property
    def output_model(self) -> ModelRef:
        if self._output is None:
            self._output = self._backend.finalize(self.record)
        return self._output

    @property
    def output_llm(self) -> LLM:
        from skyhammer.llm import LLM

        return LLM.from_model_ref(self.output_model, session=self._backend.session)

    def __repr__(self) -> str:
        return (
            f"FineTuningJob(name={self.name!r}, backend={self.record.backend!r}, "
            f"technique={self.record.technique!r})"
        )
