"""Backend protocol. A backend maps a ``TrainingRequest`` onto one AWS customization path."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from skyhammer.catalog import DatasetLimits, ModelEntry, Technique, TrainingType
from skyhammer.errors import UnsupportedParameterError
from skyhammer.jobs import JobHandle, JobRecord, JobStatus, ModelRef, TrainingRequest
from skyhammer.serving import Scaling
from skyhammer.session import Session


def map_hyperparameters(
    request: TrainingRequest, keys: dict[str, str], hint: str
) -> dict[str, str]:
    """Map SDK hyperparameter names to backend keys. Unknown names are rejected, never dropped;
    ``hyperparameters_extra`` passes backend-native keys through unchanged."""
    set_fields = request.hyperparameters.set_fields()
    unsupported = sorted(k for k in set_fields if k not in keys)
    if unsupported:
        raise UnsupportedParameterError(
            f"{request.route.backend.value} customization of {request.route.model.display_name} "
            f"does not support: {', '.join(unsupported)}. {hint}"
        )
    hp = {keys[k]: str(v) for k, v in set_fields.items()}
    hp.update({k: str(v) for k, v in request.hyperparameters.extra.items()})
    return hp


class DeploymentType(StrEnum):
    ON_DEMAND = "on_demand"
    PROVISIONED = "provisioned"


@dataclass(frozen=True)
class DeployOptions:
    name: str
    type: DeploymentType = DeploymentType.ON_DEMAND
    model_units: int | None = None  # provisioned throughput only
    tags: dict[str, str] | None = None
    # SageMaker endpoints only.
    scaling: Scaling | None = None
    # One type pins it; a list is a capacity-fallback order; None picks automatically.
    instance_type: str | list[str] | None = None
    # Multi-LoRA: serve this LoRA adapter on an existing shared endpoint (``on``), or
    # deploy a base model's endpoint ready to host adapters (``adapters``).
    on: str | None = None
    adapters: bool = False


@dataclass(frozen=True)
class DeploymentRef:
    name: str
    arn: str  # used as modelId for inference
    status: str


class Backend(Protocol):
    name: str
    session: Session
    region: str

    def dataset_limits(
        self, entry: ModelEntry, technique: Technique, training_type: TrainingType
    ) -> DatasetLimits:
        """Limits the training service actually enforces for this model and technique."""
        ...

    def format_record(self, technique: Technique, record: dict[str, Any]) -> dict[str, Any]:
        """Convert a validated OpenAI-shaped record into this backend's training schema."""
        ...

    def find(self, request: TrainingRequest) -> JobHandle | None:
        """Return an existing job for this request (idempotent re-runs), if any."""
        ...

    def submit(self, request: TrainingRequest) -> JobHandle: ...

    def describe(self, handle: JobHandle) -> JobStatus:
        """Read-only status of the training job itself."""
        ...

    def poll(self, record: JobRecord) -> JobStatus:
        """Status of the whole pipeline; may start the next phase when one completes."""
        ...

    def logs(self, record: JobRecord, limit: int = 100) -> list[str]:
        """Recent human-readable training progress lines."""
        ...

    def cancel(self, handle: JobHandle) -> None: ...

    def finalize(self, record: JobRecord) -> ModelRef:
        """Turn a completed job into a Bedrock-servable model (e.g. run Custom Model Import)."""
        ...

    def deploy(self, model: ModelRef, opts: DeployOptions) -> DeploymentRef: ...

    def describe_deployment(self, arn: str) -> DeploymentRef: ...

    def delete_deployment(self, arn: str) -> None: ...

    def delete_model(self, model: ModelRef) -> None:
        """Delete the custom/imported model (stops its monthly storage charge)."""
        ...
