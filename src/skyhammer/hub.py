"""SageMaker Public Hub metadata: recipes, hyperparameter specs, gating, hosting configs.

AWS publishes, per open-weight model, the serverless customization recipes it supports
(technique x LoRA/full x sequence length), each with an override-params document that
lists tunable hyperparameters with types, enums and bounds, plus hosting configs (LMI
container image, instance type, compute requirements). We read these instead of
hard-coding them, validate user hyperparameters locally before any spend, and cache
the documents on disk (they are versioned, so a cache entry never goes stale).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skyhammer.catalog import Technique, TrainingType
from skyhammer.errors import UnsupportedParameterError, UnsupportedRouteError

if TYPE_CHECKING:
    from skyhammer.session import Session

HUB_NAME = "SageMakerPublicHub"
CACHE_DIR = Path.home() / ".cache" / "skyhammer" / "hub"

# SDK technique -> Hub CustomizationTechnique. RFT maps by grader kind in the backend.
_TECHNIQUES = {Technique.SFT: "SFT", Technique.DPO: "DPO"}


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    default: Any
    enum: tuple[Any, ...] | None = None
    min: float | None = None
    max: float | None = None

    def check(self, value: Any) -> str:
        """Validate ``value`` and return it in the string form CreateTrainingJob expects."""
        try:
            if self.type == "integer":
                if isinstance(value, bool) or int(value) != float(value):
                    raise ValueError
                value = int(value)
            elif self.type == "float":
                value = float(value)
            elif self.type == "boolean":
                if not isinstance(value, bool):
                    raise ValueError
        except (TypeError, ValueError):
            raise UnsupportedParameterError(
                f"{self.name} must be {self.type}, got {value!r}"
            ) from None
        if self.enum is not None and value not in self.enum:
            raise UnsupportedParameterError(
                f"{self.name}={value!r} not allowed; choose one of {list(self.enum)}"
            )
        if self.min is not None and value < self.min:
            raise UnsupportedParameterError(f"{self.name}={value} is below minimum {self.min}")
        if self.max is not None and value > self.max:
            raise UnsupportedParameterError(f"{self.name}={value} is above maximum {self.max}")
        return str(value).lower() if isinstance(value, bool) else str(value)


@dataclass(frozen=True)
class HostingConfig:
    image_uri: str
    instance_type: str
    gpus: int
    memory_mb: int
    environment: dict[str, str] = field(default_factory=dict)
    cpus: int | None = None


@dataclass(frozen=True)
class Recipe:
    name: str
    technique: str  # Hub value: SFT, DPO, RLVR, RLAIF
    training_type: TrainingType
    sequence_length: str  # e.g. "4K"
    override_params_uri: str | None
    hosting: HostingConfig | None

    @property
    def max_tokens(self) -> int:
        return sequence_tokens(self.sequence_length)


# GPUs per instance for common LMI serving types (for inference component sizing).
_GPUS = {
    "xlarge": 1,
    "2xlarge": 1,
    "4xlarge": 1,
    "8xlarge": 1,
    "12xlarge": 4,
    "16xlarge": 1,
    "24xlarge": 4,
    "48xlarge": 8,
}


def instance_gpus(instance_type: str) -> int:
    size = instance_type.rsplit(".", 1)[-1]
    if instance_type.startswith(("ml.p4", "ml.p5")):
        return 8
    return _GPUS.get(size, 1)


@dataclass(frozen=True)
class BaseHosting:
    """AWS's prepacked serving package for the *base* model (deploy without training)."""

    artifact_uri: str
    hosting: HostingConfig
    network_isolation: bool


@dataclass(frozen=True)
class HubModel:
    name: str
    version: str
    arn: str
    gated: bool
    license: str | None
    recipes: tuple[Recipe, ...]
    base: BaseHosting | None = None

    def recipe(
        self, technique: str, training_type: TrainingType, min_tokens: int | None = None
    ) -> Recipe:
        """Smallest-sequence-length recipe for (technique, LoRA/full) that fits ``min_tokens``."""
        matches = sorted(
            (
                r
                for r in self.recipes
                if r.technique == technique and r.training_type is training_type
            ),
            key=lambda r: r.max_tokens,
        )
        if not matches:
            available = sorted({f"{r.technique}/{r.training_type.value}" for r in self.recipes})
            raise UnsupportedRouteError(
                f"{self.name} has no {technique} {training_type.value} recipe",
                alternatives=available,
            )
        for r in matches:
            if min_tokens is None or r.max_tokens >= min_tokens:
                return r
        raise UnsupportedParameterError(
            f"Longest {technique} recipe for {self.name} is {matches[-1].sequence_length} "
            f"tokens but the dataset needs ~{min_tokens:,}; shorten samples or drop long rows."
        )


def sequence_tokens(label: str) -> int:
    m = re.fullmatch(r"(\d+)K", label)
    if not m:
        raise ValueError(f"Unrecognized sequence length {label!r}")
    return int(m.group(1)) * 1024


def hub_technique(technique: Technique) -> str:
    return _TECHNIQUES[technique]


def parse_hub_model(resp: dict[str, Any]) -> HubModel:
    doc = json.loads(resp["HubContentDocument"])
    recipes = []
    for r in doc.get("RecipeCollection", []):
        if r.get("Type") != "FineTuning" or not r.get("CustomizationTechnique"):
            continue
        hosting = None
        if r.get("HostingConfigs"):
            h = r["HostingConfigs"][0]
            req = h.get("ComputeResourceRequirements", {})
            hosting = HostingConfig(
                image_uri=h["EcrAddress"],
                instance_type=h["InstanceType"],
                gpus=req.get("NumberOfAcceleratorDevicesRequired", 1),
                memory_mb=req.get("MinMemoryRequiredInMb", 8192),
                environment=h.get("Environment", {}),
                cpus=req.get("NumberOfCpuCoresRequired"),
            )
        recipes.append(
            Recipe(
                name=r["Name"],
                technique=r["CustomizationTechnique"],
                training_type=TrainingType.LORA if r.get("Peft") == "LORA" else TrainingType.FULL,
                sequence_length=r.get("SequenceLength", "4K"),
                override_params_uri=r.get("SmtjOverrideParamsS3Uri"),
                hosting=hosting,
            )
        )
    base = None
    if doc.get("HostingArtifactUri") and doc.get("HostingEcrUri"):
        itype = doc.get("DefaultInferenceInstanceType", "ml.g6e.2xlarge")
        env = {
            v["Name"]: str(v["Default"])
            for v in doc.get("InferenceEnvironmentVariables", [])
            if v.get("Scope", "container") == "container" and "Default" in v
        }
        base = BaseHosting(
            artifact_uri=doc["HostingArtifactUri"],
            hosting=HostingConfig(
                image_uri=doc["HostingEcrUri"],
                instance_type=itype,
                gpus=instance_gpus(itype),
                memory_mb=8192,
                environment=env,
            ),
            network_isolation=bool(doc.get("InferenceEnableNetworkIsolation", False)),
        )
    return HubModel(
        name=resp["HubContentName"],
        version=resp["HubContentVersion"],
        arn=resp["HubContentArn"],
        gated=doc.get("ModelAccess", "Open") != "Open",
        license=doc.get("License"),
        recipes=tuple(recipes),
        base=base,
    )


def parse_params(doc: dict[str, Any]) -> dict[str, ParamSpec]:
    """Tunable hyperparameters only; "system" entries (paths, names) are set by the service."""
    return {
        name: ParamSpec(
            name=name,
            type=spec.get("type", "string"),
            default=spec.get("default"),
            enum=tuple(spec["enum"]) if "enum" in spec else None,
            min=spec.get("min"),
            max=spec.get("max"),
        )
        for name, spec in doc.items()
        if spec.get("category") == "hyperparameter"
    }


class HubClient:
    def __init__(self, session: Session, region: str, cache_dir: Path = CACHE_DIR) -> None:
        self.session = session
        self.region = region
        self.cache_dir = cache_dir / region

    def model(self, model_id: str) -> HubModel:
        # Always describe (cheap) so we pick up the latest version; cache only params docs.
        resp = self.session.client("sagemaker", region=self.region).describe_hub_content(
            HubName=HUB_NAME, HubContentType="Model", HubContentName=model_id
        )
        return parse_hub_model(resp)

    def params(self, recipe: Recipe) -> dict[str, ParamSpec]:
        if not recipe.override_params_uri:
            return {}
        bucket, key = recipe.override_params_uri.removeprefix("s3://").split("/", 1)
        cached = self.cache_dir / key.replace("/", "_")
        if cached.is_file():
            doc = json.loads(cached.read_text())
        else:
            body = (
                self.session.client("s3", region=self.region)
                .get_object(Bucket=bucket, Key=key)["Body"]
                .read()
            )
            doc = json.loads(body)
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(doc))
        return parse_params(doc)
