"""Pick a training backend, region, and serving target for (model, technique).

Preference order, chosen to make the common case need zero configuration:

1. Serving: Bedrock (serverless, pay-per-use) before SageMaker endpoints (instance-hours).
2. Region: the home region first, then other regions (if ``cross_region``).
3. Backend: Bedrock-native (no merge/import step) before SageMaker serverless.
"""

from __future__ import annotations

from dataclasses import dataclass

from skyhammer.catalog import (
    BackendKind,
    Capability,
    ModelEntry,
    ServingTarget,
    Technique,
    TrainingType,
)
from skyhammer.errors import UnsupportedRouteError

_BACKENDS = (BackendKind.BEDROCK, BackendKind.SAGEMAKER)
_SERVING = (ServingTarget.BEDROCK, ServingTarget.SAGEMAKER)


@dataclass(frozen=True)
class Route:
    model: ModelEntry
    capability: Capability
    region: str
    serving: ServingTarget
    home_region: str

    @property
    def backend(self) -> BackendKind:
        return self.capability.backend

    @property
    def cross_region(self) -> bool:
        return self.region != self.home_region

    def describe(self) -> str:
        where = (
            f"{self.region} (cross-region from {self.home_region})"
            if self.cross_region
            else self.region
        )
        return f"train on {self.backend.value} in {where}, serve on {self.serving.value}"


def _region_order(
    home: str, candidates: list[Capability], cross_region: bool, allowed_regions: list[str] | None
) -> list[str]:
    others = sorted({r for c in candidates for r in c.regions} - {home}) if cross_region else []
    regions = [home, *others]
    if allowed_regions:
        regions = [r for r in regions if r in allowed_regions]
    return regions


def _serving_problem(
    model: ModelEntry, cap: Capability, region: str, target: ServingTarget
) -> str | None:
    if target is ServingTarget.SAGEMAKER:
        if cap.backend is not BackendKind.SAGEMAKER:
            return "only SageMaker-trained weights can be served from a SageMaker endpoint"
        return None
    if region not in model.bedrock_serving_regions:
        return f"Bedrock can't serve {model.alias} in {region}"
    if cap.backend is BackendKind.BEDROCK and not cap.on_demand:
        return "native customization here needs provisioned throughput to serve"
    return None


def resolve(
    model: ModelEntry,
    technique: Technique,
    region: str,
    *,
    backend: BackendKind | None = None,
    training_type: TrainingType | None = None,
    serving: ServingTarget | None = None,
    cross_region: bool = True,
    allowed_regions: list[str] | None = None,
) -> Route:
    candidates = model.capabilities_for(technique)
    if not candidates:
        supported = sorted({c.technique.value for c in model.capabilities})
        raise UnsupportedRouteError(
            f"{model.display_name} does not support {technique.value}.",
            alternatives=[f"techniques supported for {model.alias}: {', '.join(supported)}"],
        )

    regions = _region_order(region, candidates, cross_region, allowed_regions)
    rejected: dict[str, None] = {}  # ordered set of reasons
    for target in (serving,) if serving else _SERVING:
        for r in regions:
            for kind in (backend,) if backend else _BACKENDS:
                for cap in candidates:
                    if cap.backend is not kind:
                        continue
                    if r not in cap.regions:
                        rejected[f"{kind.value} {technique.value} not available in {r}"] = None
                        continue
                    if training_type and training_type not in cap.training_types:
                        rejected[f"{kind.value}: {training_type.value} fine-tuning unsupported"] = (
                            None
                        )
                        continue
                    problem = _serving_problem(model, cap, r, target)
                    if problem:
                        rejected[f"{kind.value} in {r}: {problem}"] = None
                        continue
                    return Route(model, cap, r, target, home_region=region)

    hints = list(rejected)
    if not cross_region:
        hints.append("enable cross_region to consider other regions")
    raise UnsupportedRouteError(
        f"No route for {technique.value} on {model.display_name} from {region}.",
        alternatives=hints,
    )
