"""Declarative YAML job specs for ``skyhammer apply``."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from skyhammer.catalog import BackendKind, ServingTarget, Technique, TrainingType
from skyhammer.errors import ConfigError
from skyhammer.graders import Grader, grader
from skyhammer.serving import Scaling


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetSpec(_Strict):
    path: Path
    name: str | None = None
    validation_path: Path | None = None
    validation_split: float | None = Field(default=None, gt=0, lt=1)
    auto_fix: bool = False

    @model_validator(mode="after")
    def _one_validation_source(self) -> DatasetSpec:
        if self.validation_path and self.validation_split:
            raise ValueError("set validation_path or validation_split, not both")
        return self


class GraderSpec(_Strict):
    builtin: str | None = None  # exact_match | contains | json_valid
    lambda_arn: str | None = None
    llm_judge: dict[str, str] | None = None  # {model: ..., rubric: ...}
    python: str | None = None  # "module:function"

    @model_validator(mode="after")
    def _exactly_one(self) -> GraderSpec:
        sources = (self.builtin, self.lambda_arn, self.llm_judge, self.python)
        if sum(x is not None for x in sources) != 1:
            raise ValueError("grader needs exactly one of builtin, lambda_arn, llm_judge, python")
        return self

    def build(self) -> Grader:
        if self.builtin:
            return Grader.builtin(self.builtin)
        if self.lambda_arn:
            return Grader.from_lambda(self.lambda_arn)
        if self.llm_judge:
            return Grader.llm_judge(self.llm_judge["model"], self.llm_judge.get("rubric", ""))
        module_name, _, attr = (self.python or "").partition(":")
        if not module_name or not attr:
            raise ValueError("python grader must be 'module:function'")
        obj = getattr(importlib.import_module(module_name), attr)
        return obj if isinstance(obj, Grader) else grader(obj)


class DeploySpec(_Strict):
    type: Literal["on_demand", "provisioned", "none"] = "on_demand"
    name: str | None = None
    # Provisioned Throughput only (billed per model-unit hour; see LLM.deploy).
    model_units: int = 1
    commitment: Literal["none", "1_month", "6_months"] | None = None
    max_hourly_usd: float | None = None
    accept_commitment: bool = False
    # SageMaker-served models only.
    scaling: Scaling | None = None
    instance_type: str | list[str] | None = None


class EvalSpec(_Strict):
    dataset: Path
    graders: list[GraderSpec] = []


class GuardrailSpec(_Strict):
    max_cost_usd: float | None = None


class BudgetSpec(_Strict):
    """Monthly AWS Budget on the model's tagged spend (training, storage, serving), with
    email alerts at 80% and 100% of actual and 100% of forecast."""

    monthly_usd: float = Field(gt=0)
    emails: list[str] = Field(min_length=1)


class JobSpec(_Strict):
    api_version: Literal["skyhammer/v1"] = Field(alias="apiVersion")
    kind: Literal["FineTuningJob"]
    name: str
    base_model: str
    technique: Technique
    dataset: DatasetSpec
    training_type: TrainingType = TrainingType.LORA
    backend: BackendKind | None = None
    serving: ServingTarget | None = None
    accept_eula: bool = False
    teacher_model: str | None = None
    grader: GraderSpec | None = None
    hyperparameters: dict[str, Any] = {}
    deploy: DeploySpec = DeploySpec()
    eval: EvalSpec | None = None
    guardrails: GuardrailSpec = GuardrailSpec()
    budget: BudgetSpec | None = None
    tags: dict[str, str] = {}

    @model_validator(mode="after")
    def _technique_requirements(self) -> JobSpec:
        if self.technique is Technique.DISTILLATION and not self.teacher_model:
            raise ValueError("distillation requires teacher_model")
        if self.technique is Technique.RFT and not self.grader:
            raise ValueError("rft requires grader")
        return self


def load_spec(path: str | Path) -> JobSpec:
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text())
        spec = JobSpec.model_validate(data)
    except (yaml.YAMLError, ValueError) as e:
        raise ConfigError(f"Invalid job spec {p}: {e}") from e
    # Resolve dataset paths relative to the spec file.
    base = p.parent
    spec.dataset.path = base / spec.dataset.path
    if spec.dataset.validation_path:
        spec.dataset.validation_path = base / spec.dataset.validation_path
    if spec.eval:
        spec.eval.dataset = base / spec.eval.dataset
    return spec
