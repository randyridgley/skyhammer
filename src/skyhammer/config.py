"""Settings resolution.

Precedence (highest first): explicit kwargs -> ``SKYHAMMER_*`` env vars -> ``./skyhammer.yaml``
-> ``~/.config/skyhammer/config.yaml`` -> bootstrap stack outputs (resolved lazily in ``session``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from skyhammer.errors import ConfigError

PROJECT_CONFIG = Path("skyhammer.yaml")
USER_CONFIG = Path.home() / ".config" / "skyhammer" / "config.yaml"
ENV_PREFIX = "SKYHAMMER_"


class Guardrails(BaseModel):
    max_cost_usd: float | None = None
    confirm: bool = True


class Vpc(BaseModel):
    """Private subnets and security groups that training, import and endpoints run in.

    The subnets need a route to S3 (a gateway endpoint) and to the AWS service APIs the
    job uses (interface endpoints or NAT); without them jobs fail to read data."""

    subnet_ids: list[str] = Field(min_length=1)
    security_group_ids: list[str] = Field(min_length=1)


class Settings(BaseModel):
    profile: str | None = None
    region: str = "us-east-1"
    bucket: str | None = None
    kms_key_arn: str | None = None
    bedrock_role_arn: str | None = None
    sagemaker_role_arn: str | None = None
    import_role_arn: str | None = None
    grader_role_arn: str | None = None
    bootstrap_stack: str = "skyhammer-bootstrap"
    # Train/serve in another region when the home region has no route (e.g. Nova from
    # us-west-2). Restrict with allowed_regions for data-residency requirements.
    cross_region: bool = True
    # Serverless calls: "geo" (us./eu./apac. profiles), "global", or "region" (in-region only).
    # Default: geo when cross_region is on, region when it's off.
    inference_routing: Literal["geo", "global", "region"] | None = None
    # Serverless spend guardrail for this process (USD); checked before each call.
    max_spend_usd: float | None = None
    # Look up per-token prices (cached weekly) to fill usage.cost_usd.
    track_costs: bool = True
    # Bedrock API key (bearer token) for serverless calls without AWS credentials.
    api_key: str | None = None
    allowed_regions: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    # SageMaker MLflow App (or tracking server) ARN. When set, SageMaker training jobs log
    # metrics there natively (experiment = output model name, run = job name).
    mlflow_arn: str | None = None
    guardrails: Guardrails = Field(default_factory=Guardrails)
    # VPC-only training, import and serving, per region (subnets are regional), e.g.
    #   vpc: {us-east-1: {subnet_ids: [subnet-a, subnet-b], security_group_ids: [sg-1]}}
    vpc: dict[str, Vpc] = Field(default_factory=dict)

    def vpc_for(self, region: str) -> Vpc | None:
        return self.vpc.get(region)


_ENV_KEYS = ("profile", "region", "bucket", "kms_key_arn", "bootstrap_stack")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _from_env() -> dict[str, Any]:
    out: dict[str, Any] = {
        k: os.environ[ENV_PREFIX + k.upper()]
        for k in _ENV_KEYS
        if ENV_PREFIX + k.upper() in os.environ
    }
    for var in ("SKYHAMMER_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"):
        if var in os.environ and "api_key" not in out:
            out["api_key"] = os.environ[var]
    if "profile" not in out and "AWS_PROFILE" in os.environ:
        out["profile"] = os.environ["AWS_PROFILE"]
    if "region" not in out:
        for var in ("AWS_REGION", "AWS_DEFAULT_REGION"):
            if var in os.environ:
                out["region"] = os.environ[var]
                break
    return out


def load_settings(
    *,
    project_config: Path = PROJECT_CONFIG,
    user_config: Path = USER_CONFIG,
    **overrides: Any,
) -> Settings:
    merged: dict[str, Any] = {}
    for layer in (
        _load_yaml(user_config),
        _load_yaml(project_config),
        _from_env(),
        {k: v for k, v in overrides.items() if v is not None},
    ):
        merged.update(layer)
    return Settings.model_validate(merged)


_current: Settings | None = None


def configure(**overrides: Any) -> Settings:
    """Set process-wide settings, e.g. ``skyhammer.configure(region="us-west-2")``."""
    global _current
    _current = load_settings(**overrides)
    return _current


def get_settings() -> Settings:
    global _current
    if _current is None:
        _current = load_settings()
    return _current
