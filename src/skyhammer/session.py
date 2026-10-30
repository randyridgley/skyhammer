"""boto3 session/client factory and discovery of bootstrap resources."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from skyhammer.config import Settings, get_settings
from skyhammer.errors import BootstrapMissingError, ConfigError

_BOTO_CONFIG = Config(
    retries={"mode": "adaptive", "max_attempts": 10}, user_agent_extra="skyhammer"
)


@dataclass(frozen=True)
class Resources:
    """Account/region resources the SDK needs. From config, else bootstrap stack outputs."""

    bucket: str
    kms_key_arn: str | None
    bedrock_role_arn: str
    sagemaker_role_arn: str
    import_role_arn: str
    grader_role_arn: str


# Bootstrap stack output key -> Resources field.
_OUTPUT_KEYS = {
    "BucketName": "bucket",
    "KmsKeyArn": "kms_key_arn",
    "BedrockCustomizationRoleArn": "bedrock_role_arn",
    "SageMakerTrainingRoleArn": "sagemaker_role_arn",
    "ModelImportRoleArn": "import_role_arn",
    "GraderLambdaRoleArn": "grader_role_arn",
}


class Session:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.boto = boto3.Session(
            profile_name=self.settings.profile, region_name=self.settings.region
        )
        self._resources: dict[str, Resources] = {}
        if self.settings.api_key:
            # botocore uses this bearer token for Bedrock/Bedrock Runtime clients.
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.settings.api_key
        from skyhammer.pricing import SpendTracker

        self.spend = SpendTracker(max_spend_usd=self.settings.max_spend_usd)

    @property
    def region(self) -> str:
        return self.settings.region

    def client(self, service: str, region: str | None = None) -> Any:
        boto: Any = self.boto  # service name is dynamic; skip boto3-stubs overloads
        return boto.client(service, region_name=region or self.region, config=_BOTO_CONFIG)

    def stack_outputs(self, region: str | None = None) -> dict[str, str] | None:
        cfn = self.client("cloudformation", region=region)
        try:
            stacks = cfn.describe_stacks(StackName=self.settings.bootstrap_stack)["Stacks"]
        except ClientError as e:
            if "does not exist" in str(e):
                return None
            raise
        return {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}

    @property
    def resources(self) -> Resources:
        """Resources in the home region (lineage manifests always live here)."""
        return self.resources_for(self.region)

    def resources_for(self, region: str) -> Resources:
        """Resources in ``region``. Cross-region routes train with that region's bucket/roles."""
        if region not in self._resources:
            self._resources[region] = self._resolve_resources(region)
        return self._resources[region]

    def _resolve_resources(self, region: str) -> Resources:
        s = self.settings
        # Explicit bucket/role settings describe the home region only.
        home = region == self.region
        explicit = {
            "bucket": s.bucket if home else None,
            "kms_key_arn": s.kms_key_arn if home else None,
            "bedrock_role_arn": s.bedrock_role_arn,
            "sagemaker_role_arn": s.sagemaker_role_arn,
            "import_role_arn": s.import_role_arn,
            "grader_role_arn": s.grader_role_arn,
        }
        if all(v for k, v in explicit.items() if k != "kms_key_arn"):
            return Resources(**explicit)  # type: ignore[arg-type]
        outputs = self.stack_outputs(region)
        if outputs is None:
            raise BootstrapMissingError(region)
        discovered = {field: outputs.get(key) for key, field in _OUTPUT_KEYS.items()}
        merged = {k: explicit[k] or discovered[k] for k in explicit}
        missing = [k for k, v in merged.items() if not v and k != "kms_key_arn"]
        if missing:
            raise ConfigError(f"Bootstrap stack is missing outputs for: {', '.join(missing)}")
        return Resources(**merged)  # type: ignore[arg-type]


_default: Session | None = None


def get_session() -> Session:
    global _default
    if _default is None or _default.settings is not get_settings():
        _default = Session()
    return _default
