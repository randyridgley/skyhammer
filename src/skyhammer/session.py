"""boto3 session/client factory and discovery of bootstrap resources."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

import boto3
import botocore.session
from botocore.config import Config
from botocore.credentials import AssumeRoleCredentialFetcher, DeferredRefreshableCredentials
from botocore.exceptions import ClientError

from skyhammer.config import Settings, get_settings
from skyhammer.errors import BootstrapMissingError, ConfigError

_BOTO_CONFIG = Config(
    retries={"mode": "adaptive", "max_attempts": 10},
    user_agent_extra="skyhammer",
    max_pool_connections=64,  # AsyncLLM shares one client across worker threads
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
    # Multi-turn RL jobs (TR-12); optional so stacks bootstrapped before it still work.
    agent_job_role_arn: str | None = None


_ROLES = ("bedrock", "sagemaker", "import", "grader", "agent_job")

# Bootstrap stack output key -> Resources field.
_OUTPUT_KEYS = {
    "BucketName": "bucket",
    "KmsKeyArn": "kms_key_arn",
    "BedrockCustomizationRoleArn": "bedrock_role_arn",
    "SageMakerTrainingRoleArn": "sagemaker_role_arn",
    "ModelImportRoleArn": "import_role_arn",
    "GraderLambdaRoleArn": "grader_role_arn",
    "AgentJobRoleArn": "agent_job_role_arn",
}


def _assumed(base: boto3.Session, role_arn: str, external_id: str | None) -> boto3.Session:
    """A boto3 session on ``role_arn``'s credentials, assumed from ``base`` and refreshed
    before they expire (long jobs and ``wait()`` outlive the one-hour default)."""
    source = base.get_credentials()
    if source is None:
        raise ConfigError(f"No AWS credentials to assume {role_arn} from")
    extra: dict[str, Any] = {"RoleSessionName": "skyhammer"}
    if external_id:
        extra["ExternalId"] = external_id
    fetcher = AssumeRoleCredentialFetcher(
        client_creator=base._session.create_client,
        source_credentials=source,
        role_arn=role_arn,
        extra_args=extra,
    )
    core = botocore.session.Session()
    # botocore has no public setter for refreshable credentials; this is how its own
    # assume-role provider installs them.
    core._credentials = DeferredRefreshableCredentials(  # type: ignore[attr-defined]
        refresh_using=fetcher.fetch_credentials, method="assume-role"
    )
    return boto3.Session(botocore_session=core, region_name=base.region_name)


class Session:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.boto = boto3.Session(
            profile_name=self.settings.profile, region_name=self.settings.region
        )
        if self.settings.role_arn:
            self.boto = _assumed(self.boto, self.settings.role_arn, self.settings.external_id)
        self._account_id: str | None = None
        self._resources: dict[str, Resources] = {}
        self._client_lock = threading.Lock()  # boto3 sessions aren't thread-safe
        if self.settings.api_key:
            # botocore uses this bearer token for Bedrock/Bedrock Runtime clients.
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.settings.api_key
        from skyhammer.pricing import SpendTracker

        self.spend = SpendTracker(max_spend_usd=self.settings.max_spend_usd)

    @property
    def region(self) -> str:
        return self.settings.region

    @property
    def account_id(self) -> str:
        if self._account_id is None:
            self._account_id = str(self.client("sts").get_caller_identity()["Account"])
        return self._account_id

    def assume(
        self, role_arn: str, *, region: str | None = None, external_id: str | None = None
    ) -> Session:
        """A session in another account (or with another role): same settings, credentials
        from assuming ``role_arn`` with this session's base credentials."""
        update: dict[str, Any] = {"role_arn": role_arn, "external_id": external_id}
        if region:
            update["region"] = region
        # Explicit buckets and roles belong to this account; the other one has its own.
        for key in ("bucket", "kms_key_arn", *(f"{r}_role_arn" for r in _ROLES)):
            update[key] = None
        return Session(self.settings.model_copy(update=update))

    def for_account(self, name: str) -> Session:
        """The session for a named account from settings ``accounts`` (e.g. "prod")."""
        try:
            account = self.settings.accounts[name]
        except KeyError:
            known = ", ".join(sorted(self.settings.accounts)) or "none configured"
            raise ConfigError(f"No account named {name!r} in settings (known: {known})") from None
        return self.assume(account.role_arn, region=account.region, external_id=account.external_id)

    def __repr__(self) -> str:
        who = f", role={self.settings.role_arn}" if self.settings.role_arn else ""
        return f"Session(region={self.region!r}{who})"

    def client(self, service: str, region: str | None = None) -> Any:
        boto: Any = self.boto  # service name is dynamic; skip boto3-stubs overloads
        with self._client_lock:
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
            "agent_job_role_arn": s.agent_job_role_arn,
        }
        optional = ("kms_key_arn", "agent_job_role_arn")
        if all(v for k, v in explicit.items() if k not in optional):
            return Resources(**explicit)  # type: ignore[arg-type]
        outputs = self.stack_outputs(region)
        if outputs is None:
            raise BootstrapMissingError(region)
        discovered = {field: outputs.get(key) for key, field in _OUTPUT_KEYS.items()}
        merged = {k: explicit[k] or discovered[k] for k in explicit}
        missing = [k for k, v in merged.items() if not v and k not in optional]
        if missing:
            raise ConfigError(f"Bootstrap stack is missing outputs for: {', '.join(missing)}")
        return Resources(**merged)  # type: ignore[arg-type]


_default: Session | None = None


def get_session() -> Session:
    global _default
    if _default is None or _default.settings is not get_settings():
        _default = Session()
    return _default
