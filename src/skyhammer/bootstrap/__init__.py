"""Create/update/destroy the per account+region bootstrap CloudFormation stack."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from botocore.exceptions import ClientError

from skyhammer import tags as tagging
from skyhammer.config import Settings
from skyhammer.session import Session

# Regions the catalog routes training to. Bootstrapping all of them up front means
# cross-region routes work without a second setup step (cost: one KMS key per region).
CUSTOMIZATION_REGIONS = ("us-east-1", "us-west-2")


def default_regions(settings: Settings) -> list[str]:
    regions = [settings.region]
    if settings.cross_region:
        regions += [r for r in CUSTOMIZATION_REGIONS if r != settings.region]
    if settings.allowed_regions:
        regions = [r for r in regions if r in settings.allowed_regions]
    return regions


def session_for(settings: Settings, region: str) -> Session:
    return Session(settings.model_copy(update={"region": region}))


def template_body() -> str:
    return files("skyhammer.bootstrap").joinpath("template.yaml").read_text()


def deploy(
    session: Session, *, kms_key_arn: str | None = None, wait: bool = True
) -> dict[str, str]:
    """Create or update the stack; returns its outputs."""
    cfn = session.client("cloudformation")
    name = session.settings.bootstrap_stack
    kwargs: dict[str, Any] = {
        "StackName": name,
        "TemplateBody": template_body(),
        "Parameters": [{"ParameterKey": "ExistingKmsKeyArn", "ParameterValue": kms_key_arn or ""}],
        "Capabilities": ["CAPABILITY_IAM"],
        "Tags": tagging.as_aws(
            tagging.build_tags({"component": "bootstrap"}, session.settings.tags)
        ),
    }
    exists = session.stack_outputs() is not None
    try:
        if exists:
            cfn.update_stack(**kwargs)
            waiter = "stack_update_complete"
        else:
            cfn.create_stack(**kwargs)
            waiter = "stack_create_complete"
    except ClientError as e:
        if "No updates are to be performed" not in str(e):
            raise
        waiter = None
    if wait and waiter:
        cfn.get_waiter(waiter).wait(StackName=name)
    return session.stack_outputs() or {}


def destroy(session: Session, *, wait: bool = True) -> None:
    """Delete the stack. The bucket and KMS key are retained (DeletionPolicy: Retain)."""
    cfn = session.client("cloudformation")
    cfn.delete_stack(StackName=session.settings.bootstrap_stack)
    if wait:
        cfn.get_waiter("stack_delete_complete").wait(StackName=session.settings.bootstrap_stack)
