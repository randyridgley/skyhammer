"""SE-5: VPC-only training, import and endpoints, configured per region."""

from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.validate import validate_parameters
from pydantic import ValidationError

from skyhammer.backends.bedrock import BedrockNativeBackend
from skyhammer.backends.sagemaker import SageMakerServerlessBackend
from skyhammer.config import Settings, Vpc
from skyhammer.dataset import Dataset
from skyhammer.llm import LLM
from skyhammer.serving import EndpointNames, model_request
from skyhammer.session import Session

VPC = {"us-east-1": {"subnet_ids": ["subnet-a", "subnet-b"], "security_group_ids": ["sg-1"]}}
MPG = "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/m"


def valid(service: str, op: str, kwargs: dict[str, Any]) -> None:
    model = boto3.client(service, region_name="us-east-1").meta.service_model
    validate_parameters(kwargs, model.operation_model(op).input_shape)


@pytest.fixture
def vpc_session(settings: Settings) -> Session:
    return Session(settings.model_copy(update={"vpc": {r: Vpc(**v) for r, v in VPC.items()}}))


def test_bedrock_customization_runs_in_the_vpc(vpc_session: Session, sft_file: Path) -> None:
    request, _ = LLM("nova-micro", session=vpc_session).plan(
        "sft", Dataset.from_file(sft_file), output_model="private-nova"
    )
    kw = BedrockNativeBackend(vpc_session).build_request(request, "s3://t", None, "s3://o/")
    assert kw["vpcConfig"] == {"subnetIds": ["subnet-a", "subnet-b"], "securityGroupIds": ["sg-1"]}
    valid("bedrock", "CreateModelCustomizationJob", kw)


def test_sagemaker_training_runs_in_the_vpc(vpc_session: Session, sft_file: Path) -> None:
    request, _ = LLM("qwen3-8b", session=vpc_session).plan(
        "sft", Dataset.from_file(sft_file), output_model="private-qwen"
    )
    kw = SageMakerServerlessBackend(vpc_session).build_request(
        request, "s3://t", None, "s3://o/", model_package_group_arn=MPG
    )
    assert kw["VpcConfig"] == {"SecurityGroupIds": ["sg-1"], "Subnets": ["subnet-a", "subnet-b"]}
    valid("sagemaker", "CreateTrainingJob", kw)


def test_endpoint_models_run_in_the_vpc() -> None:
    req = model_request(
        EndpointNames.for_deployment("private"),
        image_uri="img",
        environment={},
        model_dir="s3://b/m/",
        role_arn="arn:aws:iam::1:role/r",
        tags=[],
        vpc=Vpc(**VPC["us-east-1"]),
    )
    assert req["VpcConfig"]["Subnets"] == ["subnet-a", "subnet-b"]
    valid("sagemaker", "CreateModel", req)


def test_vpc_is_per_region(vpc_session: Session) -> None:
    assert vpc_session.settings.vpc_for("us-west-2") is None
    with pytest.raises(ValidationError):
        Vpc(subnet_ids=[], security_group_ids=["sg-1"])


def test_bootstrap_scopes_the_vpc_policy_to_configured_subnets(
    vpc_session: Session, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from skyhammer.bootstrap import _vpc_parameters

    sts = SimpleNamespace(get_caller_identity=lambda: {"Account": "123456789012"})
    monkeypatch.setattr(vpc_session, "client", lambda svc, region=None: sts)
    params = {p["ParameterKey"]: p["ParameterValue"] for p in _vpc_parameters(vpc_session)}
    assert params["VpcSubnetArns"] == (
        "arn:aws:ec2:us-east-1:123456789012:subnet/subnet-a,"
        "arn:aws:ec2:us-east-1:123456789012:subnet/subnet-b"
    )
    assert (
        params["VpcSecurityGroupArns"] == "arn:aws:ec2:us-east-1:123456789012:security-group/sg-1"
    )
    # No VPC configured: empty parameters, so the template creates no EC2 permissions.
    none = {p["ParameterKey"]: p["ParameterValue"] for p in _vpc_parameters(Session(settings))}
    assert none == {"VpcSubnetArns": "", "VpcSecurityGroupArns": ""}
