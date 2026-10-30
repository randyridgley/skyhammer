"""TR-16: continue customizing a fine-tuned model on the backend that produced it."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from botocore.validate import validate_parameters

from skyhammer import registry
from skyhammer.backends.bedrock import BedrockNativeBackend
from skyhammer.backends.sagemaker import SageMakerServerlessBackend
from skyhammer.dataset import Dataset
from skyhammer.errors import SkyhammerError
from skyhammer.jobs import ModelRef
from skyhammer.llm import LLM
from skyhammer.session import Session

CUSTOM = "arn:aws:bedrock:us-east-1:123456789012:custom-model/amazon.nova-micro-v1:0:128k/abc"
PACKAGE = "arn:aws:sagemaker:us-east-1:123456789012:model-package/support-qwen/1"
MPG = "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/m"


def valid(service: str, op: str, kwargs: dict[str, Any]) -> None:
    model = boto3.client(service, region_name="us-east-1").meta.service_model
    validate_parameters(kwargs, model.operation_model(op).input_shape)


def test_bedrock_continues_from_the_custom_model(session: Session, sft_file: Path) -> None:
    ft = LLM.from_model_ref(
        ModelRef("support-nova", CUSTOM, "bedrock", "nova-micro", "us-east-1"), session=session
    )
    request, _ = ft.plan("sft", Dataset.from_file(sft_file), output_model="support-nova-v2")
    assert request.source_model == CUSTOM
    kw = BedrockNativeBackend(session).build_request(request, "s3://t", None, "s3://o/")
    assert kw["baseModelIdentifier"] == CUSTOM
    valid("bedrock", "CreateModelCustomizationJob", kw)
    # A continued job is a different job from training the base on the same data.
    base, _ = LLM("nova-micro", session=session).plan(
        "sft", Dataset.from_file(sft_file), output_model="support-nova-v2"
    )
    assert base.job_name != request.job_name and base.source_model is None


def test_sagemaker_continues_from_the_registered_package(
    session: Session, sft_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        registry, "find_job", lambda s, name: SimpleNamespace(job_name="support-qwen-1234")
    )
    sm = SimpleNamespace(describe_training_job=lambda **kw: {"OutputModelPackageArn": PACKAGE})
    real_client = session.client
    monkeypatch.setattr(
        session,
        "client",
        lambda svc, region=None: sm if svc == "sagemaker" else real_client(svc, region),
    )
    ft = LLM.from_model_ref(
        ModelRef(
            "support-qwen", "arn:imported", "sagemaker", "qwen3-8b", "us-east-1", api="openai_chat"
        ),
        session=session,
    )
    request, _ = ft.plan("sft", Dataset.from_file(sft_file), output_model="support-qwen-v2")
    assert request.source_model == PACKAGE and request.route.backend.value == "sagemaker"
    monkeypatch.setattr(session, "client", real_client)
    kw = SageMakerServerlessBackend(session).build_request(
        request, "s3://t", None, "s3://o/", model_package_group_arn=MPG
    )
    assert kw["ModelPackageConfig"] == {
        "ModelPackageGroupArn": MPG,
        "SourceModelPackageArn": PACKAGE,
    }
    valid("sagemaker", "CreateTrainingJob", kw)


def test_continuation_stays_on_its_backend(session: Session, sft_file: Path) -> None:
    ft = LLM.from_model_ref(
        ModelRef("support-nova", CUSTOM, "bedrock", "nova-micro", "us-east-1"), session=session
    )
    with pytest.raises(SkyhammerError, match="continued customization stays there"):
        ft.plan("sft", Dataset.from_file(sft_file), output_model="v2", backend="sagemaker")
    imported = LLM.from_model_ref(
        ModelRef(
            "x",
            "arn:aws:bedrock:us-east-1:1:imported-model/x",
            "bedrock",
            "nova-micro",
            "us-east-1",
        ),
        session=session,
    )
    with pytest.raises(SkyhammerError, match="isn't a Bedrock custom model"):
        imported.plan("sft", Dataset.from_file(sft_file), output_model="v2")
