"""Validate built requests against botocore's service models (no network)."""

from pathlib import Path
from typing import Any

from botocore.stub import ANY, Stubber

from skyhammer.backends.bedrock import BedrockNativeBackend
from skyhammer.backends.sagemaker import SageMakerServerlessBackend
from skyhammer.dataset import Dataset
from skyhammer.llm import LLM
from skyhammer.session import Session

MPG = "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/m"


def _call_stubbed(client: Any, op: str, kwargs: dict[str, Any], response: dict[str, Any]) -> None:
    with Stubber(client) as stub:
        stub.add_response(op, response, {k: ANY for k in kwargs})
        getattr(client, op)(**kwargs)  # raises ParamValidationError on schema mismatch


def test_bedrock_customization_request_is_schema_valid(session: Session, sft_file: Path) -> None:
    request, _ = LLM("nova-micro", session=session).plan(
        "sft", Dataset.from_file(sft_file), output_model="m", epochs=2, validation_split=0.1
    )
    backend = BedrockNativeBackend(session)
    kwargs = backend.build_request(request, "s3://b/t.jsonl", "s3://b/v.jsonl", "s3://b/o/")
    _call_stubbed(
        backend.bedrock,
        "create_model_customization_job",
        kwargs,
        {"jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-customization-job/x"},
    )


def test_sagemaker_serverless_request_is_schema_valid(session: Session, sft_file: Path) -> None:
    request, _ = LLM("qwen3-8b", session=session).plan(
        "sft", Dataset.from_file(sft_file), output_model="m", epochs=2, validation_split=0.1
    )
    backend = SageMakerServerlessBackend(session)
    kwargs = backend.build_request(
        request, "s3://b/t.jsonl", "s3://b/v.jsonl", "s3://b/o/", model_package_group_arn=MPG
    )
    _call_stubbed(
        backend.sagemaker,
        "create_training_job",
        kwargs,
        {"TrainingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:training-job/m"},
    )
