"""SV-14: AWQ quantization through SageMaker inference optimization jobs."""

from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.validate import validate_parameters

from skyhammer import LLM, optimization
from skyhammer.errors import SkyhammerError, UnsupportedParameterError
from skyhammer.jobs import ModelRef
from skyhammer.session import Session


class FakeSageMaker:
    def __init__(self) -> None:
        self._real = boto3.client("sagemaker", region_name="us-east-1")
        self.meta = self._real.meta
        self.created: dict[str, Any] = {}
        self.status = "INPROGRESS"

    def create_optimization_job(self, **kw: Any) -> dict[str, str]:
        op = self.meta.service_model.operation_model("CreateOptimizationJob")
        validate_parameters(kw, op.input_shape)
        self.created = kw
        return {"OptimizationJobArn": "arn:aws:sagemaker:us-east-1:1:optimization-job/x"}

    def describe_optimization_job(self, OptimizationJobName: str) -> dict[str, Any]:
        if not self.created:
            raise ClientError({"Error": {"Code": "ResourceNotFound", "Message": "no"}}, "D")
        return {
            "OptimizationJobArn": "arn:aws:sagemaker:us-east-1:1:optimization-job/x",
            "OptimizationJobStatus": self.status,
            "OptimizationOutput": {"RecommendedInferenceImage": "djl-inference:lmi"},
        }


class FakeS3:
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def get_paginator(self, name: str) -> Any:
        keys = self.keys

        class P:
            def paginate(self, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
                return [{"Contents": [{"Key": k} for k in keys if k.startswith(Prefix)]}]

        return P()


@pytest.fixture
def fake(session: Session, monkeypatch: pytest.MonkeyPatch) -> FakeSageMaker:
    sm = FakeSageMaker()
    real = session.client

    def client(svc: str, region: str | None = None) -> Any:
        return sm if svc == "sagemaker" else real(svc, region=region)

    monkeypatch.setattr(session, "client", client)
    return sm


def test_base_model_job(session: Session, fake: FakeSageMaker) -> None:
    job = LLM("qwen3-8b", session=session).create_optimization_job(
        "qwen3-8b-awq", instance_type="ml.g6.12xlarge", max_hours=1
    )
    kw = fake.created
    assert kw["ModelSource"]["S3"]["S3Uri"].startswith("s3://jumpstart-cache-prod-us-east-1/")
    assert kw["ModelSource"]["S3"]["ModelAccessConfig"] == {"AcceptEula": True}  # not gated
    assert kw["DeploymentInstanceType"] == "ml.g6.12xlarge"
    assert kw["OptimizationConfigs"] == [
        {"ModelQuantizationConfig": {"OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}}}
    ]
    assert kw["StoppingCondition"] == {"MaxRuntimeInSeconds": 3600}
    assert kw["OutputConfig"]["KmsKeyId"] == session.settings.kms_key_arn
    assert kw["OutputConfig"]["S3OutputLocation"] == job.output_uri
    assert {"Key": "skyhammer:name", "Value": "qwen3-8b-awq"} in kw["Tags"]
    assert job.status().state == "running"
    # Same inputs reattach instead of starting a second (billed) job.
    fake.created["marker"] = True
    again = LLM("qwen3-8b", session=session).create_optimization_job(
        "qwen3-8b-awq", instance_type="ml.g6.12xlarge", max_hours=1
    )
    assert again.name == job.name and fake.created.get("marker")


def test_checks_before_starting(session: Session, fake: FakeSageMaker) -> None:
    llm = LLM("qwen3-8b", session=session)
    with pytest.raises(UnsupportedParameterError, match="deploy\\(quantize='fp8'\\)"):
        llm.create_optimization_job("x-fp8", quantize="fp8")
    with pytest.raises(UnsupportedParameterError, match=r"can't target ml\.m5\.large"):
        llm.create_optimization_job("x-awq", instance_type="ml.m5.large")
    with pytest.raises(UnsupportedParameterError, match="max_hours"):
        llm.create_optimization_job("x-awq", max_hours=48)
    with pytest.raises(SkyhammerError, match="gated license"):
        LLM("llama-3.1-8b", session=session).create_optimization_job("llama-awq")
    bedrock_tune = ModelRef(
        name="support-nova",
        model_arn="arn:aws:bedrock:us-east-1:1:custom-model/x",
        backend="bedrock",
        base_alias="nova-micro",
        region="us-east-1",
    )
    with pytest.raises(SkyhammerError):
        LLM.from_model_ref(bedrock_tune, session=session).create_optimization_job("n-awq")
    assert not fake.created


def test_fine_tune_source_and_registration(
    session: Session, fake: FakeSageMaker, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = ModelRef(
        name="support-qwen",
        model_arn="s3://b/jobs/j/output/",
        backend="sagemaker",
        base_alias="qwen3-0.6b",
        region="us-east-1",
        api="openai_chat",
        serving="sagemaker",
        artifacts_uri="s3://b/jobs/j/output/",
    )
    monkeypatch.setattr(
        "skyhammer.backends.sagemaker.resolve_model_dir",
        lambda s3, uri: uri + "checkpoints/hf_merged/",
    )
    job = LLM.from_model_ref(ref, session=session).create_optimization_job("support-qwen-awq")
    assert (
        fake.created["ModelSource"]["S3"]["S3Uri"] == "s3://b/jobs/j/output/checkpoints/hf_merged/"
    )
    assert fake.created["DeploymentInstanceType"] == "ml.g7.2xlarge"  # the SFT recipe profile

    saved: list[ModelRef] = []
    monkeypatch.setattr(optimization.registry, "save_model_ref", lambda s, r, **kw: saved.append(r))
    prefix = job.output_uri.removeprefix(f"s3://{session.settings.bucket}/")
    s3 = FakeS3(
        [prefix + "x/config.json", prefix + "x/model.safetensors", prefix + "x/d/config.json"]
    )
    real = session.client
    monkeypatch.setattr(
        session, "client", lambda svc, region=None: s3 if svc == "s3" else real(svc)
    )
    with pytest.raises(SkyhammerError, match="not completed"):
        job.finalize()
    fake.status = "COMPLETED"
    out = job.finalize()
    assert out.artifacts_uri == job.output_uri + "x/"  # the shallowest config.json
    assert (out.name, out.serving, out.base_alias) == (
        "support-qwen-awq",
        "sagemaker",
        "qwen3-0.6b",
    )
    assert saved == [out]


def test_find_model_dir_needs_a_config() -> None:
    with pytest.raises(SkyhammerError, match="No optimized model"):
        optimization.find_model_dir(FakeS3(["out/readme.md"]), "s3://b/out/")
    assert optimization.find_model_dir(FakeS3(["out/config.json"]), "s3://b/out/") == "s3://b/out/"


def test_iam_policy() -> None:
    from skyhammer.iam import optimization_policy

    (stmt,) = optimization_policy("123456789012")["Statement"]
    assert "sagemaker:CreateOptimizationJob" in stmt["Action"]
    assert stmt["Resource"] == "arn:aws:sagemaker:*:123456789012:optimization-job/*"
