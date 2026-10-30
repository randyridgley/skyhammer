import io
import json
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.validate import validate_parameters
from moto import mock_aws

from skyhammer.backends.sagemaker import (
    SageMakerServerlessBackend,
    import_job_name,
    resolve_model_dir,
)
from skyhammer.catalog import TrainingType
from skyhammer.dataset import Dataset
from skyhammer.errors import SkyhammerError, UnsupportedParameterError
from skyhammer.hub import parse_hub_model, sequence_tokens
from skyhammer.inference import Chat
from skyhammer.jobs import FineTuningJob, JobHandle, JobRecord, JobState
from skyhammer.llm import LLM
from skyhammer.session import Session
from tests.conftest import FIXTURES


def hub() -> Any:
    return parse_hub_model(json.loads((FIXTURES / "hub_qwen3_06b.json").read_text()))


def test_parse_real_hub_metadata() -> None:
    m = hub()
    assert not m.gated and m.license == "Apache 2.0"
    r = m.recipe("SFT", TrainingType.LORA)
    assert r.sequence_length == "4K" and r.override_params_uri
    assert r.hosting and "djl-inference" in r.hosting.image_uri
    assert m.recipe("SFT", TrainingType.FULL).training_type is TrainingType.FULL
    assert m.recipe("RLVR", TrainingType.LORA).max_tokens == 1024


def test_recipe_selection_respects_sequence_length() -> None:
    assert sequence_tokens("4K") == 4096
    with pytest.raises(UnsupportedParameterError, match="Longest SFT recipe"):
        hub().recipe("SFT", TrainingType.LORA, min_tokens=5000)


class FakeSageMaker:
    def __init__(self, status: str = "InProgress") -> None:
        self.status = status

    def describe_training_job(self, TrainingJobName: str) -> dict[str, Any]:
        return {
            "TrainingJobStatus": self.status,
            "ModelArtifacts": {"S3ModelArtifacts": "s3://b/jobs/j/output/model/"},
        }


class FakeBedrock:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.created: list[dict[str, Any]] = []

    def list_model_import_jobs(self, nameContains: str) -> dict[str, Any]:
        return {
            "modelImportJobSummaries": [
                {"jobName": n, "jobArn": f"arn:{n}"} for n in self.jobs if nameContains in n
            ]
        }

    def get_model_import_job(self, jobIdentifier: str) -> dict[str, Any]:
        return self.jobs[jobIdentifier.removeprefix("arn:")]

    def create_model_import_job(self, **kwargs: Any) -> dict[str, Any]:
        client = boto3.client("bedrock", region_name="us-east-1")
        validate_parameters(
            kwargs, client.meta.service_model.operation_model("CreateModelImportJob").input_shape
        )
        self.created.append(kwargs)
        self.jobs[kwargs["jobName"]] = {"status": "InProgress"}
        return {"jobArn": f"arn:{kwargs['jobName']}"}


def test_train_then_import_state_machine(
    session: Session, sft_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = LLM("qwen3-0.6b", session=session).plan(
        "sft", Dataset.from_file(sft_file), output_model="support-qwen-v1"
    )
    backend = SageMakerServerlessBackend(session)
    sm, br = FakeSageMaker(), FakeBedrock()
    backend.sagemaker = sm
    monkeypatch.setattr(type(backend), "_bedrock", property(lambda self: br))
    monkeypatch.setattr(
        "skyhammer.backends.sagemaker.resolve_model_dir",
        lambda s3, uri: uri + "checkpoints/hf_merged/",
    )
    record = JobRecord.from_request(
        request, JobHandle("sagemaker", request.job_name, "arn:tj", "us-east-1")
    )
    job = FineTuningJob(record, backend)

    assert job.status().state is JobState.TRAINING
    sm.status = "Completed"
    assert job.status().state is JobState.IMPORTING  # starts the import
    assert job.status().state is JobState.IMPORTING  # idempotent: no second import
    assert len(br.created) == 1
    created = br.created[0]
    assert created["jobName"] == import_job_name(request.job_name)
    assert created["modelDataSource"]["s3DataSource"]["s3Uri"].endswith("hf_merged/")
    assert created["roleArn"] == session.settings.import_role_arn
    assert created["importedModelKmsKeyId"] == session.settings.kms_key_arn

    br.jobs[created["jobName"]] = {"status": "Completed", "importedModelArn": "arn:imported"}
    assert job.wait(poll_seconds=0) is job
    ref = job.output_model
    assert (ref.model_arn, ref.api, ref.serving) == ("arn:imported", "openai_chat", "bedrock")


def test_import_failure_surfaces(
    session: Session, sft_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = LLM("qwen3-0.6b", session=session).plan(
        "sft", Dataset.from_file(sft_file), output_model="m"
    )
    backend = SageMakerServerlessBackend(session)
    backend.sagemaker = FakeSageMaker("Completed")
    br = FakeBedrock()
    br.jobs[import_job_name(request.job_name)] = {
        "status": "Failed",
        "failureMessage": "bad config.json",
    }
    monkeypatch.setattr(type(backend), "_bedrock", property(lambda self: br))
    st = backend.poll(
        JobRecord.from_request(
            request, JobHandle("sagemaker", request.job_name, "arn", "us-east-1")
        )
    )
    assert st.state is JobState.FAILED and "bad config.json" in (st.message or "")


@mock_aws
def test_resolve_model_dir() -> None:
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="skyhammer-test")
    s3.put_object(
        Bucket="skyhammer-test", Key="out/model/checkpoints/hf_merged/config.json", Body=b"{}"
    )
    assert (
        resolve_model_dir(s3, "s3://skyhammer-test/out/model")
        == "s3://skyhammer-test/out/model/checkpoints/hf_merged/"
    )
    s3.put_object(
        Bucket="skyhammer-test", Key="out2/model/checkpoints/hf/adapter_config.json", Body=b"{}"
    )
    with pytest.raises(SkyhammerError, match="only a LoRA adapter"):
        resolve_model_dir(s3, "s3://skyhammer-test/out2/model/")
    with pytest.raises(SkyhammerError, match="No Hugging Face model"):
        resolve_model_dir(s3, "s3://skyhammer-test/missing/")


class FakeRuntime:
    def __init__(self, cold_failures: int = 0) -> None:
        self.cold_failures = cold_failures
        self.bodies: list[dict[str, Any]] = []

    def invoke_model(self, **kw: Any) -> dict[str, Any]:
        if self.cold_failures:
            self.cold_failures -= 1
            raise ClientError(
                {"Error": {"Code": "ModelNotReadyException", "Message": "x"}}, "InvokeModel"
            )
        self.bodies.append(json.loads(kw["body"]))
        resp = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hi!"},
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        return {"body": io.BytesIO(json.dumps(resp).encode())}


def test_imported_model_chat_uses_openai_body_and_survives_cold_start() -> None:
    rt, sleeps = FakeRuntime(cold_failures=2), []
    chat = Chat(rt, "arn:imported", api="openai_chat", sleep=sleeps.append)
    resp = chat.completions.create(messages=[{"role": "user", "content": "hello"}], max_tokens=8)
    assert resp.choices[0].message.content == "hi!" and resp.usage.total_tokens == 5
    assert rt.bodies == [{"messages": [{"role": "user", "content": "hello"}], "max_tokens": 8}]
    assert sleeps == [5.0, 10.0]


def test_cold_start_gives_up_after_timeout() -> None:
    chat = Chat(
        FakeRuntime(cold_failures=99),
        "arn",
        api="openai_chat",
        cold_start_timeout_s=10,
        sleep=lambda s: None,
    )
    with pytest.raises(ClientError):
        chat.completions.create(messages=[{"role": "user", "content": "x"}])


def test_base_model_chat_uses_inference_profile(session: Session) -> None:
    assert LLM("nova-micro", session=session).model_id == "us.amazon.nova-micro-v1:0"
    with pytest.raises(SkyhammerError, match=r"call \.deploy\(\) for a dedicated"):
        _ = LLM("qwen3-8b", session=session).model_id
