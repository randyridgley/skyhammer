import json
from pathlib import Path
from typing import Any

import boto3
import pytest
from moto import mock_aws
from typer.testing import CliRunner

from skyhammer import registry
from skyhammer.backends.base import DeploymentRef, DeployOptions
from skyhammer.backends.bedrock import BedrockNativeBackend, format_metrics_csv
from skyhammer.backends.sagemaker import format_log_event
from skyhammer.cli.main import app
from skyhammer.errors import SkyhammerError
from skyhammer.inference import ChatCompletion, Choice, Message, Usage
from skyhammer.jobs import FineTuningJob, JobRecord, JobState, JobStatus, ModelRef
from skyhammer.llm import LLM
from skyhammer.session import Session


def record(name: str = "support-nova-v1", job: str = "support-nova-v1-abcd1234") -> JobRecord:
    return JobRecord(
        job_name=job,
        job_id=f"arn:job/{job}",
        backend="bedrock",
        region="us-east-1",
        technique="sft",
        base_model="nova-micro",
        backend_model_id="amazon.nova-micro-v1:0:128k",
        serving="bedrock",
        output_model=name,
        training_type="lora",
        fingerprint="f" * 64,
        created_at=1.0,
    )


class FakeBackend:
    name = "bedrock"
    region = "us-east-1"

    def __init__(self, session: Session) -> None:
        self.session = session
        self.deployed: list[DeployOptions] = []
        self.deleted: list[str] = []

    def poll(self, rec: JobRecord) -> JobStatus:
        return JobStatus(JobState.COMPLETED, output_model_arn=f"arn:model/{rec.output_model}")

    def finalize(self, rec: JobRecord) -> ModelRef:
        return ModelRef(
            rec.output_model, f"arn:model/{rec.output_model}", "bedrock", rec.base_model, rec.region
        )

    def deploy(self, model: ModelRef, opts: DeployOptions) -> DeploymentRef:
        self.deployed.append(opts)
        return DeploymentRef(opts.name, f"arn:deployment/{opts.name}", "Active")

    def describe_deployment(self, arn: str) -> DeploymentRef:
        return DeploymentRef(arn.rsplit("/", 1)[-1], arn, "Active")

    def delete_deployment(self, arn: str) -> None:
        self.deleted.append(arn)

    def delete_model(self, model: ModelRef) -> None:
        self.deleted.append(model.model_arn)


@pytest.fixture
def s3_session(session: Session, monkeypatch: pytest.MonkeyPatch) -> Any:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(
            Bucket=session.settings.bucket or ""
        )
        fake = FakeBackend(session)
        monkeypatch.setattr("skyhammer.backends.get_backend", lambda *a, **k: fake)
        monkeypatch.setattr("skyhammer.llm.get_backend", lambda *a, **k: fake)
        yield session, fake


def test_reattach_by_model_or_job_name(s3_session: Any) -> None:
    session, _ = s3_session
    registry.save_job(session, record())
    registry.save_job(session, record(job="support-nova-v1-ffff0000"))  # retrain
    by_model = FineTuningJob.get("support-nova-v1", session=session)
    assert by_model.name == "support-nova-v1-ffff0000"  # latest job wins
    assert FineTuningJob.get("support-nova-v1-abcd1234", session=session).name.endswith("abcd1234")
    assert registry.load_model(session, "support-nova-v1")["jobs"] == [
        "support-nova-v1-abcd1234",
        "support-nova-v1-ffff0000",
    ]
    with pytest.raises(SkyhammerError, match="skyhammer jobs list"):
        FineTuningJob.get("nope", session=session)


def test_list_skips_deleted_models(s3_session: Any) -> None:
    session, _ = s3_session
    registry.save_job(session, record("a", "a-1"))
    registry.save_job(session, record("b", "b-1"))
    registry.update_model(session, "a", deleted=True)
    assert [j.record.output_model for j in FineTuningJob.list(session=session)] == ["b"]


def test_llm_get_reuses_recorded_deployment_and_delete_marks_manifest(s3_session: Any) -> None:
    session, fake = s3_session
    registry.save_job(session, record())
    ft = LLM.get("support-nova-v1", session=session)
    with pytest.raises(SkyhammerError, match="must be deployed"):
        _ = ft.model_id
    ft.deploy(wait=False)
    again = LLM.get("support-nova-v1", session=session)
    assert again.model_id == "arn:deployment/support-nova-v1"  # no redeploy needed
    again.delete()
    assert fake.deleted == ["arn:deployment/support-nova-v1", "arn:model/support-nova-v1"]
    assert registry.load_model(session, "support-nova-v1")["deleted"] is True
    with pytest.raises(SkyhammerError, match="was deleted"):
        LLM.get("support-nova-v1", session=session)


class FakeBedrockClient:
    """Mirrors real behavior: Get by unknown name -> ValidationException; List by name works."""

    def __init__(self, existing: dict[str, str] | None = None) -> None:
        self.existing = existing or {}
        self.created: list[str] = []

    def get_custom_model_deployment(self, customModelDeploymentIdentifier: str) -> Any:
        from botocore.exceptions import ClientError

        raise ClientError({"Error": {"Code": "ValidationException"}}, "Get")

    def list_custom_model_deployments(self, nameContains: str) -> Any:
        return {
            "modelDeploymentSummaries": [
                {
                    "customModelDeploymentName": n,
                    "customModelDeploymentArn": f"arn:{n}",
                    "status": st,
                }
                for n, st in self.existing.items()
                if nameContains in n
            ]
        }

    def create_custom_model_deployment(self, **kw: Any) -> Any:
        self.created.append(kw["modelDeploymentName"])
        return {"customModelDeploymentArn": f"arn:{kw['modelDeploymentName']}"}


@pytest.mark.parametrize(
    "existing, creates",
    [({}, True), ({"m": "Active"}, False), ({"m": "Creating"}, False), ({"m": "Failed"}, True)],
)
def test_bedrock_deploy_is_idempotent(
    session: Session, existing: dict[str, str], creates: bool
) -> None:
    backend = BedrockNativeBackend(session)
    backend.bedrock = FakeBedrockClient(existing)
    ref = ModelRef("m", "arn:model/m", "bedrock", "nova-micro", "us-east-1")
    out = backend.deploy(ref, DeployOptions(name="m"))
    assert out.arn == "arn:m"
    assert backend.bedrock.created == (["m"] if creates else [])


def test_log_formatters_on_real_output_samples() -> None:
    csv_body = "step_number,epoch_number,training_loss\n1,0,3.9625627994537354\n6,0,0.5392840504\n"
    assert format_metrics_csv("step_wise_training_metrics.csv", csv_body) == [
        "[step_wise_training_metrics] step 1 epoch 0 training_loss=3.9626",
        "[step_wise_training_metrics] step 6 epoch 0 training_loss=0.5393",
    ]
    event = (
        '{"schema_version": "1.0", "timestamp": "2026-10-29T19:42:37Z", '
        '"category": "TRAINING_COMPLETE", "message": "Training completed"}'
    )
    assert format_log_event(event) == "19:42:37 TRAINING_COMPLETE Training completed"
    assert format_log_event("plain text line\n") == "plain text line"


SPEC = """
apiVersion: skyhammer/v1
kind: FineTuningJob
name: {name}
base_model: {base}
technique: sft
dataset: {{path: train.jsonl}}
deploy: {{type: {deploy}}}
{extra}
"""


def write_spec(
    tmp_path: Path,
    sft_file: Path,
    name: str = "support-nova-v1",
    base: str = "nova-micro",
    deploy: str = "on_demand",
    extra: str = "",
) -> Path:
    (tmp_path / "train.jsonl").write_text(sft_file.read_text())
    p = tmp_path / "job.yaml"
    p.write_text(SPEC.format(name=name, base=base, deploy=deploy, extra=extra))
    return p


def test_apply_plan_notes_sagemaker_endpoint_billing(tmp_path: Path, sft_file: Path) -> None:
    spec = write_spec(tmp_path, sft_file, name="g", base="gemma-4-31b")
    result = CliRunner().invoke(app, ["apply", str(spec), "--plan"])
    assert result.exit_code == 0, result.output
    assert "instance-hour" in " ".join(result.output.split())


class FakeChat:
    class completions:
        @staticmethod
        def create(**kw: Any) -> ChatCompletion:
            return ChatCompletion(
                "id", "m", [Choice(0, Message("assistant", "a1"), "stop")], Usage(1, 1, 2)
            )


class FakeLLM:
    def __init__(self) -> None:
        self.deployed: dict[str, Any] | None = None
        self.manifest: dict[str, Any] = {}
        self.chat = FakeChat()
        self.model_id = "arn:deployment/x"

    def deploy(self, **kw: Any) -> "FakeLLM":
        self.deployed = kw
        return self

    def _record_in_manifest(self, **changes: Any) -> None:
        self.manifest.update(changes)


def test_apply_trains_deploys_and_evaluates(
    tmp_path: Path, sft_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
        for i in (1, 2)
    ]
    (tmp_path / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    spec = write_spec(
        tmp_path, sft_file, extra="eval: {dataset: eval.jsonl, graders: [{builtin: exact_match}]}"
    )
    ft = FakeLLM()
    calls: list[str] = []

    class FakeJob:
        name = "support-nova-v1-1234"
        record = record()
        output_llm = ft

        def wait(self, **kw: Any) -> None:
            calls.append("wait")
            assert kw.get("progress") is True  # the CLI shows stage changes and the loss

    monkeypatch.setattr(
        LLM, "create_job", lambda self, *a, **k: calls.append("create") or FakeJob()
    )
    result = CliRunner().invoke(app, ["apply", str(spec), "--yes"])
    assert result.exit_code == 0, result.output
    assert calls == ["create", "wait"]
    assert ft.deployed == {"name": None, "scaling": None, "instance_type": None}
    assert ft.manifest["eval"]["means"] == {"exact-match": 0.5}
    assert "exact-match" in result.output and "Done." in result.output


def test_apply_no_wait_returns_after_submit(
    tmp_path: Path, sft_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = write_spec(tmp_path, sft_file)

    class FakeJob:
        name = "support-nova-v1-1234"
        record = record()

        def wait(self, **kw: Any) -> None:
            raise AssertionError("should not wait")

    monkeypatch.setattr(LLM, "create_job", lambda self, *a, **k: FakeJob())
    result = CliRunner().invoke(app, ["apply", str(spec), "--yes", "--no-wait"])
    assert result.exit_code == 0, result.output
    assert "skyhammer jobs wait support-nova-v1" in " ".join(result.output.split())
