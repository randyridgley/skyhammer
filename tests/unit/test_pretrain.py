import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from skyhammer import pretrain, pricing
from skyhammer.backends.sagemaker import SageMakerServerlessBackend
from skyhammer.catalog import TrainingType
from skyhammer.dataset import Dataset
from skyhammer.errors import (
    CostLimitExceededError,
    SkyhammerError,
    UnsupportedParameterError,
    UnsupportedRouteError,
)
from skyhammer.llm import LLM
from skyhammer.session import Session

# Live Price List rows (2026-11-01, us-east-1): SageMaker training, per instance-hour.
TRAIN_PRICES = {"ml.g5.2xlarge": 1.515, "ml.g6e.2xlarge": 2.8, "ml.g5.12xlarge": 7.09}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    p = tmp_path / "docs.jsonl"
    p.write_text("\n".join(json.dumps({"text": f"Document {i}. " * 50}) for i in range(20)))
    return p


@pytest.fixture
def prices(monkeypatch: pytest.MonkeyPatch) -> None:
    def lookup(session: Any, service: str, usage: str) -> dict[str, Any] | None:
        assert service == "AmazonSageMaker"
        usd = TRAIN_PRICES.get(usage.removeprefix("USE1-Train:"))
        return {"unit": "Hrs", "usd": usd} if usd else None

    monkeypatch.setattr(pricing, "_lookup", lookup)


def build(session: Session, corpus: Path, model: str = "qwen3-0.6b", **kw: Any) -> dict[str, Any]:
    request, _ = LLM(model, session=session).plan(
        "cpt", Dataset.from_file(corpus), output_model="domain-qwen-v1", **kw
    )
    backend = SageMakerServerlessBackend(session)
    return pretrain.build_request(
        session,
        backend.region,
        request,
        backend.hub.model(request.route.capability.backend_model_id),
        training_uri="s3://b/train/",
        validation_uri=None,
        output_uri="s3://b/out/",
        script_uri="s3://b/code/sourcedir.tar.gz",
    )


def test_cpt_request_is_a_capped_training_job(session: Session, corpus: Path, prices: None) -> None:
    kw = build(session, corpus, epochs=2)
    assert "ServerlessJobConfig" not in kw and "ModelPackageConfig" not in kw
    assert kw["AlgorithmSpecification"]["TrainingImage"].startswith(
        "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:"
    )
    # Base weights come from the open JumpStart artifacts, as a "model" channel.
    channels = {
        c["ChannelName"]: c["DataSource"]["S3DataSource"]["S3Uri"] for c in kw["InputDataConfig"]
    }
    assert channels["model"].startswith("s3://jumpstart-cache-prod-us-east-1/")
    assert channels["train"] == "s3://b/train/"
    hp = kw["HyperParameters"]
    assert json.loads(hp["sagemaker_program"]) == "cpt_train.py"
    assert json.loads(hp["sagemaker_submit_directory"]) == "s3://b/code/sourcedir.tar.gz"
    assert (hp["epochs"], hp["training_type"], hp["learning_rate"]) == ("2", '"lora"', "0.0001")
    assert kw["ResourceConfig"]["InstanceType"] == "ml.g5.2xlarge"
    assert kw["OutputDataConfig"]["CompressionType"] == "NONE"  # Bedrock imports it as is
    assert kw["StoppingCondition"] == {"MaxRuntimeInSeconds": 4 * 3600}
    assert {"Key": "skyhammer:technique", "Value": "cpt"} in kw["Tags"]


def test_cpt_is_priced_per_instance_hour_and_guarded(
    session: Session, corpus: Path, prices: None
) -> None:
    llm = LLM("qwen3-8b", session=session)
    _, est = llm.plan("cpt", Dataset.from_file(corpus), output_model="m", max_hours=2)
    assert est.usd_per_hour == 2.8  # 8B LoRA fits one L40S (ml.g6e.2xlarge)
    assert est.usd == pytest.approx(5.6)
    with pytest.raises(CostLimitExceededError):
        llm.create_cpt_job(Dataset.from_file(corpus), output_model="m", max_cost_usd=1)


def test_instance_sizing() -> None:
    assert pretrain.instance_for(0.6, TrainingType.LORA) == "ml.g5.2xlarge"
    assert pretrain.instance_for(8, TrainingType.LORA) == "ml.g6e.2xlarge"
    assert pretrain.instance_for(32, TrainingType.LORA) == "ml.g6e.12xlarge"
    assert pretrain.instance_for(1.7, TrainingType.FULL) == "ml.g6e.2xlarge"
    with pytest.raises(UnsupportedParameterError, match="GPU memory"):
        pretrain.instance_for(70, TrainingType.FULL)


def test_full_cpt_and_overrides(session: Session, corpus: Path, prices: None) -> None:
    kw = build(
        session,
        corpus,
        training_type="full",
        hyperparameters_extra={"instance_type": "ml.g5.12xlarge", "max_seq_length": 4096},
    )
    hp = kw["HyperParameters"]
    assert (hp["training_type"], hp["learning_rate"], hp["max_seq_length"]) == (
        '"full"',
        "1e-05",
        "4096",
    )
    assert "instance_type" not in hp
    assert kw["ResourceConfig"]["InstanceType"] == "ml.g5.12xlarge"


def test_bad_hyperparameters_fail_in_plan(session: Session, corpus: Path) -> None:
    llm = LLM("qwen3-0.6b", session=session)
    ds = Dataset.from_file(corpus)
    with pytest.raises(UnsupportedParameterError, match="warmup_ratio"):
        llm.plan("cpt", ds, output_model="m", warmup_steps=10)
    with pytest.raises(UnsupportedParameterError, match="merge_weights"):
        llm.plan("cpt", ds, output_model="m", hyperparameters_extra={"merge_weights": False})


def test_cpt_routes(session: Session, corpus: Path) -> None:
    ds = Dataset.from_file(corpus)
    with pytest.raises(UnsupportedRouteError, match="HyperPod") as e:
        LLM("nova-micro", session=session).plan("cpt", ds, output_model="m")
    assert "qwen3-0.6b" in e.value.alternatives
    with pytest.raises(UnsupportedRouteError):  # MXFP4 weights; no CPT capability
        LLM("gpt-oss-20b", session=session).plan("cpt", ds, output_model="m")


def test_cpt_needs_text_rows(session: Session, tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"text": ""}) + "\n" + json.dumps({"text": "ok"}))
    with pytest.raises(SkyhammerError):
        LLM("qwen3-0.6b", session=session).plan("cpt", Dataset.from_file(p), output_model="m")


def test_long_documents_are_allowed(session: Session, tmp_path: Path) -> None:
    p = tmp_path / "book.jsonl"
    p.write_text(json.dumps({"text": "word " * 60_000}))  # far past any SFT recipe length
    request, _ = LLM("qwen3-0.6b", session=session).plan(
        "cpt", Dataset.from_file(p), output_model="m"
    )
    assert request.dataset.report and request.dataset.report.ok


def test_script_bundle_is_reproducible() -> None:
    data = pretrain.script_bundle()
    assert data == pretrain.script_bundle()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        assert tar.getnames() == ["cpt_train.py"]
        source = tar.extractfile("cpt_train.py")
        assert source is not None and b"merge_and_unload" in source.read()
    script = Path(pretrain.__file__).parent / "trainers" / "cpt_train.py"
    compile(script.read_text(), str(script), "exec")


def test_submit_uploads_script_and_creates_job(
    session: Session, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}

    class FakeS3:
        def put_object(self, **kw: Any) -> None:
            calls["put"] = kw

    class FakeSageMaker:
        def create_training_job(self, **kw: Any) -> dict[str, str]:
            calls["create"] = kw
            return {
                "TrainingJobArn": "arn:aws:sagemaker:us-east-1:1:training-job/"
                + kw["TrainingJobName"]
            }

    request, _ = LLM("qwen3-0.6b", session=session).plan(
        "cpt", Dataset.from_file(corpus), output_model="domain-qwen-v1"
    )
    backend = SageMakerServerlessBackend(session)
    backend.sagemaker = FakeSageMaker()
    monkeypatch.setattr(session, "client", lambda name, region=None: FakeS3())
    monkeypatch.setattr(type(request.dataset), "upload", lambda self, b, t: "s3://b/train/")
    handle = backend.submit(request)
    assert handle.name == request.job_name
    assert calls["put"]["Key"].startswith("code/cpt/")
    assert calls["put"]["Body"] == pretrain.script_bundle()
    submit_dir = json.loads(calls["create"]["HyperParameters"]["sagemaker_submit_directory"])
    assert submit_dir.endswith(calls["put"]["Key"])
