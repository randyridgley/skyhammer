"""TR-17: your own training code as a SageMaker training job, imported into Bedrock."""

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.validate import validate_parameters

from skyhammer import custom_training, pricing
from skyhammer.backends.sagemaker import SageMakerServerlessBackend
from skyhammer.custom_training import TrainingCode
from skyhammer.dataset import Dataset
from skyhammer.errors import SkyhammerError, UnsupportedRouteError
from skyhammer.llm import LLM
from skyhammer.session import Session

from .test_pretrain import TRAIN_PRICES


@pytest.fixture
def prices(monkeypatch: pytest.MonkeyPatch) -> None:
    def lookup(session: Any, service: str, usage: str) -> dict[str, Any] | None:
        usd = TRAIN_PRICES.get(usage.removeprefix("USE1-Train:"))
        return {"unit": "Hrs", "usd": usd} if usd else None

    monkeypatch.setattr(pricing, "_lookup", lookup)


@pytest.fixture
def trainer(tmp_path: Path) -> Path:
    d = tmp_path / "trainer"
    (d / "lib").mkdir(parents=True)
    (d / "train.py").write_text("import lib.util\nprint('training')\n")
    (d / "lib" / "util.py").write_text("X = 1\n")
    (d / "requirements.txt").write_text("trl==0.20.0\n")
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "train.cpython-312.pyc").write_bytes(b"\0")
    (d / ".git").mkdir()
    (d / ".git" / "HEAD").write_text("ref: main\n")
    return d


@pytest.fixture
def rows(tmp_path: Path) -> Dataset:
    # Any JSON objects: your code decides the format (here: prompt/answer pairs).
    return Dataset.from_records(
        [{"prompt": f"q{i}", "answer": f"a{i}", "weight": 1.0} for i in range(20)], name="pairs"
    )


def build(session: Session, request: Any, code_uri: str | None) -> dict[str, Any]:
    backend = SageMakerServerlessBackend(session)
    kwargs = custom_training.build_request(
        session,
        backend.region,
        request,
        backend.hub.model(request.route.capability.backend_model_id),
        training_uri="s3://b/train/",
        validation_uri=None,
        output_uri="s3://b/out/",
        code_uri=code_uri,
    )
    shape = boto3.client("sagemaker", region_name="us-east-1").meta.service_model
    validate_parameters(kwargs, shape.operation_model("CreateTrainingJob").input_shape)
    return kwargs


def test_script_mode_job(session: Session, trainer: Path, rows: Dataset, prices: None) -> None:
    code = TrainingCode(str(trainer), "train.py", environment={"WANDB_DISABLED": "true"})
    request, estimate = LLM("qwen3-0.6b", session=session).plan(
        "custom",
        rows,
        output_model="pairs-qwen-v1",
        code=code,
        epochs=3,
        hyperparameters_extra={"beta_kl": 0.1, "mode": "pairs"},
    )
    kw = build(session, request, "s3://b/code/custom/abc/sourcedir.tar.gz")
    hp = kw["HyperParameters"]
    assert json.loads(hp["sagemaker_program"]) == "train.py"
    assert json.loads(hp["sagemaker_submit_directory"]).endswith("sourcedir.tar.gz")
    # Everything you set reaches your script, typed (the toolkit decodes JSON).
    assert (hp["epochs"], hp["beta_kl"], hp["mode"]) == ("3", "0.1", '"pairs"')
    assert "instance_type" not in hp
    assert kw["AlgorithmSpecification"]["TrainingImage"].startswith(
        "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:"
    )
    channels = {c["ChannelName"] for c in kw["InputDataConfig"]}
    assert channels == {"model", "train"}
    assert kw["Environment"] == {"WANDB_DISABLED": "true"}
    assert kw["OutputDataConfig"]["CompressionType"] == "NONE"
    assert kw["ResourceConfig"]["InstanceType"] == "ml.g5.2xlarge"
    assert kw["StoppingCondition"] == {"MaxRuntimeInSeconds": 4 * 3600}  # always capped
    assert {"Key": "skyhammer:technique", "Value": "custom"} in kw["Tags"]
    assert estimate.usd == pytest.approx(4 * 1.515)  # the cap is the bound


def test_container_mode_job(session: Session, rows: Dataset, prices: None) -> None:
    image = "111122223333.dkr.ecr.us-east-1.amazonaws.com/my-trainer:1.4"
    request, _ = LLM("qwen3-8b", session=session).plan(
        "custom",
        rows,
        output_model="pairs-qwen8-v1",
        code=TrainingCode(image=image),
        max_hours=2,
        learning_rate=2e-5,
        hyperparameters_extra={"mode": "pairs", "instance_type": "ml.g6e.12xlarge"},
    )
    kw = build(session, request, None)
    hp = kw["HyperParameters"]
    assert "sagemaker_program" not in hp
    assert (hp["mode"], hp["learning_rate"]) == ("pairs", "2e-05")  # plain strings
    assert kw["AlgorithmSpecification"]["TrainingImage"] == image
    assert kw["ResourceConfig"]["InstanceType"] == "ml.g6e.12xlarge"
    assert kw["StoppingCondition"]["MaxRuntimeInSeconds"] == 2 * 3600


def test_code_checks(tmp_path: Path, trainer: Path) -> None:
    with pytest.raises(SkyhammerError, match="source_dir"):
        TrainingCode()
    with pytest.raises(SkyhammerError, match="not a directory"):
        TrainingCode(str(tmp_path / "missing"))
    with pytest.raises(SkyhammerError, match=r"'main\.py' not found"):
        TrainingCode(str(trainer), "main.py")


def test_bundle_is_reproducible_and_skips_caches(trainer: Path) -> None:
    code = TrainingCode(str(trainer))
    data = code.bundle()
    assert data == code.bundle()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        assert sorted(tar.getnames()) == ["lib/util.py", "requirements.txt", "train.py"]
    before = code.fingerprint()
    (trainer / "train.py").write_text("print('v2')\n")
    assert TrainingCode(str(trainer)).fingerprint() != before  # changed code, new job


def test_code_change_means_a_new_job(session: Session, trainer: Path, rows: Dataset) -> None:
    llm = LLM("qwen3-0.6b", session=session)
    first, _ = llm.plan("custom", rows, output_model="m", code=TrainingCode(str(trainer)))
    (trainer / "lib" / "util.py").write_text("X = 2\n")
    second, _ = llm.plan("custom", rows, output_model="m", code=TrainingCode(str(trainer)))
    assert first.job_name != second.job_name


def test_routes(session: Session, trainer: Path, rows: Dataset) -> None:
    code = TrainingCode(str(trainer))
    with pytest.raises(UnsupportedRouteError):  # Nova weights aren't open
        LLM("nova-micro", session=session).plan("custom", rows, output_model="m", code=code)
    with pytest.raises(SkyhammerError, match=r"code=TrainingCode\("):
        LLM("qwen3-0.6b", session=session).plan("custom", rows, output_model="m")


def test_submit_uploads_code_and_creates_job(
    session: Session, trainer: Path, rows: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}

    class FakeS3:
        def put_object(self, **kw: Any) -> None:
            calls["put"] = kw

    class FakeSageMaker:
        def create_training_job(self, **kw: Any) -> dict[str, str]:
            calls["create"] = kw
            return {"TrainingJobArn": "arn:aws:sagemaker:us-east-1:1:training-job/x"}

    code = TrainingCode(str(trainer))
    request, _ = LLM("qwen3-0.6b", session=session).plan(
        "custom", rows, output_model="pairs-qwen-v1", code=code
    )
    backend = SageMakerServerlessBackend(session)
    backend.sagemaker = FakeSageMaker()
    monkeypatch.setattr(session, "client", lambda name, region=None: FakeS3())
    uploaded: dict[str, Any] = {}

    def upload(ds: Dataset, b: Any, t: Any) -> str:
        uploaded["rows"] = ds.records()
        return "s3://b/train/"

    monkeypatch.setattr(Dataset, "upload", upload)
    backend.submit(request)
    assert calls["put"]["Key"].startswith("code/custom/")
    assert calls["put"]["Body"] == code.bundle()
    assert uploaded["rows"][0] == {"prompt": "q0", "answer": "a0", "weight": 1.0}  # unchanged
    assert backend.format_record(request.technique, {"x": 1}) == {"x": 1}
