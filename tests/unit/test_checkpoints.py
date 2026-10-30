"""TR-15: list checkpoints; callable ones (bedrock-mantle) become chattable models."""

import io
import json
from typing import Any

import pytest
from botocore.exceptions import ClientError

from skyhammer.backends.mantle import BedrockMantleBackend
from skyhammer.backends.sagemaker import SageMakerServerlessBackend
from skyhammer.errors import SkyhammerError
from skyhammer.jobs import JobRecord
from skyhammer.session import Session


def record(backend: str, job_name: str = "sums-x") -> JobRecord:
    return JobRecord(
        job_name=job_name,
        job_id="ftjob-1",
        backend=backend,
        region="us-west-2",
        technique="rft",
        base_model="qwen3-32b",
        backend_model_id="qwen.qwen3-32b",
        serving="bedrock",
        output_model="sums",
        training_type="lora",
        fingerprint="f",
    )


class FakeMantle:
    def request(self, method: str, path: str, **_: Any) -> dict[str, Any]:
        assert path.startswith("/v1/fine_tuning/jobs/ftjob-1/checkpoints")
        return {
            "data": [
                {
                    "step_number": 20,
                    "fine_tuned_model_checkpoint": "ft:q:sums:ckpt-step-20",
                    "metrics": {"reward": 0.8},
                },
                {
                    "step_number": 10,
                    "fine_tuned_model_checkpoint": "ft:q:sums:ckpt-step-10",
                    "metrics": {"reward": 0.5, "x": None},
                },
            ]
        }


def test_mantle_checkpoints_are_callable_models(session: Session) -> None:
    backend = BedrockMantleBackend(session, "us-west-2")
    backend.client = FakeMantle()  # type: ignore[assignment]
    first, second = backend.checkpoints(record("mantle"))
    assert (first.step, first.metrics, first.model_id) == (
        10,
        {"reward": 0.5},
        "ft:q:sums:ckpt-step-10",
    )
    llm = second.llm(session)
    assert llm.model_ref is not None and llm.model_id == "ft:q:sums:ckpt-step-20"
    assert llm.model_ref.api == "mantle" and llm.model_ref.region == "us-west-2"


class FakeS3:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def get_paginator(self, op: str) -> Any:
        prefix = self.prefix

        class P:
            def paginate(self, **kw: Any) -> list[dict[str, Any]]:
                assert kw["Delimiter"] == "/"
                return [
                    {
                        "CommonPrefixes": [
                            {"Prefix": f"{prefix}checkpoints/step_0010/"},
                            {"Prefix": f"{prefix}checkpoints/step_0005/"},
                        ]
                    }
                ]

        return P()

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if Key == f"{self.prefix}metrics.jsonl":  # evaluation metrics not written yet
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        assert Key == f"{self.prefix}train_metrics.jsonl"
        lines = [
            {
                "train/gpt_loss": "2.2",
                "train/lr": "0.0",
                "train/global_step": "5",
            },  # recorded live shape
            {"train/gpt_loss": "1.1", "train/global_step": "10"},
        ]
        return {"Body": io.BytesIO("\n".join(json.dumps(x) for x in lines).encode())}


def test_sagemaker_checkpoints_are_training_state(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = "sums-x"
    prefix = f"jobs/{job}/output/{job}/output/model/"
    real = session.client
    monkeypatch.setattr(
        session,
        "client",
        lambda svc, region=None: FakeS3(prefix) if svc == "s3" else real(svc, region),
    )
    ckpts = SageMakerServerlessBackend(session).checkpoints(record("sagemaker", job))
    assert [c.step for c in ckpts] == [5, 10]
    assert ckpts[0].metrics == {"gpt_loss": 2.2, "lr": 0.0}
    assert ckpts[1].uri == f"s3://{session.settings.bucket}/{prefix}checkpoints/step_0010/"
    with pytest.raises(SkyhammerError, match="training state"):
        ckpts[0].llm()
