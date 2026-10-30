from pathlib import Path

import pytest

from skyhammer.backends.bedrock import BedrockNativeBackend
from skyhammer.dataset import Dataset
from skyhammer.errors import DatasetValidationError, UnsupportedParameterError
from skyhammer.llm import LLM
from skyhammer.session import Session


def plan(session: Session, sft_file: Path, **kw: object):  # type: ignore[no-untyped-def]
    llm = LLM("nova-micro", session=session)
    return llm.plan(
        kw.pop("technique", "sft"),
        Dataset.from_file(sft_file),  # type: ignore[arg-type]
        output_model="support-nova-v1",
        **kw,
    )


def test_build_request_sft(session: Session, sft_file: Path) -> None:
    request, estimate = plan(session, sft_file, epochs=3, learning_rate=1e-5, validation_split=0.1)
    kwargs = BedrockNativeBackend(session).build_request(
        request, "s3://b/train.jsonl", "s3://b/val.jsonl", "s3://b/out/"
    )
    assert kwargs["jobName"].startswith("support-nova-v1-")
    assert len(kwargs["jobName"]) <= 63
    assert kwargs["baseModelIdentifier"] == "amazon.nova-micro-v1:0:128k"
    assert kwargs["customizationType"] == "FINE_TUNING"
    assert kwargs["hyperParameters"] == {"epochCount": "3", "learningRate": "1e-05"}
    assert kwargs["validationDataConfig"] == {"validators": [{"s3Uri": "s3://b/val.jsonl"}]}
    assert kwargs["customModelKmsKeyId"] == session.settings.kms_key_arn
    tag_keys = {t["key"] for t in kwargs["jobTags"]}
    assert {"skyhammer:name", "skyhammer:dataset-sha", "cost-center"} <= tag_keys
    assert estimate.training_tokens > 0


def test_same_request_same_job_name(session: Session, sft_file: Path) -> None:
    a, _ = plan(session, sft_file, epochs=3)
    b, _ = plan(session, sft_file, epochs=3)
    c, _ = plan(session, sft_file, epochs=4)
    assert a.job_name == b.job_name != c.job_name


def test_lora_rank_rejected_on_bedrock(session: Session, sft_file: Path) -> None:
    request, _ = plan(session, sft_file, lora_rank=16)
    with pytest.raises(UnsupportedParameterError, match="lora_rank"):
        BedrockNativeBackend(session).build_request(request, "s3://t", None, "s3://o")


def test_distillation_request(session: Session, sft_file: Path) -> None:
    request, _ = plan(
        session, sft_file, technique="distillation", teacher_model="us.amazon.nova-premier-v1:0"
    )
    kwargs = BedrockNativeBackend(session).build_request(request, "s3://t", None, "s3://o")
    assert kwargs["customizationType"] == "DISTILLATION"
    teacher = kwargs["customizationConfig"]["distillationConfig"]["teacherModelConfig"]
    assert teacher["teacherModelIdentifier"] == "us.amazon.nova-premier-v1:0"
    assert "hyperParameters" not in kwargs


def test_invalid_dataset_raises_before_aws(session: Session, tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"messages": []}\n')
    with pytest.raises(DatasetValidationError):
        LLM("nova-micro", session=session).plan(
            "sft",
            Dataset.from_file(p),  # type: ignore[arg-type]
            output_model="x",
        )


def test_job_stage_from_status_details() -> None:
    from skyhammer.backends.bedrock import stage

    done = {"status": "Completed"}
    assert stage(
        {"validationDetails": done, "dataProcessingDetails": {"status": "InProgress"}}
    ) == ("processing data (distillation: teacher answers)")
    live = {  # from the live distillation job, 2026-10-29
        "validationDetails": done,
        "dataProcessingDetails": done,
        "trainingDetails": {"status": "InProgress"},
    }
    assert stage(live) == "training" and stage({}) is None
