from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from skyhammer.backends.sagemaker import SageMakerServerlessBackend
from skyhammer.dataset import Dataset
from skyhammer.errors import DatasetValidationError, SkyhammerError, UnsupportedParameterError
from skyhammer.graders import Grader
from skyhammer.llm import LLM
from skyhammer.session import Session

MPG = "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/m"
EVALUATOR = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/skyhammer/JsonDoc/g/1.0.0"


def build(
    session: Session, sft_file: Path, model: str = "qwen3-8b", technique: str = "sft", **kw: Any
) -> dict[str, Any]:
    request, _ = LLM(model, session=session).plan(
        technique, Dataset.from_file(sft_file), output_model="support-qwen-v1", **kw
    )
    return SageMakerServerlessBackend(session).build_request(
        request,
        "s3://b/train.jsonl",
        "s3://b/val.jsonl" if kw.get("validation_split") else None,
        "s3://b/out/",
        model_package_group_arn=MPG,
        evaluator_arn=EVALUATOR if technique == "rft" else None,
    )


def test_sft_lora_request(session: Session, sft_file: Path) -> None:
    kw = build(session, sft_file, epochs=3, learning_rate=1e-4, validation_split=0.1)
    assert kw["TrainingJobName"].startswith("support-qwen-v1-")
    assert kw["RoleArn"] == session.settings.sagemaker_role_arn
    assert kw["ServerlessJobConfig"] == {
        "BaseModelArn": "arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/"
        "Model/huggingface-reasoning-qwen3-8b",
        "JobType": "FineTuning",
        "CustomizationTechnique": "SFT",
        "AcceptEula": False,
        "SequenceLength": "4K",
        "Peft": "LORA",
    }
    assert [c["ChannelName"] for c in kw["InputDataConfig"]] == ["train", "validation"]
    assert kw["OutputDataConfig"]["KmsKeyId"] == session.settings.kms_key_arn
    assert kw["OutputDataConfig"]["CompressionType"] == "NONE"
    hp = kw["HyperParameters"]
    assert (hp["max_epochs"], hp["learning_rate"]) == ("3", "0.0001")
    assert (hp["lora_rank"], hp["merge_weights"]) == ("16", "true")  # recipe defaults
    assert {"Key": "skyhammer:serving", "Value": "bedrock"} in kw["Tags"]
    assert kw["ModelPackageConfig"] == {"ModelPackageGroupArn": MPG}
    assert "StoppingCondition" not in kw  # token-billed jobs aren't runtime-capped


def test_full_fine_tuning_omits_peft(session: Session, sft_file: Path) -> None:
    kw = build(session, sft_file, training_type="full")
    assert "Peft" not in kw["ServerlessJobConfig"]


def test_sdk_hyperparameters_map_to_recipe_keys(session: Session, sft_file: Path) -> None:
    hp = build(session, sft_file, lora_rank=64, batch_size=16, warmup_ratio=0.05)["HyperParameters"]
    assert (hp["lora_rank"], hp["global_batch_size"], hp["lr_warmup_steps_ratio"]) == (
        "64",
        "16",
        "0.05",
    )


def test_hyperparameters_validated_against_recipe_bounds(session: Session, sft_file: Path) -> None:
    with pytest.raises(UnsupportedParameterError, match="above maximum"):
        build(session, sft_file, learning_rate=1e-3)
    with pytest.raises(UnsupportedParameterError, match="choose one of"):
        build(session, sft_file, lora_rank=12)
    with pytest.raises(UnsupportedParameterError, match="use warmup_ratio"):
        build(session, sft_file, warmup_steps=10)
    with pytest.raises(UnsupportedParameterError, match="isn't tunable"):
        build(session, sft_file, hyperparameters_extra={"lora_r": 16})
    kw = build(session, sft_file, hyperparameters_extra={"lora_dropout": 0.1})
    assert kw["HyperParameters"]["lora_dropout"] == "0.1"


def test_unmerged_lora_rejected_for_bedrock_serving(session: Session, sft_file: Path) -> None:
    with pytest.raises(UnsupportedParameterError, match="merged weights"):
        build(session, sft_file, hyperparameters_extra={"merge_weights": False})


def test_gated_model_requires_eula(session: Session, sft_file: Path) -> None:
    with pytest.raises(SkyhammerError, match="accept_eula=True"):
        build(session, sft_file, model="llama-3.1-8b")
    kw = build(session, sft_file, model="llama-3.1-8b", accept_eula=True)
    assert kw["ServerlessJobConfig"]["AcceptEula"] is True


def test_rft_with_lambda_grader_is_rlvr(session: Session, tmp_path: Path) -> None:
    import json

    p = tmp_path / "rft.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"messages": [{"role": "user", "content": f"q{i}"}], "answer": str(i)})
            for i in range(40)
        )
    )
    arn = "arn:aws:lambda:us-east-1:123456789012:function:skyhammer-grader-em"
    kw = build(session, p, technique="rft", grader=Grader.from_lambda(arn))
    assert kw["ServerlessJobConfig"]["CustomizationTechnique"] == "RLVR"
    # RLVR takes a Hub evaluator wrapping the Lambda, never the Lambda ARN (seen live)
    assert kw["ServerlessJobConfig"]["EvaluatorArn"] == EVALUATOR


def test_not_importable_model_routes_to_sagemaker_serving(session: Session, sft_file: Path) -> None:
    request, _ = LLM("gemma-4-31b", session=session).plan(
        "sft", Dataset.from_file(sft_file), output_model="g", accept_eula=True
    )
    assert request.route.serving.value == "sagemaker"


def _rows(tmp_path: Path, n: int) -> Path:
    import json

    p = tmp_path / f"rows{n}.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ]
                }
            )
            for i in range(n)
        )
    )
    return p


def test_small_dataset_adjusts_split_so_validation_has_a_batch(
    session: Session, tmp_path: Path
) -> None:
    # 50 rows at the recipe's 0.9 split would leave 5 validation rows < batch 8 (the
    # exact failure seen on a real job); the SDK moves the split instead.
    hp = build(session, _rows(tmp_path, 50))["HyperParameters"]
    assert hp["train_val_split_ratio"] == "0.84"


def test_too_small_dataset_fails_before_submit(session: Session, tmp_path: Path) -> None:
    with pytest.raises(UnsupportedParameterError, match="at least 8 training and 8 validation"):
        build(session, _rows(tmp_path, 12))
    with pytest.raises(UnsupportedParameterError, match="validation"):
        build(session, _rows(tmp_path, 50), hyperparameters_extra={"train_val_split_ratio": 0.9})


def test_large_dataset_keeps_recipe_split(session: Session, tmp_path: Path) -> None:
    assert build(session, _rows(tmp_path, 200))["HyperParameters"]["train_val_split_ratio"] == "0.9"


def test_dataset_limits_come_from_hub_recipes(session: Session) -> None:
    from skyhammer.catalog import Technique, TrainingType, get_model

    backend = SageMakerServerlessBackend(session)
    entry = get_model("qwen3-0.6b")
    sft = backend.dataset_limits(entry, Technique.SFT, TrainingType.LORA)
    assert sft.max_tokens_per_sample == 4096 and "SFT recipes" in sft.source
    rl = backend.dataset_limits(entry, Technique.RFT, TrainingType.LORA)
    assert rl.max_tokens_per_sample == 1024  # RLVR LoRA recipe is 1K in the real Hub data
    assert backend.dataset_limits(entry, Technique.CPT, TrainingType.LORA) == entry.limits


def test_plan_rejects_samples_longer_than_recipe_before_spend(
    session: Session, tmp_path: Path
) -> None:
    import json

    p = tmp_path / "long.jsonl"
    long = {
        "messages": [
            {"role": "user", "content": "x " * 10_000},
            {"role": "assistant", "content": "ok"},
        ]
    }
    short = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    p.write_text("\n".join(json.dumps(r) for r in [long] + [short] * 40))
    with pytest.raises(DatasetValidationError, match="exceeds model limit 4,096"):
        LLM("qwen3-0.6b", session=session).plan("sft", Dataset.from_file(p), output_model="m")


def test_dataset_limits_use_the_sagemaker_capability_id(session: Session) -> None:
    # Llama lists a Bedrock-native SFT capability first; the Hub lookup must use the
    # SageMaker model id (regression: live test fell back to generic limits).
    from skyhammer.catalog import Technique, TrainingType, get_model

    seen: list[str] = []
    backend = SageMakerServerlessBackend(session)
    real = backend.hub.model
    backend.hub.model = lambda mid: seen.append(mid) or real(mid)  # type: ignore[method-assign]
    backend.dataset_limits(get_model("llama-3.1-8b"), Technique.SFT, TrainingType.LORA)
    assert seen == ["meta-textgeneration-llama-3-1-8b-instruct"]


def test_dpo_request_uses_dpo_recipe_and_beta(session: Session, tmp_path: Path) -> None:
    import json

    p = tmp_path / "prefs.jsonl"
    rows = [
        {
            "input": {"messages": [{"role": "user", "content": f"q{i}"}]},
            "preferred_output": [{"role": "assistant", "content": "helpful answer"}],
            "non_preferred_output": [{"role": "assistant", "content": "rude answer"}],
        }
        for i in range(100)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    kw = build(session, p, model="qwen3-0.6b", technique="dpo", beta=0.05, epochs=1)
    assert kw["ServerlessJobConfig"]["CustomizationTechnique"] == "DPO"
    assert kw["HyperParameters"]["adam_beta"] == "0.05"
    with pytest.raises(UnsupportedParameterError, match="above maximum"):
        build(session, p, model="qwen3-0.6b", technique="dpo", beta=0.5)


def test_rl_jobs_are_capped_by_runtime_and_cost(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from skyhammer import pricing
    from skyhammer.errors import CostLimitExceededError

    monkeypatch.setattr(
        pricing, "training_price", lambda *a, **k: pricing.TrainingPrice(usd_per_hour=90.0)
    )
    p = tmp_path / "rft.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"messages": [{"role": "user", "content": f"q{i}"}], "answer": str(i)})
            for i in range(40)
        )
    )
    grader = Grader.from_lambda(
        "arn:aws:lambda:us-east-1:123456789012:function:skyhammer-grader-em"
    )
    kw = build(session, p, technique="rft", grader=grader)
    assert kw["StoppingCondition"] == {"MaxRuntimeInSeconds": 4 * 3600}  # default cap
    llm = LLM("qwen3-8b", session=session)
    _, est = llm.plan("rft", Dataset.from_file(p), output_model="m", grader=grader, max_hours=1.5)
    assert est.usd == 135.0 and "at most $135.00 (stops after 1.5h)" in str(est)
    with pytest.raises(CostLimitExceededError):
        llm.create_rft_job(Dataset.from_file(p), grader=grader, output_model="m", max_cost_usd=50)


def test_plan_catches_recipe_batch_problems_before_spend(session: Session, tmp_path: Path) -> None:
    """Seen live: a 64-row RL dataset passed plan() but failed at submit (batch size 128)."""
    with pytest.raises(UnsupportedParameterError, match="Batch size"):
        LLM("qwen3-0.6b", session=session).plan(
            "sft", Dataset.from_file(_small(tmp_path, 20)), output_model="m", batch_size=128
        )


def _small(tmp_path: Path, n: int) -> Path:
    import json

    p = tmp_path / "small.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a"}]}
        for i in range(n)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_reward_lambda_registered_as_hub_evaluator(session: Session) -> None:
    import json

    import boto3
    from botocore.stub import Stubber

    from skyhammer.backends.sagemaker import evaluator_hub_content

    lam = "arn:aws:lambda:us-east-1:123456789012:function:skyhammer-grader-em-1234abcd"
    req = evaluator_hub_content("skyhammer-grader-em-1234abcd", lam)
    shape = boto3.client("sagemaker", region_name="us-east-1").meta.service_model
    from botocore.validate import validate_parameters

    validate_parameters(req, shape.operation_model("ImportHubContent").input_shape)
    doc = json.loads(req["HubContentDocument"])
    assert doc["SubType"] == "AWS/Evaluator" and req["HubContentType"] == "JsonDoc"
    assert json.loads(doc["JsonContent"]) == {"Reference": lam, "EvaluatorType": "RewardFunction"}

    backend = SageMakerServerlessBackend(session)
    sm = boto3.client("sagemaker", region_name="us-east-1")
    stub = Stubber(sm)
    hub_arn = "arn:aws:sagemaker:us-east-1:123456789012:hub-content/skyhammer/JsonDoc/x/1.0.0"
    stub.add_response(
        "describe_hub",
        {
            "HubName": "skyhammer",
            "HubArn": "arn:h",
            "HubStatus": "InService",
            "CreationTime": "2026-01-01",
            "LastModifiedTime": "2026-01-01",
        },
        {"HubName": "skyhammer"},
    )
    stub.add_client_error("describe_hub_content", "ResourceNotFound")
    stub.add_response("import_hub_content", {"HubArn": "arn:h", "HubContentArn": hub_arn}, req)
    stub.add_response(
        "describe_hub_content",
        {
            "HubContentName": req["HubContentName"],
            "HubContentArn": hub_arn,
            "HubContentVersion": "1.0.0",
            "HubContentType": "JsonDoc",
            "DocumentSchemaVersion": "2.0.0",
            "HubName": "skyhammer",
            "HubArn": "arn:h",
            "HubContentDocument": "{}",
            "HubContentStatus": "Available",
            "CreationTime": "2026-01-01",
        },
    )
    stub.activate()
    backend.__dict__["sagemaker"] = sm
    assert backend.ensure_evaluator(lam) == hub_arn
    stub.assert_no_pending_responses()


def test_hub_description_matches_aws_pattern() -> None:
    """The real pattern isn't in botocore's model (it says ``.*``); this one is from the
    live CreateHub ValidationException (2026-10-29)."""
    import re

    from skyhammer.backends.sagemaker import HUB_DESCRIPTION

    assert re.fullmatch(r"[a-zA-Z0-9](-*[a-zA-Z0-9 .,])*", HUB_DESCRIPTION)


def test_import_waits_for_a_free_slot(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seen live: two finished jobs importing at once hit "Concurrent model import jobs"."""
    from botocore.exceptions import ClientError

    from skyhammer.jobs import JobState, JobStatus

    backend = SageMakerServerlessBackend(session)
    done = JobStatus(JobState.COMPLETED, output_artifacts="s3://b/out/")
    monkeypatch.setattr(backend, "describe", lambda handle: done)
    monkeypatch.setattr(backend, "_find_import", lambda name: None)
    quota = ClientError(
        {"Error": {"Code": "ServiceQuotaExceededException", "Message": "Concurrent model import"}},
        "CreateModelImportJob",
    )
    calls: list[int] = []

    def start(record: Any, uri: str) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise quota

    monkeypatch.setattr(backend, "_start_import", start)
    record = SimpleNamespace(serving="bedrock", job_name="j", handle=None)
    first = backend.poll(record)  # type: ignore[arg-type]
    assert first.state is JobState.IMPORTING and "free Bedrock import slot" in (first.message or "")
    assert backend.poll(record).message == "importing into Bedrock"  # type: ignore[arg-type]
