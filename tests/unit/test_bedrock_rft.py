"""Bedrock-native RFT (Nova 2 Lite, Qwen3-32B, GPT-OSS): data format, request, reward Lambda."""

from typing import Any

import boto3
import pytest
from botocore.validate import validate_parameters

from skyhammer import Grader, grader
from skyhammer.backends.bedrock import BedrockNativeBackend, to_bedrock_rft
from skyhammer.dataset import Dataset
from skyhammer.errors import DatasetValidationError, UnsupportedParameterError
from skyhammer.graders import lambda_source
from skyhammer.llm import LLM
from skyhammer.session import Session

LAMBDA = "arn:aws:lambda:us-east-1:123456789012:function:skyhammer-grader-sums-br-abc"


@grader
def correct_sum(sample: dict, output: str) -> float:
    return 1.0 if reference_answer(sample) in output else 0.0  # noqa: F821 - in the Lambda


def problems(n: int = 100) -> Dataset:
    return Dataset.from_records(
        [
            {
                "messages": [{"role": "user", "content": f"What is {i} + 1?"}],
                "answer": str(i + 1),
                "difficulty": "easy",
            }
            for i in range(n)
        ],
        name="sums",
    )


def plan(session: Session, **kw: Any) -> Any:
    request, _ = LLM("nova-2-lite", session=session).plan(
        "rft", kw.pop("dataset", problems()), output_model="sums-nova", grader=correct_sum, **kw
    )
    return request


def test_rft_rows_carry_reference_and_custom_fields() -> None:
    row = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "2+2?"}]}],
        "answer": "4",
        "id": "q1",
        "domain": "math",
    }
    assert to_bedrock_rft(row) == {
        "id": "q1",
        "domain": "math",
        "messages": [{"role": "user", "content": "2+2?"}],
        "reference_answer": "4",
    }
    # A trailing assistant turn is the reference, not part of the prompt.
    with_turn = {
        "messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]
    }
    assert to_bedrock_rft(with_turn)["reference_answer"] == "4"
    assert len(to_bedrock_rft(with_turn)["messages"]) == 1
    structured = {"messages": [{"role": "user", "content": "x"}], "reference_answer": {"n": 2}}
    assert to_bedrock_rft(structured)["reference_answer"] == {"n": 2}


def test_rft_request_is_schema_valid(session: Session) -> None:
    request = plan(
        session, epochs=2, learning_rate=1e-5, hyperparameters_extra={"inferenceMaxTokens": 512}
    )
    kwargs = BedrockNativeBackend(session).build_request(
        request, "s3://b/train.jsonl", None, "s3://b/out/", grader_arn=LAMBDA
    )
    assert kwargs["customizationType"] == "REINFORCEMENT_FINE_TUNING"
    assert kwargs["baseModelIdentifier"] == "amazon.nova-2-lite-v1:0:256k"
    rft = kwargs["customizationConfig"]["rftConfig"]
    assert rft["graderConfig"] == {"lambdaGrader": {"lambdaArn": LAMBDA}}
    assert rft["hyperParameters"] == {
        "epochCount": 2,
        "learningRate": 1e-5,
        "inferenceMaxTokens": 512,
    }
    assert "hyperParameters" not in kwargs  # RFT settings live in rftConfig
    shape = boto3.client("bedrock", region_name="us-east-1").meta.service_model.operation_model(
        "CreateModelCustomizationJob"
    )
    validate_parameters(kwargs, shape.input_shape)


def test_rft_needs_a_lambda_reward(session: Session) -> None:
    backend = BedrockNativeBackend(session)
    request = plan(session)
    with pytest.raises(UnsupportedParameterError, match="deploy the grader"):
        backend.build_request(request, "s3://t", None, "s3://o")
    request.grader = Grader.llm_judge("nova-pro", "Is it correct?")
    with pytest.raises(UnsupportedParameterError, match="Lambda rewards"):
        backend.build_request(request, "s3://t", None, "s3://o", grader_arn=LAMBDA)
    request.grader = Grader.from_lambda(LAMBDA)
    kwargs = backend.build_request(request, "s3://t", None, "s3://o")
    assert (
        kwargs["customizationConfig"]["rftConfig"]["graderConfig"]["lambdaGrader"]["lambdaArn"]
        == LAMBDA
    )


def test_rft_needs_100_prompts(session: Session) -> None:
    with pytest.raises(DatasetValidationError, match="at least 100"):
        plan(session, dataset=problems(99))


def test_rft_is_routed_to_bedrock_in_its_region(session: Session) -> None:
    request = plan(session)
    assert (request.route.backend.value, request.route.region) == ("bedrock", "us-east-1")


def load_handler(g: Grader, monkeypatch: pytest.MonkeyPatch, contract: str) -> Any:
    monkeypatch.setenv("SKYHAMMER_REWARD_CONTRACT", contract)
    namespace: dict[str, Any] = {}
    exec(compile(lambda_source(g), "skyhammer_grader.py", "exec"), namespace)
    return namespace["lambda_handler"]


def test_bedrock_reward_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = load_handler(correct_sum, monkeypatch, "bedrock")
    # The documented Bedrock input: custom fields and the reference arrive in metadata.
    batch = [
        {
            "id": "123",
            "messages": [
                {"role": "user", "content": "What is 3 + 4?"},
                {"role": "assistant", "content": "3 + 4 = 7"},
            ],
            "metadata": {"reference_answer": "7", "difficulty": "easy"},
        },
        {
            "id": "124",
            "messages": [
                {"role": "user", "content": "1+1?"},
                {"role": "assistant", "content": "3"},
            ],
            "metadata": {"reference_answer": {"text": "2"}},
        },
    ]
    out = handler(batch, None)
    assert isinstance(out, list)  # a plain list, not SageMaker's {"statusCode", "body"}
    assert [(r["id"], r["aggregate_reward_score"]) for r in out] == [("123", 1.0), ("124", 0.0)]
    assert out[0]["metrics_list"][0]["type"] == "Reward"
    assert handler({"trajectories": batch[:1]}, None)[0]["aggregate_reward_score"] == 1.0
    # SageMaker deployments of the same grader keep their reply shape.
    sm = load_handler(correct_sum, monkeypatch, "sagemaker")
    assert "body" in sm(batch, None)


def test_bedrock_and_sagemaker_rewards_deploy_separately() -> None:
    br, sm = correct_sum.function_name_for("bedrock"), correct_sum.function_name_for("sagemaker")
    assert br != sm and "-br-" in br and len(br) <= 64
    assert correct_sum.function_name == sm
