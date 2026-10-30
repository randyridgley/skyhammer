import io
import json
import zipfile
from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from moto import mock_aws

from skyhammer import Grader, grader
from skyhammer.formats import to_sagemaker_rlvr
from skyhammer.graders import GraderKind, _lambda_zip, judge_prompt, lambda_source
from skyhammer.session import Session


@grader
def word_overlap(sample: dict, output: str) -> float:
    """Fraction of reference words present in the output."""
    ref = reference_answer(sample).lower().split()  # noqa: F821 - provided in the Lambda
    return sum(w in output.lower() for w in ref) / len(ref) if ref else 0.0


def load_handler(g: Grader) -> Any:
    namespace: dict[str, Any] = {}
    exec(compile(lambda_source(g), "skyhammer_grader.py", "exec"), namespace)
    return namespace["lambda_handler"]


def test_lambda_speaks_sagemaker_rlvr_and_eval_contracts() -> None:
    handler = load_handler(word_overlap)
    # SageMaker RLVR: a list of samples, the answer in the last assistant turn, reference in
    # reference_answer.text (from AWS's sample reward Lambda, 2026-10-29).
    batch = [
        {
            "id": "s1",
            "messages": [
                {"role": "user", "content": "Capital of France?"},
                {"role": "assistant", "content": "It is Paris, France"},
            ],
            "reference_answer": {"text": "Paris France"},
        },
        {"id": "s2", "response": "no idea", "extra_info": {"reference_answer": {"text": "Rome"}}},
    ]
    resp = handler(batch, None)
    results = json.loads(resp["body"])
    assert [r["aggregate_reward_score"] for r in results] == [1.0, 0.0]
    assert results[0]["id"] == "s1"
    assert results[0]["metrics_list"][0] == {"name": "word-overlap", "value": 1.0, "type": "Reward"}
    # API Gateway / function URL style body, and {"batch": [...]}
    assert json.loads(handler({"body": json.dumps({"batch": batch[:1]})}, None)["body"])
    # Skyhammer eval contract used by Grader.from_lambda(...)(sample, output)
    assert handler({"sample": {"answer": "Paris"}, "output": "Paris!"}, None) == {"score": 1.0}


def test_builtin_graders_package_too() -> None:
    handler = load_handler(Grader.builtin("exact_match"))
    assert handler({"sample": {"answer": "Paris"}, "output": " paris "}, None) == {"score": 1.0}
    with zipfile.ZipFile(io.BytesIO(_lambda_zip(word_overlap))) as z:
        assert z.namelist() == ["skyhammer_grader.py"]


def test_rlvr_records_carry_the_reference_answer() -> None:
    rec = {"messages": [{"role": "user", "content": "2+2?"}], "answer": "4"}
    assert to_sagemaker_rlvr(rec) == {
        "prompt": [{"role": "user", "content": "2+2?"}],
        "reward_model": {"ground_truth": "4", "style": "rule"},
    }
    with_turn = {"messages": [*rec["messages"], {"role": "assistant", "content": "4"}]}
    assert to_sagemaker_rlvr(with_turn)["reward_model"]["ground_truth"] == "4"
    assert to_sagemaker_rlvr(with_turn)["prompt"] == rec["messages"]


def test_deploy_creates_a_tagged_versioned_lambda_once(session: Session) -> None:
    with mock_aws():
        iam = boto3.client("iam", region_name="us-east-1")
        role = iam.create_role(
            RoleName="grader",
            AssumeRolePolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": []}),
        )["Role"]["Arn"]
        session.settings.grader_role_arn = role
        deployed = word_overlap.deploy(session)
        assert deployed.kind is GraderKind.LAMBDA and deployed.name == "word-overlap"
        assert deployed.lambda_arn and word_overlap.function_name in deployed.lambda_arn
        lam = boto3.client("lambda", region_name="us-east-1")
        fn = lam.get_function(FunctionName=word_overlap.function_name)
        assert fn["Tags"]["skyhammer:kind"] == "grader"
        assert fn["Configuration"]["Handler"] == "skyhammer_grader.lambda_handler"
        assert word_overlap.deploy(session).lambda_arn == deployed.lambda_arn  # idempotent
        word_overlap.delete(session)
        assert not lam.list_functions()["Functions"]


def test_llm_judge_scores_with_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    from skyhammer import llm as llm_mod

    seen: dict[str, Any] = {}

    class FakeLLM:
        def __init__(self, model: str) -> None:
            seen["model"] = model
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, messages: list[dict[str, Any]], **kw: Any) -> Any:
            seen.update(prompt=messages[0]["content"], **kw)
            msg = SimpleNamespace(parsed=lambda: {"reasoning": "ok", "score": 4})
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    monkeypatch.setattr(llm_mod, "LLM", FakeLLM)
    judge = Grader.llm_judge("claude-haiku-4-5", rubric="Polite and correct.")
    sample = {"messages": [{"role": "user", "content": "Capital of France?"}], "answer": "Paris"}
    assert judge(sample, "Paris, of course!") == 0.75
    assert seen["model"] == "claude-haiku-4-5" and seen["temperature"] == 0
    assert seen["response_format"]["type"] == "json_schema"
    assert "Reference answer:\nParis" in seen["prompt"] and "Polite and correct." in seen["prompt"]
    assert "Reference" not in judge_prompt("r", {"messages": []}, "x")
