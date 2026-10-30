"""TR-11 RLAIF: an LLM judge deploys as a reward Lambda that any RL backend can call."""

from types import SimpleNamespace
from typing import Any

import boto3
import pytest

from skyhammer import Grader
from skyhammer.backends.sagemaker import _technique
from skyhammer.catalog import Technique
from skyhammer.graders import lambda_source

JUDGE = Grader.llm_judge("nova-pro", "Reward correct, concise answers.", name="concise")


class FakeRuntime:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def converse(self, **kw: Any) -> dict[str, Any]:
        assert (
            kw["modelId"] == "us.amazon.nova-pro-v1:0" and kw["inferenceConfig"]["temperature"] == 0
        )
        self.prompts.append(kw["messages"][0]["content"][0]["text"])
        return {"output": {"message": {"content": [{"text": self.reply}]}}}


def load(monkeypatch: pytest.MonkeyPatch, contract: str, reply: str) -> tuple[Any, FakeRuntime]:
    runtime = FakeRuntime(reply)
    monkeypatch.setattr(boto3, "client", lambda *a, **k: runtime)
    monkeypatch.setenv("SKYHAMMER_REWARD_CONTRACT", contract)
    namespace: dict[str, Any] = {}
    exec(
        compile(lambda_source(JUDGE, "us.amazon.nova-pro-v1:0"), "skyhammer_grader.py", "exec"),
        namespace,
    )
    return namespace["lambda_handler"], runtime


SAMPLE = {
    "id": "1",
    "messages": [
        {"role": "user", "content": "What is 10 + 10?"},
        {"role": "assistant", "content": "It's 20."},
    ],
    "metadata": {"reference_answer": "20"},
}


def test_judge_lambda_scores_with_the_rubric(monkeypatch: pytest.MonkeyPatch) -> None:
    handler, runtime = load(monkeypatch, "bedrock", '{"reasoning": "Right and short.", "score": 4}')
    [result] = handler([SAMPLE], None)
    assert (result["id"], result["aggregate_reward_score"]) == ("1", 0.75)  # 1-5 -> 0-1
    prompt = runtime.prompts[0]
    assert "Reward correct, concise answers." in prompt and "Reference answer:\n20" in prompt
    assert "Answer to grade:\nIt's 20." in prompt


def test_unparseable_judge_reply_scores_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    handler, _ = load(monkeypatch, "sagemaker", "I think it's fine.")
    resp = handler([SAMPLE], None)
    assert '"aggregate_reward_score": 0.0' in resp["body"]  # SageMaker reply shape
    assert handler({"sample": SAMPLE, "output": "20"}, None) == {"score": 0.0}  # eval contract


def test_judges_deploy_under_their_own_names() -> None:
    other = Grader.llm_judge("nova-pro", "Reward polite answers.", name="concise")
    assert JUDGE.function_name_for("bedrock") != other.function_name_for("bedrock")
    assert JUDGE.function_name_for("bedrock").startswith("skyhammer-grader-concise-br-")


def test_sagemaker_trains_judge_rewards_through_the_lambda_evaluator() -> None:
    request = SimpleNamespace(technique=Technique.RFT, grader=JUDGE)
    assert _technique(request, "arn:evaluator") == ("RLVR", "arn:evaluator")  # type: ignore[arg-type]


def test_throttled_judge_scores_zero_instead_of_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    from botocore.exceptions import ClientError

    class Throttled(FakeRuntime):
        def converse(self, **kw: Any) -> dict[str, Any]:
            raise ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "Too many requests"}},
                "Converse",
            )

    runtime = Throttled("")
    monkeypatch.setattr(boto3, "client", lambda *a, **k: runtime)
    monkeypatch.setenv("SKYHAMMER_REWARD_CONTRACT", "bedrock")
    namespace: dict[str, Any] = {}
    exec(compile(lambda_source(JUDGE, "us.amazon.nova-pro-v1:0"), "m.py", "exec"), namespace)
    [result] = namespace["lambda_handler"]([SAMPLE], None)
    assert result["aggregate_reward_score"] == 0.0  # the batch completes; training goes on
