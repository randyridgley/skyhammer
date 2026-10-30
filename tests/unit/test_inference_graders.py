from typing import Any

import pytest

from skyhammer.graders import Grader, grader
from skyhammer.inference import Chat, from_converse_response, to_converse_request


def test_converse_request_mapping() -> None:
    req = to_converse_request(
        "arn:dep",
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
        max_tokens=10,
        temperature=0.2,
        stop="END",
    )
    assert req == {
        "modelId": "arn:dep",
        "system": [{"text": "s"}],
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        "inferenceConfig": {"maxTokens": 10, "temperature": 0.2, "stopSequences": ["END"]},
    }


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "4"}]}},
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
        }


def test_chat_completions_openai_shape() -> None:
    rt = FakeRuntime()
    resp = Chat(rt, "arn:dep").completions.create(messages=[{"role": "user", "content": "2+2"}])
    assert resp.choices[0].message.content == "4"
    assert resp.choices[0].finish_reason == "length"
    assert resp.usage.total_tokens == 6
    assert resp.raw["stopReason"] == "max_tokens"


def test_from_converse_response_defaults() -> None:
    assert from_converse_response("m", {}).choices[0].message.content == ""


def test_python_grader() -> None:
    @grader
    def exact_match(sample: dict[str, Any], output: str) -> float:
        return float(output.strip() == sample["answer"])

    assert isinstance(exact_match, Grader)
    assert exact_match.name == "exact-match"
    assert exact_match.test([{"answer": "4"}, {"answer": "5"}], ["4 ", "4"]) == [1.0, 0.0]
    assert "def exact_match" in (exact_match.source or "")


def test_grader_range_checked() -> None:
    bad = grader(lambda s, o: 2.0, name="bad")
    with pytest.raises(ValueError, match="expected"):
        bad.test([{}], ["x"])


def test_lambda_and_judge_fingerprints_differ() -> None:
    a = Grader.from_lambda("arn:aws:lambda:us-east-1:1:function:g")
    b = Grader.llm_judge("amazon.nova-pro-v1:0", "rubric")
    assert a.fingerprint() != b.fingerprint()


def test_lambda_grader_invoked_for_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        "skyhammer.graders._invoke_lambda", lambda arn, s, o: calls.append((arn, s, o)) or 0.5
    )
    g = Grader.from_lambda("arn:aws:lambda:us-east-1:1:function:g")
    assert g({"answer": "4"}, "4") == 0.5
    assert calls == [("arn:aws:lambda:us-east-1:1:function:g", {"answer": "4"}, "4")]


def test_builtin_graders() -> None:
    sft_record = {
        "messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]
    }
    em, contains, js = (Grader.builtin(n) for n in ("exact_match", "contains", "json_valid"))
    assert em(sft_record, " 4 ") == 1.0 and em({"answer": "Paris"}, "paris") == 1.0
    assert em(sft_record, "four") == 0.0
    assert contains({"answer": "Paris"}, "The capital is Paris.") == 1.0
    assert contains({}, "anything") == 0.0
    assert (js({}, '{"a": 1}'), js({}, "nope")) == (1.0, 0.0)
    with pytest.raises(ValueError, match="Unknown builtin"):
        Grader.builtin("bleu")
