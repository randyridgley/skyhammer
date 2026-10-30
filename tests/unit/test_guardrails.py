"""Bedrock guardrails on Converse (on the call), and on imported models and SageMaker
endpoints (ApplyGuardrail before and after the call).

Fixtures are real responses recorded live on 2026-10-30 with a deny-topic guardrail.
"""

import io
import json
from pathlib import Path
from typing import Any

import boto3
from botocore.validate import validate_parameters

from skyhammer.inference import Chat, from_converse_response

FIXTURES = Path(__file__).parents[1] / "fixtures"
RT = boto3.client("bedrock-runtime", region_name="us-east-1")
GUARDRAIL = {"id": "gr1", "version": 3, "trace": True}
BLOCKED = "Sorry, I can't help with investment advice."
GUARDED = "BLOCKED"  # the live ApplyGuardrail fixture's blocked message


def fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    return data


def test_converse_block_is_content_filter() -> None:
    resp = from_converse_response("m", fixture("guardrail_converse_blocked.json"))
    assert resp.choices[0].finish_reason == "content_filter"
    assert resp.choices[0].message.content == BLOCKED
    assert resp.raw["trace"]["guardrail"]["actionReason"] == "Guardrail blocked."


class FakeRuntime:
    """bedrock-runtime / sagemaker-runtime stand-in: ApplyGuardrail blocks 'stocks'."""

    def __init__(self, answer: str = "4") -> None:
        self.answer = answer
        self.calls: list[str] = []
        self.checked: list[tuple[str, list[str]]] = []

    def apply_guardrail(self, **kw: Any) -> dict[str, Any]:
        validate_parameters(kw, RT.meta.service_model.operation_model("ApplyGuardrail").input_shape)
        texts = [b["text"]["text"] for b in kw["content"]]
        self.checked.append((kw["source"], texts))
        if any("stocks" in t for t in texts):
            return fixture("apply_guardrail_intervened.json")  # recorded live
        return {"action": "NONE", "outputs": []}

    def invoke_model(self, **kw: Any) -> dict[str, Any]:
        self.calls.append("invoke_model")
        assert "guardrailIdentifier" not in kw  # InvokeModel rejects OpenAI bodies with one
        body = {
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": self.answer}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
        return {"body": io.BytesIO(json.dumps(body).encode())}

    def invoke_endpoint(self, **kw: Any) -> dict[str, Any]:
        self.calls.append("invoke_endpoint")
        return {"Body": self.invoke_model(**kw)["body"]}


def ask(text: str) -> list[dict[str, Any]]:
    return [{"role": "system", "content": "Be brief."}, {"role": "user", "content": text}]


def test_imported_model_blocked_prompt_skips_the_model() -> None:
    rt = FakeRuntime()
    chat = Chat(rt, "arn:imported", api="openai_chat")
    r = chat.completions.create(messages=ask("Which stocks should I buy?"), guardrail=GUARDRAIL)
    assert (r.choices[0].finish_reason, r.choices[0].message.content) == ("content_filter", GUARDED)
    assert rt.calls == []  # never reached the model
    assert rt.checked == [("INPUT", ["Be brief.", "Which stocks should I buy?"])]
    assert r.raw["guardrail"]["input"]["action"] == "GUARDRAIL_INTERVENED"  # trace=True


def test_imported_model_answer_is_checked_too() -> None:
    rt = FakeRuntime(answer="Buy these stocks: ...")
    r = Chat(rt, "arn:imported", api="openai_chat").completions.create(
        messages=ask("Any tips?"), guardrail={"id": "gr1"}
    )
    assert rt.calls == ["invoke_model"]
    assert [src for src, _ in rt.checked] == ["INPUT", "OUTPUT"]
    assert r.choices[0].finish_reason == "content_filter"
    assert r.choices[0].message.content == GUARDED
    assert "guardrail" not in r.raw  # no trace requested


def test_allowed_call_is_unchanged_and_streams() -> None:
    rt = FakeRuntime()
    chat = Chat(rt, "arn:imported", api="openai_chat")
    r = chat.completions.create(messages=ask("2+2?"), guardrail=GUARDRAIL)
    assert (r.choices[0].finish_reason, r.choices[0].message.content) == ("stop", "4")
    blocked = chat.completions.create(
        messages=ask("Which stocks?"), guardrail=GUARDRAIL, stream=True
    ).final()
    assert (blocked.choices[0].finish_reason, blocked.choices[0].message.content) == (
        "content_filter",
        GUARDED,
    )


def test_sagemaker_endpoint_uses_bedrock_for_the_check() -> None:
    sm, br = FakeRuntime(), FakeRuntime()
    chat = Chat(sm, "endpoint", api="sagemaker", component="ic", guardrail_runtime=br)
    r = chat.completions.create(messages=ask("Which stocks?"), guardrail=GUARDRAIL)
    assert r.choices[0].finish_reason == "content_filter"
    assert br.checked and not sm.checked and sm.calls == []
