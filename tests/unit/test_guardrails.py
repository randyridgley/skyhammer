"""Bedrock guardrails on Converse, imported models (InvokeModel) and SageMaker endpoints.

Fixtures are real responses recorded live on 2026-10-30 with a deny-topic guardrail.
"""

import io
import json
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.validate import validate_parameters

from skyhammer.inference import (
    Chat,
    from_converse_response,
    from_openai_chat_response,
    invoke_guardrail_args,
    openai_stream_chunks,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
RT = boto3.client("bedrock-runtime", region_name="us-east-1")
GUARDRAIL = {"id": "gr1", "version": 3, "trace": True}
BLOCKED = "Sorry, I can't help with investment advice."


def fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    return data


def test_converse_block_is_content_filter() -> None:
    resp = from_converse_response("m", fixture("guardrail_converse_blocked.json"))
    assert resp.choices[0].finish_reason == "content_filter"
    assert resp.choices[0].message.content == BLOCKED
    assert resp.raw["trace"]["guardrail"]["actionReason"] == "Guardrail blocked."


def test_invoke_model_guardrail_args_are_valid_request_params() -> None:
    args = invoke_guardrail_args({"guardrail": GUARDRAIL})
    assert args == {"guardrailIdentifier": "gr1", "guardrailVersion": "3", "trace": "ENABLED"}
    shape = RT.meta.service_model.operation_model("InvokeModel").input_shape
    validate_parameters({"modelId": "m", "body": b"{}", **args}, shape)
    assert invoke_guardrail_args({}) == {}


def test_invoke_model_block_is_content_filter() -> None:
    # InvokeModel keeps the model's stop reason and flags the block in a separate field.
    recorded = fixture("guardrail_invoke_model_blocked.json")
    openai_body = {
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": BLOCKED}}],
        "amazon-bedrock-guardrailAction": recorded["amazon-bedrock-guardrailAction"],
    }
    resp = from_openai_chat_response("arn:imported", openai_body)
    assert resp.choices[0].finish_reason == "content_filter"
    assert resp.choices[0].message.content == BLOCKED
    unblocked = {**openai_body, "amazon-bedrock-guardrailAction": "NONE"}
    assert from_openai_chat_response("m", unblocked).choices[0].finish_reason == "stop"


def test_invoke_model_stream_block_is_content_filter() -> None:
    chunk = {
        "choices": [{"index": 0, "delta": {"content": BLOCKED}, "finish_reason": "stop"}],
        "amazon-bedrock-guardrailAction": "INTERVENED",
    }
    events = [{"chunk": {"bytes": json.dumps(chunk).encode()}}]
    [c] = list(openai_stream_chunks("m", events))
    assert c.choices[0].finish_reason == "content_filter"


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke_model(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        body = {"choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "ok"}}]}
        return {"body": io.BytesIO(json.dumps(body).encode())}


def test_imported_model_chat_sends_guardrail_outside_the_body() -> None:
    rt = FakeRuntime()
    Chat(rt, "arn:imported", api="openai_chat").completions.create(
        messages=[{"role": "user", "content": "hi"}], guardrail=GUARDRAIL
    )
    [call] = rt.calls
    assert call["guardrailIdentifier"] == "gr1" and call["trace"] == "ENABLED"
    assert "guardrail" not in json.loads(call["body"])


def test_sagemaker_endpoint_rejects_guardrail() -> None:
    chat = Chat(object(), "endpoint", api="sagemaker")
    with pytest.raises(ValueError, match="SageMaker endpoints"):
        chat.completions.create(messages=[{"role": "user", "content": "hi"}], guardrail=GUARDRAIL)
