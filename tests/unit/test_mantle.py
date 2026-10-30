"""bedrock-mantle (Bedrock's OpenAI-compatible fine-tuning API): open-weight RFT."""

import json
from typing import Any

import pytest

from skyhammer import Dataset, grader
from skyhammer.backends.mantle import BedrockMantleBackend
from skyhammer.catalog import BackendKind, Technique, get_model
from skyhammer.errors import UnsupportedParameterError
from skyhammer.inference import Chat
from skyhammer.jobs import JobHandle, JobState
from skyhammer.llm import LLM
from skyhammer.mantle import MantleError, _multipart
from skyhammer.router import resolve
from skyhammer.session import Session

LAMBDA = "arn:aws:lambda:us-west-2:123456789012:function:skyhammer-grader-sums-br-abc"


@grader
def correct_sum(sample: dict, output: str) -> float:
    return 1.0 if reference_answer(sample) in output else 0.0  # noqa: F821 - in the Lambda


def problems(n: int = 100) -> Dataset:
    return Dataset.from_records(
        [
            {"messages": [{"role": "user", "content": f"What is {i} + 1?"}], "answer": str(i + 1)}
            for i in range(n)
        ],
        name="sums",
    )


class FakeClient:
    def __init__(self, responses: dict[tuple[str, str], Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, Any]] = []

    def request(self, method: str, path: str, **kw: Any) -> Any:
        self.calls.append((method, path, kw))
        resp = self.responses.get((method, path.split("?")[0]))
        if isinstance(resp, Exception):
            raise resp
        return resp or {}


def test_open_weight_rft_routes_cross_region_to_mantle() -> None:
    route = resolve(get_model("qwen3-32b"), Technique.RFT, "us-east-1", backend=BackendKind.MANTLE)
    assert (route.backend, route.region, route.cross_region) == (
        BackendKind.MANTLE,
        "us-west-2",
        True,
    )
    assert route.serving.value == "bedrock"
    # Without asking for it, home-region SageMaker still wins.
    assert (
        resolve(get_model("qwen3-32b"), Technique.RFT, "us-east-1").backend is BackendKind.SAGEMAKER
    )


def plan(session: Session, **kw: Any) -> Any:
    request, est = LLM("qwen3-32b", session=session).plan(
        "rft", problems(), output_model="sums-qwen", grader=correct_sum, backend="mantle", **kw
    )
    return request, est


def test_job_body(session: Session) -> None:
    request, est = plan(
        session, epochs=2, batch_size=16, hyperparameters_extra={"learning_rate_multiplier": 0.5}
    )
    assert "unknown" in str(est)  # not in the Price List yet
    body = BedrockMantleBackend(session, "us-west-2").build_request(
        request, "file-abc", grader_arn=LAMBDA
    )
    assert body == {
        "model": "qwen.qwen3-32b",
        "training_file": "file-abc",
        "method": {
            "type": "reinforcement",
            "reinforcement": {
                "grader": {"type": "lambda", "lambda": {"function": LAMBDA}},
                "hyperparameters": {
                    "n_epochs": 2,
                    "batch_size": 16,
                    "learning_rate_multiplier": 0.5,
                },
            },
        },
    }


def test_unsupported_settings_are_rejected(session: Session) -> None:
    request, _ = plan(session, learning_rate=1e-5)
    with pytest.raises(UnsupportedParameterError, match="learning_rate"):
        BedrockMantleBackend(session, "us-west-2").build_request(request, "f", grader_arn=LAMBDA)
    with pytest.raises(Exception, match="max_hours"):
        plan(session, max_hours=1)


def test_submit_uploads_then_creates_the_job(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = plan(session, batch_size=16)
    backend = BedrockMantleBackend(session, "us-west-2")
    backend.client = FakeClient(
        {
            ("POST", "/v1/files"): {"id": "file-1"},
            ("POST", "/v1/fine_tuning/jobs"): {"id": "ftjob-1", "status": "validating_files"},
        }
    )
    deployed: list[tuple[str, str]] = []

    def fake_deploy(self: Any, session: Any = None, *, region: str, contract: str) -> Any:
        deployed.append((region, contract))
        from skyhammer.graders import Grader

        return Grader.from_lambda(LAMBDA)

    monkeypatch.setattr(type(correct_sum), "deploy", fake_deploy)
    handle = backend.submit(request)
    assert (handle.job_id, handle.region, handle.backend) == ("ftjob-1", "us-west-2", "mantle")
    assert deployed == [("us-west-2", "bedrock")]  # Bedrock's reward contract, in-region
    (_, _, upload), (_, _, create) = backend.client.calls
    filename, content = upload["files"]["file"]
    assert filename == f"{request.job_name}.jsonl" and upload["fields"] == {"purpose": "fine-tune"}
    first = json.loads(content.splitlines()[0])
    assert first == {
        "messages": [{"role": "user", "content": "What is 0 + 1?"}],
        "reference_answer": "1",
    }
    assert create["json_body"]["training_file"] == "file-1"


def test_status_logs_and_finalize(session: Session) -> None:
    backend = BedrockMantleBackend(session, "us-west-2")
    handle = JobHandle("mantle", "sums-qwen-x", "ftjob-1", "us-west-2")
    backend.client = FakeClient(
        {
            ("GET", "/v1/fine_tuning/jobs/ftjob-1"): {
                "status": "succeeded",
                "fine_tuned_model": "ft:qwen.qwen3-32b:sums",
            },
            ("GET", "/v1/fine_tuning/jobs/ftjob-1/events"): {
                "data": [
                    {"level": "info", "message": "Job succeeded"},
                    {"level": "info", "message": "Step 10", "data": {"reward": 0.9}},
                ]
            },
        }
    )
    st = backend.describe(handle)
    assert (st.state, st.output_model_arn) == (JobState.COMPLETED, "ft:qwen.qwen3-32b:sums")
    record = type(
        "R",
        (),
        {
            "handle": handle,
            "job_id": "ftjob-1",
            "job_name": "sums-qwen-x",
            "output_model": "sums-qwen",
            "base_model": "qwen3-32b",
        },
    )()
    assert backend.logs(record) == ['info: Step 10 {"reward": 0.9}', "info: Job succeeded"]
    ref = backend.finalize(record)
    assert (ref.api, ref.model_arn, ref.region) == ("mantle", "ft:qwen.qwen3-32b:sums", "us-west-2")


def test_delete_model_tolerates_unsupported(session: Session) -> None:
    backend = BedrockMantleBackend(session, "us-west-2")
    backend.client = FakeClient(
        {("DELETE", "/v1/models/ft:x"): MantleError(404, "DELETE", "/", "")}
    )
    backend.delete_model(type("M", (), {"model_arn": "ft:x"})())  # logs, doesn't raise


def test_chat_with_a_mantle_fine_tune() -> None:
    client = FakeClient(
        {
            ("POST", "/v1/chat/completions"): {
                "id": "c1",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "42"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
        }
    )
    chat = Chat(client, "ft:qwen.qwen3-32b:sums", api="mantle")
    r = chat.completions.create(messages=[{"role": "user", "content": "17+25?"}], max_tokens=20)
    assert r.choices[0].message.content == "42" and r.usage.total_tokens == 6
    body = client.calls[0][2]["json_body"]
    assert body["model"] == "ft:qwen.qwen3-32b:sums" and body["max_tokens"] == 20
    streamed = chat.completions.create(messages=[{"role": "user", "content": "x"}], stream=True)
    assert streamed.final().choices[0].message.content == "42"


def test_multipart_encoding() -> None:
    body, content_type = _multipart({"purpose": "fine-tune"}, {"file": ("a.jsonl", b'{"x":1}\n')})
    boundary = content_type.split("boundary=")[1]
    assert body.startswith(f"--{boundary}\r\n".encode()) and body.endswith(
        f"--{boundary}--\r\n".encode()
    )
    assert b'name="purpose"\r\n\r\nfine-tune\r\n' in body
    assert b'filename="a.jsonl"' in body and b'{"x":1}\n\r\n' in body


def test_batch_size_is_checked_against_the_training_split(session: Session) -> None:
    backend = BedrockMantleBackend(session, "us-west-2")
    request, _ = plan(session)  # 100 prompts -> ~90 train, default batch 128
    with pytest.raises(UnsupportedParameterError, match=r"~90 of 100 .* 128 by default"):
        backend.build_request(request, "f", grader_arn=LAMBDA)
    request, _ = plan(session, batch_size=96)
    with pytest.raises(UnsupportedParameterError, match="batch size is 96"):
        backend.build_request(request, "f", grader_arn=LAMBDA)
