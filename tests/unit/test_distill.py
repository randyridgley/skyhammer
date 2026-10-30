from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from moto import mock_aws

from skyhammer import LLM, Dataset, distill
from skyhammer.batch import BatchResult
from skyhammer.catalog import Technique
from skyhammer.errors import CostLimitExceededError, SkyhammerError
from skyhammer.session import Session


def completion(text: str | None) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeTeacher:
    """Duck-typed teacher LLM: answers live or by batch, records what it was asked."""

    def __init__(self, session: Session, fail: set[str] = frozenset()) -> None:  # type: ignore[assignment]
        self.session = session
        self.model_id = "us.anthropic.claude-haiku-4-5"
        self.fail = fail
        self.live: list[str] = []
        self.batches: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, messages: list[dict[str, Any]], **params: Any) -> Any:
        q = messages[-1]["content"]
        self.live.append(q)
        if q in self.fail:
            raise SkyhammerError("throttled")
        return completion(f"answer to {q} ")

    def batch(self, requests: list[dict[str, Any]], *, name: str, **params: Any) -> Any:
        self.batches.append({"name": name, "n": len(requests), **params})
        results = [
            BatchResult(r["custom_id"], completion(f"batch {r['messages'][-1]['content']}"))
            for r in requests
        ]
        results[0] = BatchResult(results[0].custom_id, None, "content filtered")
        return SimpleNamespace(wait=lambda **_: None, results=lambda: results)


def prompts(n: int, with_answers: bool = False) -> Dataset:
    recs = []
    for i in range(n):
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": f"q{i}"}]
        if with_answers:
            msgs.append({"role": "assistant", "content": "old"})
        recs.append({"messages": msgs})
    return Dataset.from_records(recs, "support")


@pytest.fixture
def s3(session: Session) -> Any:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=session.settings.bucket or "")
        yield client


def test_live_answers_replace_old_turns_drop_failures_and_are_cached(
    session: Session, s3: Any
) -> None:
    teacher = FakeTeacher(session, fail={"q1"})
    ds = distill.teacher_dataset(teacher, prompts(5, with_answers=True), max_tokens=64)  # type: ignore[arg-type]
    rows = [r for _, r in ds._rows]
    assert ds.name == "support-teacher" and len(rows) == 4
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": "answer to q0"}
    assert [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant"]
    assert not teacher.batches

    again = FakeTeacher(session)
    cached = distill.teacher_dataset(again, prompts(5, with_answers=True), max_tokens=64)  # type: ignore[arg-type]
    assert again.live == [] and cached.sha256 == ds.sha256  # same SFT fingerprint on re-run
    obj = s3.list_objects_v2(Bucket=session.settings.bucket)["Contents"][0]
    assert obj["Key"].startswith("distill/")
    head = s3.head_object(Bucket=session.settings.bucket, Key=obj["Key"])
    assert head["ServerSideEncryption"] == "aws:kms"

    changed = FakeTeacher(session)
    distill.teacher_dataset(changed, prompts(5), max_tokens=128)  # type: ignore[arg-type]
    assert len(changed.live) == 5  # different settings -> new generation


def test_batch_on_request_or_from_1000_prompts(session: Session, s3: Any) -> None:
    teacher = FakeTeacher(session)
    ds = distill.teacher_dataset(teacher, prompts(100), temperature=0.2, use_batch=True)  # type: ignore[arg-type]
    assert teacher.live == [] and teacher.batches[0]["n"] == 100
    assert teacher.batches[0]["name"].startswith("skyhammer-distill-")
    assert teacher.batches[0]["temperature"] == 0.2
    assert len(ds) == 99  # the filtered record is dropped
    assert next(r for _, r in ds._rows)["messages"][-1]["content"] == "batch q1"


def test_prompts_must_end_with_user_turn() -> None:
    with pytest.raises(SkyhammerError, match="end with a user message"):
        distill.prompt_messages({"messages": [{"role": "system", "content": "x"}]})


def test_open_weight_student_distills_via_teacher_answers_then_sft(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}

    def fake_create(self: LLM, technique: Technique, dataset: Dataset, **kw: Any) -> str:
        calls.update(technique=technique, dataset=dataset, **kw)
        return "job"

    answers = Dataset.from_records([{"messages": []}], "support-teacher")
    monkeypatch.setattr(LLM, "_create_job", fake_create)
    monkeypatch.setattr(distill, "teacher_dataset", lambda *a, **k: answers)
    monkeypatch.setattr(distill, "estimate_teacher_cost", lambda *a, **k: 0.5)
    teacher = LLM("claude-haiku-4-5", session=session)

    LLM("qwen3-0.6b", session=session).create_distillation_job(
        prompts(10),
        teacher=teacher,
        output_model="qwen-distilled",
        tags={"team": "x"},  # type: ignore[arg-type]
    )
    assert calls["technique"] is Technique.SFT and calls["dataset"] is answers
    assert calls["tags"] == {"distilled-from": teacher.model_id, "team": "x"}

    with pytest.raises(CostLimitExceededError):
        LLM("qwen3-0.6b", session=session).create_distillation_job(
            prompts(10),
            teacher=teacher,
            output_model="qwen-distilled",
            max_cost_usd=0.1,  # type: ignore[arg-type]
        )

    LLM("nova-micro", session=session).create_job(
        "distillation",
        prompts(10),
        output_model="nova-distilled",
        teacher_model="amazon.nova-pro-v1:0",
    )
    assert calls["technique"] is Technique.DISTILLATION
    assert calls["teacher_model"] == "amazon.nova-pro-v1:0"


def test_plan_explains_two_step_distillation(session: Session) -> None:
    with pytest.raises(SkyhammerError, match="two steps"):
        LLM("qwen3-0.6b", session=session).plan(
            "distillation", prompts(10), output_model="x", teacher_model="nova-pro"
        )


def test_small_jobs_stay_live_by_default(session: Session, s3: Any) -> None:
    """Seen live: a 100-prompt batch sat in Bedrock's queue 73+ min to save ~4 cents."""
    teacher = FakeTeacher(session)
    distill.teacher_dataset(teacher, prompts(150))  # type: ignore[arg-type]
    assert len(teacher.live) == 150 and not teacher.batches
    with pytest.raises(SkyhammerError, match="at least 100"):
        distill.teacher_dataset(FakeTeacher(session), prompts(5), use_batch=True)  # type: ignore[arg-type]
