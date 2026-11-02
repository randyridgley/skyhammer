import json
import re
from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from moto import mock_aws

from skyhammer import Dataset, synthetic
from skyhammer.errors import CostLimitExceededError, SkyhammerError
from skyhammer.session import Session


class FakeGenerator:
    """Duck-typed generator LLM returning structured examples; records each prompt."""

    def __init__(self, session: Session, repeat: bool = False, fail_every: int = 0) -> None:
        self.session = session
        self.name = "nova-pro"
        self.model_id = "us.amazon.nova-pro-v1:0"
        self.repeat = repeat
        self.fail_every = fail_every
        self.prompts: list[str] = []
        self.params: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, messages: list[dict[str, Any]], **params: Any) -> Any:
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        self.params.append(params)
        if self.fail_every and len(self.prompts) % self.fail_every == 0:
            raise SkyhammerError("throttled")
        batch = re.search(r"batch (\d+)", prompt)
        tag = "same" if self.repeat else (batch.group(1) if batch else "x")
        k = int(re.search(r"Write (\d+)", prompt).group(1))  # type: ignore[union-attr]
        data = {
            "examples": [{"question": f"Q{tag}-{i}?", "answer": f"A{tag}-{i}"} for i in range(k)]
        }
        msg = SimpleNamespace(parsed=lambda: data)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


@pytest.fixture
def s3(session: Session, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(synthetic, "estimate_cost", lambda *a, **k: 0.25)
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=session.settings.bucket or "")
        yield client


def test_non_contextual_generation_is_structured_and_cached(session: Session, s3: Any) -> None:
    gen = FakeGenerator(session)
    seeds = Dataset.from_records(
        [
            {
                "messages": [
                    {"role": "user", "content": "How do I reset 2FA?"},
                    {"role": "assistant", "content": "Settings > Security."},
                ]
            }
        ],
        "seed",
    )
    ds = synthetic.synthesize(
        gen,
        n=12,
        task="Customer support for a SaaS app",
        examples=seeds,
        system="Be brief.",  # type: ignore[arg-type]
    )
    assert len(ds) == 12 and len(gen.prompts) == 3  # ceil(12 / 5) calls
    assert gen.params[0]["response_format"]["type"] == "json_schema"
    assert "Customer support for a SaaS app" in gen.prompts[0]
    assert "How do I reset 2FA?" in gen.prompts[0]  # seed style shown
    first = ds._rows[0][1]["messages"]
    assert [m["role"] for m in first] == ["system", "user", "assistant"]
    # Same inputs again: served from the S3 cache, no generation calls.
    again = FakeGenerator(session)
    cached = synthetic.synthesize(
        again,
        n=12,
        task="Customer support for a SaaS app",
        examples=seeds,
        system="Be brief.",  # type: ignore[arg-type]
    )
    assert again.prompts == [] and cached.sha256 == ds.sha256


def test_contextual_generation_grounds_each_call_in_a_chunk(session: Session, s3: Any) -> None:
    gen = FakeGenerator(session)
    docs = ["\n\n".join(f"Section {i}: " + "detail " * 400 for i in range(6))]
    ds = synthetic.synthesize(gen, n=10, documents=docs, per_call=2)  # type: ignore[arg-type]
    assert len(ds) == 10 and len(gen.prompts) == 5
    assert all("<source>" in p for p in gen.prompts)
    # Calls spread across the document instead of all reading the first chunk.
    sections = {re.search(r"Section (\d)", p).group(1) for p in gen.prompts}  # type: ignore[union-attr]
    assert len(sections) > 1


def test_duplicates_and_failed_calls_are_dropped(session: Session, s3: Any) -> None:
    ds = synthetic.synthesize(FakeGenerator(session, repeat=True), n=15, task="t")  # type: ignore[arg-type]
    assert len(ds) == 5  # every call repeated the same five questions
    flaky = FakeGenerator(session, fail_every=2)
    assert len(synthetic.synthesize(flaky, n=20, task="other")) == 10  # type: ignore[arg-type]


def test_cost_bound_checked_before_any_call(session: Session, s3: Any) -> None:
    gen = FakeGenerator(session)
    with pytest.raises(CostLimitExceededError):
        synthetic.synthesize(gen, n=50, task="t", max_cost_usd=0.1)  # type: ignore[arg-type]
    assert gen.prompts == []


def test_input_checks(session: Session) -> None:
    gen = FakeGenerator(session)
    with pytest.raises(SkyhammerError, match="task description"):
        synthetic.synthesize(gen, n=5)  # type: ignore[arg-type]
    with pytest.raises(SkyhammerError, match="n must be"):
        synthetic.synthesize(gen, n=0, task="t")  # type: ignore[arg-type]


def test_chunking_and_call_plan() -> None:
    text = "\n\n".join("word " * 300 for _ in range(10))
    chunks = synthetic.chunk(text, tokens=1000)
    assert 2 <= len(chunks) <= 5 and "".join(chunks).count("word") == 3000
    assert synthetic.plan_calls(10, 5, []) == [None, None]
    assert synthetic.plan_calls(10, 5, ["a"]) == ["a", "a"]
    assert synthetic.plan_calls(5, 5, ["a", "b", "c"]) == ["a"]
    assert synthetic.plan_calls(10, 5, ["a", "b", "c", "d"]) == ["a", "c"]


def test_output_validates_as_sft(session: Session, s3: Any) -> None:
    from skyhammer.catalog import Technique, get_model

    ds = Dataset.synthesize(FakeGenerator(session), n=10, task="t")  # type: ignore[arg-type]
    assert ds.validate(get_model("nova-micro"), Technique.SFT).ok
    assert json.loads(json.dumps(ds._rows[0][1]))  # plain JSON rows
