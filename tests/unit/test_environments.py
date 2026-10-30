from collections import Counter
from typing import Any

import pytest
from typer.testing import CliRunner

from skyhammer import Environment, TrafficSplit, registry
from skyhammer.cli.main import app
from skyhammer.environments import list_environments, parse
from skyhammer.errors import SkyhammerError
from skyhammer.inference import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    ChunkChoice,
    Delta,
    Message,
    Stream,
    Usage,
)
from skyhammer.llm import LLM
from skyhammer.traffic import normalize_weights, pick, sticky_point
from tests.unit.test_registry_cli import record, s3_session  # noqa: F401  (fixture)

MESSAGES = [{"role": "user", "content": "hi"}]


# ---- names and weights ----


def test_parse_environment_refs() -> None:
    assert parse("support-qwen@prod") == ("support-qwen", "prod")
    for bad in ("support-qwen", "support-qwen@", "@prod", "a@Prod", "a_b@prod", "a@b@c"):
        with pytest.raises(SkyhammerError, match="Invalid environment"):
            parse(bad)


def test_weights_normalize_and_reject_nonsense() -> None:
    assert normalize_weights({"a": 90, "b": 10}) == {"a": 0.9, "b": 0.1}
    for bad in ({}, {"a": -1, "b": 2}, {"a": 0}):
        with pytest.raises(SkyhammerError):
            normalize_weights(bad)


def test_pick_buckets_skip_zero_weights() -> None:
    w = normalize_weights({"a": 1, "off": 0, "b": 3})
    assert pick(w, 0.0) == "a"
    assert pick(w, 0.24) == "a"
    assert pick(w, 0.25) == "b"
    assert pick(w, 0.9999999) == "b"
    assert pick(w, 1.0) == "b"  # rounding never lands on a zero-weight model


def test_sticky_points_are_stable_uniform_and_salted() -> None:
    assert sticky_point("user-1") == sticky_point("user-1")
    assert sticky_point("user-1", "exp-a") != sticky_point("user-1", "exp-b")
    w = normalize_weights({"a": 90, "b": 10})
    share = Counter(pick(w, sticky_point(f"user-{i}")) for i in range(5000))
    assert 0.07 < share["b"] / 5000 < 0.13


# ---- traffic split ----


class FakeLLM:
    def __init__(self, name: str, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.model_id = f"arn:{name}"
        self.chat = self
        self.completions = self
        self.params: list[dict[str, Any]] = []

    def create(self, *, messages: Any, stream: bool = False, **params: Any) -> Any:
        self.params.append(params)
        if self.fail:
            raise RuntimeError("boom")
        usage = Usage(10, 2, 12, cost_usd=0.001)
        if stream:
            chunks = [
                ChatCompletionChunk("c", self.model_id, [ChunkChoice(0, Delta(content="ok"))]),
                ChatCompletionChunk("c", self.model_id, [], usage=usage),
            ]
            return Stream(self.model_id, iter(chunks))
        return ChatCompletion(
            "c", self.model_id, [Choice(0, Message("assistant", self.name), "stop")], usage
        )


def test_split_routes_by_weight_and_reports_the_variant() -> None:
    points = iter([0.1, 0.95, 0.5])
    split = TrafficSplit(
        {"v1": 90, "v2": 10},
        llms={"v1": FakeLLM("v1"), "v2": FakeLLM("v2")},
        rng=lambda: next(points),
    )
    served = [split.chat.completions.create(messages=MESSAGES).variant for _ in range(3)]
    assert served == ["v1", "v2", "v1"]
    assert split.stats["v1"].calls == 2
    assert split.stats["v2"].prompt_tokens == 10
    assert split.stats["v1"].cost_usd == pytest.approx(0.002)
    rows = {r["model"]: r for r in split.summary()}
    assert rows["v2"]["weight"] == pytest.approx(0.1)
    assert rows["v1"]["error_rate"] == 0


def test_sticky_key_pins_a_user_and_is_not_sent_to_the_model() -> None:
    llms = {"v1": FakeLLM("v1"), "v2": FakeLLM("v2")}
    split = TrafficSplit({"v1": 50, "v2": 50}, llms=llms)
    served = {
        split.chat.completions.create(messages=MESSAGES, sticky_key="user-42", max_tokens=5).variant
        for _ in range(10)
    }
    assert len(served) == 1
    sent = llms[served.pop()].params
    assert sent == [{"max_tokens": 5}] * 10


def test_split_counts_errors_and_reraises() -> None:
    split = TrafficSplit({"bad": 1}, llms={"bad": FakeLLM("bad", fail=True)})
    with pytest.raises(RuntimeError):
        split.chat.completions.create(messages=MESSAGES)
    assert split.stats["bad"].calls == 1
    assert split.stats["bad"].errors == 1
    assert split.summary()[0]["mean_latency_s"] is None


def test_split_streams_carry_the_variant_and_usage() -> None:
    split = TrafficSplit({"v2": 1}, llms={"v2": FakeLLM("v2")})
    s = split.chat.completions.create(messages=MESSAGES, stream=True)
    assert s.variant == "v2"
    final = s.final()
    assert final.variant == "v2"
    assert final.choices[0].message.content == "ok"
    assert split.stats["v2"].completion_tokens == 2


# ---- environments in the manifest store ----


def test_promote_split_rollback_and_history(s3_session: Any) -> None:  # noqa: F811
    session, _ = s3_session
    registry.save_job(session, record("support-v1", "support-v1-1"))
    registry.save_job(session, record("support-v2", "support-v2-1"))
    env = Environment("support@prod", session=session)
    assert not env.exists
    with pytest.raises(SkyhammerError, match="doesn't exist yet"):
        _ = env.targets
    with pytest.raises(SkyhammerError, match="no earlier version"):
        env.rollback()

    env.promote("support-v1", note="first release")
    env.split({"support-v1": 90, "support-v2": 10}, note="canary")
    env.promote("support-v2")
    assert env.targets == {"support-v2": 100}
    assert [h["targets"] for h in env.history] == [
        {"support-v1": 90, "support-v2": 10},
        {"support-v1": 100},
    ]
    assert env.history[0]["note"] == "canary"

    assert env.rollback() == {"support-v1": 90, "support-v2": 10}
    assert env.rollback() == {"support-v1": 100}
    assert env.history == []

    Environment("support@dev", session=session).promote("support-v2")
    listed = {e["ref"]: e["targets"] for e in list_environments(session)}
    assert listed == {"support@dev": {"support-v2": 100}, "support@prod": {"support-v1": 100}}
    Environment("support@dev", session=session).delete()
    assert [e["ref"] for e in list_environments(session)] == ["support@prod"]


def test_environments_refuse_deleted_models_and_nested_refs(s3_session: Any) -> None:  # noqa: F811
    session, _ = s3_session
    registry.save_job(session, record("old", "old-1"))
    registry.update_model(session, "old", deleted=True)
    env = Environment("support@prod", session=session)
    with pytest.raises(SkyhammerError, match="was deleted"):
        env.promote("old")
    with pytest.raises(SkyhammerError, match="not at other environments"):
        env.promote("support@dev")
    env.promote("nova-micro")  # base models are fine: checked when called


def test_llm_get_resolves_an_environment(s3_session: Any) -> None:  # noqa: F811
    session, _ = s3_session
    registry.save_job(session, record("support-v1", "support-v1-1"))
    registry.save_job(session, record("support-v2", "support-v2-1"))
    env = Environment("support@prod", session=session)
    env.promote("support-v1")
    llm = LLM.get("support@prod", session=session)
    assert llm.model_ref is not None and llm.model_ref.name == "support-v1"

    env.split({"support-v1": 90, "support-v2": 10})
    with pytest.raises(SkyhammerError, match="splits traffic"):
        LLM.get("support@prod", session=session)
    env.split({"support-v1": 0, "support-v2": 100})  # a zero weight doesn't count
    assert LLM.get("support@prod", session=session).model_ref.name == "support-v2"  # type: ignore[union-attr]

    split = env.traffic()
    assert split.weights == {"support-v1": 0.0, "support-v2": 1.0}
    assert split.salt == "support@prod"


def test_env_cli(s3_session: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    session, _ = s3_session
    monkeypatch.setattr("skyhammer.cli.main.Session", lambda: session)
    registry.save_job(session, record("support-v1", "support-v1-1"))
    registry.save_job(session, record("support-v2", "support-v2-1"))
    runner = CliRunner()

    out = runner.invoke(app, ["env", "promote", "support-v1", "support@prod"])
    assert out.exit_code == 0, out.output
    out = runner.invoke(
        app, ["env", "split", "support@prod", "support-v1=90", "support-v2=10", "--note", "canary"]
    )
    assert "support-v1 90%, support-v2 10%" in out.output
    out = runner.invoke(app, ["env", "show", "support@prod"])
    assert "canary" not in out.output.splitlines()[0]
    assert "support-v1 100%" in out.output  # history
    out = runner.invoke(app, ["env", "list"])
    assert "support@prod" in out.output
    out = runner.invoke(app, ["env", "rollback", "support@prod"])
    assert "support-v1 100%" in out.output
    out = runner.invoke(app, ["env", "split", "support@prod", "nonsense"])
    assert out.exit_code != 0
    out = runner.invoke(app, ["env", "delete", "support@prod", "--yes"])
    assert out.exit_code == 0
    assert list_environments(session) == []
