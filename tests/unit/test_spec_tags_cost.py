from pathlib import Path

import pytest

from skyhammer.catalog import Technique
from skyhammer.config import Guardrails, load_settings
from skyhammer.cost import check_guardrails, estimate_training
from skyhammer.errors import ConfigError, CostLimitExceededError
from skyhammer.spec import load_spec
from skyhammer.tags import build_tags

SPEC = """
apiVersion: skyhammer/v1
kind: FineTuningJob
name: support-nova-v1
base_model: nova-micro
technique: sft
dataset: {path: data/train.jsonl, validation_split: 0.1}
hyperparameters: {epochs: 3, learning_rate: 0.00001}
guardrails: {max_cost_usd: 50}
"""


def test_load_spec_resolves_paths(tmp_path: Path) -> None:
    p = tmp_path / "jobs" / "job.yaml"
    p.parent.mkdir()
    p.write_text(SPEC)
    spec = load_spec(p)
    assert spec.technique is Technique.SFT
    assert spec.dataset.path == tmp_path / "jobs" / "data" / "train.jsonl"
    assert spec.hyperparameters["epochs"] == 3


@pytest.mark.parametrize(
    "patch, match",
    [
        ("technique: distillation", "teacher_model"),
        ("technique: rft", "grader"),
        ("bogus: 1", "Extra inputs"),
    ],
)
def test_spec_validation(tmp_path: Path, patch: str, match: str) -> None:
    body = SPEC.replace("technique: sft", patch) if "technique" in patch else SPEC + patch
    p = tmp_path / "job.yaml"
    p.write_text(body)
    with pytest.raises(ConfigError, match=match):
        load_spec(p)


def test_tags_sdk_wins_and_are_cleaned() -> None:
    tags = build_tags({"name": "m1"}, {"skyhammer:name": "evil", "team": "a,b"})
    assert tags["skyhammer:name"] == "m1"
    assert tags["team"] == "a_b"


def test_cost_guardrail() -> None:
    est = estimate_training(1_000_000, 2, "bedrock", "nova-micro", usd_per_1k_tokens=0.01)
    assert est.usd == pytest.approx(20.0)
    check_guardrails(est, Guardrails(max_cost_usd=25))
    with pytest.raises(CostLimitExceededError):
        check_guardrails(est, Guardrails(), max_cost_usd=10)


def test_default_epochs_noted() -> None:
    est = estimate_training(1000, None, "sagemaker", "qwen3-8b")
    assert est.training_tokens == 5000
    assert any("5 epochs" in n for n in est.notes)


def test_unknown_price_does_not_block() -> None:
    est = estimate_training(1000, 1, "bedrock", "nova-micro")
    assert est.usd is None
    check_guardrails(est, Guardrails(max_cost_usd=1))


def test_settings_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "skyhammer.yaml").write_text("region: us-west-2\nbucket: from-file\n")
    monkeypatch.setenv("SKYHAMMER_BUCKET", "from-env")
    s = load_settings(project_config=tmp_path / "skyhammer.yaml")
    assert (s.region, s.bucket) == ("us-west-2", "from-env")
    assert (
        load_settings(project_config=tmp_path / "skyhammer.yaml", region="eu-central-1").region
        == "eu-central-1"
    )


def test_pricing_usage_types() -> None:
    from skyhammer.catalog import Technique, TrainingType, get_model
    from skyhammer.pricing import usage_type
    from skyhammer.router import resolve

    nova = resolve(get_model("nova-micro"), Technique.SFT, "us-east-1")
    assert (
        usage_type(nova, Technique.SFT, TrainingType.LORA)
        == "USE1-NovaMicro-Customization-Training"
    )
    qwen = resolve(get_model("qwen3-0.6b"), Technique.SFT, "us-east-1")
    assert usage_type(qwen, Technique.SFT, TrainingType.FULL) == (
        "USE1-ServerlessTraining:FineTuning-SFT-FULLRANK-huggingface-reasoning-qwen3-06b"
    )


def test_price_units_parsed(monkeypatch: pytest.MonkeyPatch, session: object) -> None:
    from skyhammer import pricing
    from skyhammer.catalog import Technique, TrainingType, get_model
    from skyhammer.router import resolve

    route = resolve(get_model("qwen3-0.6b"), Technique.SFT, "us-east-1")
    monkeypatch.setattr(pricing, "_lookup", lambda *a: {"unit": "1M tokens", "usd": 0.375})
    p = pricing.training_price(session, route, Technique.SFT, TrainingType.LORA)  # type: ignore[arg-type]
    assert p.usd_per_1k_tokens == pytest.approx(0.000375)
    monkeypatch.setattr(pricing, "_lookup", lambda *a: {"unit": "Hours", "usd": 80.0})
    p = pricing.training_price(session, route, Technique.RFT, TrainingType.LORA)  # type: ignore[arg-type]
    assert p.usd_per_hour == 80.0


def test_uncapped_time_billed_estimate_shows_the_rate() -> None:
    from skyhammer.cost import CostEstimate

    est = CostEstimate(1000, None, usd_per_hour=80.0)
    assert est.usd is None
    assert str(est) == "billed by time at $80.00/hour, no runtime cap"
