from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skyhammer import config
from skyhammer.config import Settings
from skyhammer.session import Session


def chat(
    user: str = "What is 2+2?", assistant: str | None = "4", system: str | None = None
) -> dict[str, Any]:
    msgs: list[dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return {"messages": msgs}


@pytest.fixture(autouse=True)
def _isolated_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    if request.node.get_closest_marker("aws"):
        return  # integration tests use the real credential chain
    for var in ("AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "USER_CONFIG", tmp_path / "no-user-config.yaml")
    monkeypatch.setattr(config, "_current", None)
    from skyhammer import tokenization

    monkeypatch.setattr(tokenization, "tokenizer_counter", lambda repo: None)  # no HF downloads


@pytest.fixture
def settings() -> Settings:
    return Settings(
        region="us-east-1",
        bucket="skyhammer-test-bucket",
        kms_key_arn="arn:aws:kms:us-east-1:123456789012:key/abc",
        bedrock_role_arn="arn:aws:iam::123456789012:role/bedrock",
        sagemaker_role_arn="arn:aws:iam::123456789012:role/sagemaker",
        import_role_arn="arn:aws:iam::123456789012:role/import",
        grader_role_arn="arn:aws:iam::123456789012:role/grader",
        tags={"cost-center": "1234"},
    )


@pytest.fixture
def session(settings: Settings) -> Session:
    return Session(settings)


@pytest.fixture
def sft_file(tmp_path: Path) -> Path:
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(chat(f"q{i}", f"a{i}")) for i in range(100)) + "\n")
    return p


FIXTURES = Path(__file__).parent / "fixtures"


class FakeHub:
    """HubClient backed by real SageMaker Public Hub metadata captured for Qwen3 0.6B."""

    gated_models = ("llama",)  # Gemma 4 is Apache 2.0 (not gated) in the live Hub

    def __init__(self, session: Any, region: str, **_: Any) -> None:
        self.region = region

    def model(self, model_id: str) -> Any:
        from skyhammer.hub import parse_hub_model

        resp = json.loads((FIXTURES / "hub_qwen3_06b.json").read_text())
        resp["HubContentName"] = model_id
        if any(g in model_id for g in self.gated_models):
            doc = json.loads(resp["HubContentDocument"])
            doc["ModelAccess"] = "Gated"
            resp["HubContentDocument"] = json.dumps(doc)
        return parse_hub_model(resp)

    def params(self, recipe: Any) -> Any:
        from skyhammer.hub import parse_params

        if recipe.technique == "DPO":
            name = "dpo_lora"
        else:
            name = "sft_fft" if recipe.training_type.value == "full" else "sft_lora"
        return parse_params(json.loads((FIXTURES / f"recipe_params_{name}.json").read_text()))


@pytest.fixture(autouse=True)
def _fake_hub(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("aws"):
        return  # integration tests read the real SageMaker Public Hub
    monkeypatch.setattr("skyhammer.backends.sagemaker.HubClient", FakeHub)


@pytest.fixture(autouse=True)
def _offline_pricing(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("aws"):
        return
    from skyhammer import pricing

    monkeypatch.setattr(pricing, "_lookup", lambda session, service, usage: None)


@pytest.fixture(autouse=True)
def _fixture_model_directory(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    from skyhammer.serverless import ModelDirectory

    monkeypatch.setattr(ModelDirectory, "_cache", {})
    if request.node.get_closest_marker("aws"):
        return
    data = json.loads((FIXTURES / "bedrock_models_us_east_1.json").read_text())
    monkeypatch.setattr(ModelDirectory, "_load", lambda self: data)


@pytest.fixture(autouse=True)
def _fixture_inference_prices(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    from skyhammer.pricing import InferencePrices, parse_price_row

    monkeypatch.setattr(InferencePrices, "_cache", {})
    if request.node.get_closest_marker("aws"):
        return
    products = json.loads((FIXTURES / "price_products_sample.json").read_text())
    rows = [r for p in products if (r := parse_price_row(p))]
    monkeypatch.setattr(
        InferencePrices, "for_session", classmethod(lambda cls, session: InferencePrices(rows))
    )
