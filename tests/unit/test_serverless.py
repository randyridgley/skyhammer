import json
import re
from pathlib import Path

import pytest

from skyhammer.errors import SkyhammerError
from skyhammer.llm import LLM
from skyhammer.serverless import ModelDirectory, Routing, derive_alias, parse_directory
from skyhammer.session import Session
from tests.conftest import FIXTURES


def directory() -> dict[str, object]:
    return parse_directory(json.loads((FIXTURES / "bedrock_models_us_east_1.json").read_text()))


@pytest.mark.parametrize(
    "model_id, alias",
    [
        ("amazon.nova-micro-v1:0", "nova-micro"),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5"),
        ("anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("anthropic.claude-sonnet-5-5", "claude-sonnet-5-5"),
        ("meta.llama3-3-70b-instruct-v1:0", "llama3-3-70b-instruct"),
        ("qwen.qwen3-32b-v1:0", "qwen3-32b"),
        ("openai.gpt-oss-20b-1:0", "gpt-oss-20b"),
        ("deepseek.r1-v1:0", "deepseek-r1"),
        ("cohere.rerank-v3-5:0", "rerank-v3-5"),
        ("amazon.titan-embed-text-v2:0", "titan-embed-text-v2"),
        ("mistral.magistral-small-2509", "magistral-small-2509"),
    ],
)
def test_derive_alias(model_id: str, alias: str) -> None:
    assert derive_alias(model_id) == alias


def test_aliases_never_drop_version_numbers() -> None:
    """Regression: '-\\d+$' stripping once mapped claude-sonnet-5-5 to 'claude-sonnet-5'."""
    for mid in [
        "anthropic.claude-sonnet-4-6",
        "anthropic.claude-opus-4-7",
        "zai.glm-5",
        "anthropic.claude-fable-5-1",
        "cohere.rerank-v3-5:0",
        "moonshotai.kimi-k2.5",
    ]:
        rest = mid.split(".", 1)[1]
        keep = re.findall(r"\d+", re.sub(r":\d+$", "", rest))
        assert re.findall(r"\d+", derive_alias(mid)) == keep, mid


def test_colliding_aliases_are_dropped() -> None:
    data = {
        "modelSummaries": [
            {"modelId": "x.foo-v1:0", "inferenceTypesSupported": ["ON_DEMAND"]},
            {"modelId": "x.foo-v1", "inferenceTypesSupported": ["ON_DEMAND"]},
            {"modelId": "x.bar-v2:0", "inferenceTypesSupported": ["ON_DEMAND"]},
        ]
    }
    models = parse_directory(data)
    assert models["x.foo-v1:0"].alias is None and models["x.foo-v1"].alias is None
    assert models["x.bar-v2:0"].alias == "bar-v2"


def test_invocation_routing() -> None:
    d = directory()
    nova, claude, qwen = (
        d[k]
        for k in ("amazon.nova-micro-v1:0", "anthropic.claude-sonnet-4-6", "qwen.qwen3-32b-v1:0")
    )
    assert nova.invocation_id(Routing.GEO, "us-east-1") == "us.amazon.nova-micro-v1:0"
    assert nova.invocation_id(Routing.REGION, "us-east-1") == "amazon.nova-micro-v1:0"
    assert claude.invocation_id(Routing.GEO, "us-west-2") == "us.anthropic.claude-sonnet-4-6"
    assert claude.invocation_id(Routing.GLOBAL, "us-east-1") == "global.anthropic.claude-sonnet-4-6"
    with pytest.raises(SkyhammerError, match="cross_region is off"):
        claude.invocation_id(Routing.REGION, "us-east-1")
    assert qwen.invocation_id(Routing.GEO, "us-east-1") == "qwen.qwen3-32b-v1:0"  # in-region only


def test_kinds_and_context_variants_excluded() -> None:
    d = directory()
    assert d["amazon.titan-embed-text-v2:0"].kind == "embedding"
    assert d["cohere.rerank-v3-5:0"].kind == "rerank"
    assert d["anthropic.claude-sonnet-4-6"].kind == "chat"
    assert "amazon.nova-micro-v1:0:128k" not in d  # fine-tuning base id, not invocable


def test_find_by_alias_id_profile_and_suggestions(session: Session) -> None:
    md = ModelDirectory.for_session(session)
    assert md.find("Claude-Sonnet-4-6").model_id == "anthropic.claude-sonnet-4-6"
    assert md.find("us.anthropic.claude-sonnet-4-6").model_id == "anthropic.claude-sonnet-4-6"
    assert md.find("qwen.qwen3-32b-v1:0").alias == "qwen3-32b"
    with pytest.raises(SkyhammerError, match="Did you mean: claude-sonnet-4-6"):
        md.find("claude-sonet-4-6")


def test_llm_any_bedrock_model(session: Session) -> None:
    assert LLM("claude-sonnet-4-6", session=session).model_id == "us.anthropic.claude-sonnet-4-6"
    assert (
        LLM("claude-sonnet-4-6", session=session, routing="global").model_id
        == "global.anthropic.claude-sonnet-4-6"
    )
    # Fine-tunable catalog aliases map to their serverless base model.
    assert LLM("llama-3.3-70b", session=session).model_id == "us.meta.llama3-3-70b-instruct-v1:0"
    assert LLM("qwen3-32b", session=session).model_id == "qwen.qwen3-32b-v1:0"


def test_cross_region_off_means_region_routing(settings: object) -> None:
    s = Session(settings.model_copy(update={"cross_region": False}))  # type: ignore[attr-defined]
    assert LLM("nova-micro", session=s).model_id == "amazon.nova-micro-v1:0"
    with pytest.raises(SkyhammerError, match="cross_region is off"):
        _ = LLM("claude-sonnet-4-6", session=s).model_id


def test_serverless_only_models_cannot_be_fine_tuned(session: Session, sft_file: Path) -> None:
    from skyhammer.dataset import Dataset

    with pytest.raises(SkyhammerError, match="isn't in the fine-tuning catalog"):
        LLM("claude-sonnet-4-6", session=session).create_sft_job(
            Dataset.from_file(sft_file), output_model="x"
        )
