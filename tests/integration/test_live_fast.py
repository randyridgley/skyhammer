"""Fast live checks: free or a fraction of a cent. Catches AWS drift (models, recipes, prices)."""

from __future__ import annotations

import pytest

from skyhammer.backends.sagemaker import SageMakerServerlessBackend
from skyhammer.catalog import CATALOG, BackendKind, ModelEntry, Technique, TrainingType
from skyhammer.hub import HubClient
from skyhammer.llm import LLM
from skyhammer.pricing import training_price
from skyhammer.router import resolve
from skyhammer.session import Session


def test_catalog_matches_bedrock(live_session: Session) -> None:
    """Every Bedrock-native SFT capability in us-east-1 is offered by Bedrock."""
    bedrock = live_session.client("bedrock", region="us-east-1")
    available = {
        m["modelId"]
        for m in bedrock.list_foundation_models(byCustomizationType="FINE_TUNING")["modelSummaries"]
    }
    for entry in CATALOG.values():
        for cap in entry.capabilities:
            if (
                cap.backend is BackendKind.BEDROCK
                and "us-east-1" in cap.regions
                and cap.technique is Technique.SFT
            ):
                assert cap.backend_model_id in available, cap.backend_model_id


def test_every_open_weight_model_has_hub_recipes(live_session: Session) -> None:
    hub = HubClient(live_session, "us-east-1")
    seen: set[str] = set()
    for entry in CATALOG.values():
        for cap in entry.capabilities:
            if cap.backend is not BackendKind.SAGEMAKER or cap.backend_model_id in seen:
                continue
            seen.add(cap.backend_model_id)
            model = hub.model(cap.backend_model_id)
            assert any(r.technique == "SFT" for r in model.recipes), cap.backend_model_id
            assert model.gated == entry.requires_eula, f"{entry.alias}: catalog EULA flag drifted"


def test_recipe_params_include_mapped_keys(live_session: Session) -> None:
    hub = HubClient(live_session, "us-east-1")
    recipe = hub.model("huggingface-reasoning-qwen3-06b").recipe("SFT", TrainingType.LORA)
    params = hub.params(recipe)
    for key in (
        "max_epochs",
        "learning_rate",
        "global_batch_size",
        "lora_rank",
        "lora_alpha",
        "lr_warmup_steps_ratio",
        "train_val_split_ratio",
        "merge_weights",
    ):
        assert key in params, key


@pytest.mark.parametrize("alias", ["nova-micro", "nova-lite", "qwen3-0.6b", "llama-3.1-8b"])
def test_training_prices_resolve(live_session: Session, alias: str) -> None:
    route = resolve(CATALOG[alias], Technique.SFT, "us-east-1")
    price = training_price(live_session, route, Technique.SFT, TrainingType.LORA)
    assert price.usd_per_1k_tokens and price.usd_per_1k_tokens > 0, price.usage_type


def test_dataset_limits_from_live_hub(live_session: Session) -> None:
    backend = SageMakerServerlessBackend(live_session)
    limits = backend.dataset_limits(CATALOG["llama-3.1-8b"], Technique.SFT, TrainingType.LORA)
    assert limits.max_tokens_per_sample >= 4096 and "recipes" in limits.source


def test_base_model_chat(live_session: Session) -> None:
    resp = LLM("nova-micro", session=live_session).chat.completions.create(
        messages=[{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=5
    )
    assert resp.choices[0].message.content.strip()
    assert resp.usage.total_tokens > 0


def test_catalog_serverless_ids_resolve(live_session: Session) -> None:
    """Drift: every fine-tunable model's serverless base ID still exists in Bedrock."""
    from skyhammer.serverless import ModelDirectory

    directory = ModelDirectory.for_session(live_session)
    for entry in CATALOG.values():
        if entry.inference_id:
            directory.find(entry.inference_id)  # raises with suggestions if Bedrock renamed it


_PUBLIC_HF = [e for e in CATALOG.values() if e.hf_repo and not e.requires_eula]


@pytest.mark.parametrize("entry", _PUBLIC_HF, ids=lambda e: e.alias)
def test_bedrock_import_supports_catalog_architectures(entry: ModelEntry) -> None:
    """Drift: models we import into Bedrock still use a Custom Model Import architecture, and
    their tokenizer (exact token counts, DA-6) still loads."""
    import json

    from huggingface_hub import hf_hub_download

    from skyhammer.imports import SUPPORTED_ARCHITECTURES
    from skyhammer.tokenization import tokenizer_counter

    assert entry.hf_repo
    with open(hf_hub_download(entry.hf_repo, "config.json")) as f:
        archs = set(json.load(f).get("architectures") or [])
    if entry.bedrock_serving_regions:
        assert archs & SUPPORTED_ARCHITECTURES, f"{entry.alias}: {archs} not importable"
    assert tokenizer_counter.__wrapped__(entry.hf_repo), f"{entry.alias}: tokenizer didn't load"
