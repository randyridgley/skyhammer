"""Qwen 3.6: text fine-tuning on SageMaker, from real Hub metadata (2026-11-04)."""

import json
from pathlib import Path

import pytest

from skyhammer import LLM, Dataset
from skyhammer.backends.sagemaker import SageMakerServerlessBackend
from skyhammer.catalog import Technique, get_model
from skyhammer.errors import SkyhammerError, UnsupportedRouteError
from skyhammer.hub import parse_hub_model
from skyhammer.serving import optimization_environment
from skyhammer.session import Session

FIXTURE = Path(__file__).parent.parent / "fixtures" / "hub_qwen3_6_27b.json"


def rows(tmp_path: Path, n: int = 64) -> Dataset:
    p = tmp_path / "chat.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ]
                }
            )
            for i in range(n)
        )
    )
    return Dataset.from_file(p)


def test_base_hosting_from_ranked_inference_config() -> None:
    # No top-level HostingEcrUri: image and per-instance profiles live in AWS's ranked
    # inference configs, so the top-ranked one is used.
    base = parse_hub_model(json.loads(FIXTURE.read_text())).base
    assert base is not None
    assert base.artifact_uri.endswith(
        "/huggingface-vlm-qwen3-6-27b/artifacts/inference-prepack/v1.0.0/"
    )
    assert "/vllm:0.19.1-" in base.hosting.image_uri
    assert (base.hosting.instance_type, base.hosting.gpus) == ("ml.p5en.48xlarge", 8)
    assert sorted(base.variants) == [
        "ml.g7e.12xlarge",
        "ml.g7e.24xlarge",
        "ml.g7e.48xlarge",
        "ml.p5en.48xlarge",
    ]
    small = base.for_instance("ml.g7e.12xlarge")
    assert small.environment["SM_VLLM_TENSOR_PARALLEL_SIZE"] == "2"
    assert small.environment["SAGEMAKER_PROGRAM"] == "inference.py"  # container defaults kept
    assert base.network_isolation
    # fp8 and speculative decoding are LMI container options; this is AWS's vLLM container.
    with pytest.raises(SkyhammerError, match="LMI"):
        optimization_environment(base.hosting, quantize="fp8")


def test_sft_job_uses_the_verl_recipe(session: Session, tmp_path: Path) -> None:
    request, _ = LLM("qwen3.6-27b", session=session).plan(
        "sft", rows(tmp_path), output_model="q36", batch_size=16
    )
    assert request.route.serving.value == "sagemaker"  # Bedrock can't import it
    backend = SageMakerServerlessBackend(session)
    recipe, hp = backend.check(request)
    assert recipe is not None and recipe.name == "verl-sft-qwen-3-dot-6-27b-lora"
    assert recipe.sequence_length == "16K"
    # The verl recipe's own defaults and limits (not the Qwen3 ones).
    assert (hp["lora_rank"], hp["lora_alpha"], hp["global_batch_size"]) == ("32", "64", "16")
    assert "merge_weights" not in hp


def test_capabilities() -> None:
    big, moe = get_model("qwen3.6-27b"), get_model("qwen3.6-35b-a3b")
    assert {c.technique for c in big.capabilities} == {
        Technique.SFT,
        Technique.DPO,
        Technique.RFT,
        Technique.CUSTOM,
    }
    assert not big.bedrock_serving_regions and big.hf_repo == "Qwen/Qwen3.6-27B"
    # The MoE model has no DPO recipe in the Hub.
    assert not moe.capabilities_for(Technique.DPO)
    with pytest.raises(UnsupportedRouteError):
        LLM("qwen3.6-35b-a3b").plan("dpo", Dataset.from_records([], "x"), output_model="m")


def test_system_prompt_moves_into_the_first_user_turn(session: Session, tmp_path: Path) -> None:
    # Qwen 3.6's chat template, rendered turn by turn by verl, rejects a system message
    # on its own ("No user query found in messages", live 2026-11-04).
    p = tmp_path / "sys.jsonl"
    row = {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    }
    p.write_text("\n".join(json.dumps(row) for _ in range(40)))
    ds = Dataset.from_file(p)
    request, _ = LLM("qwen3.6-27b", session=session).plan(
        "sft", ds, output_model="q", batch_size=16
    )
    backend = SageMakerServerlessBackend(session)
    out = json.loads(ds._encode(backend, request.technique).splitlines()[0])
    assert out["messages"] == [
        {"role": "user", "content": "Be brief.\n\nHi"},
        {"role": "assistant", "content": "Hello"},
    ]
    # A new upload, so a new job (not a reattach to one trained on the old rows).
    assert (
        request.fingerprint()
        != request.__class__(**{**request.__dict__, "dataset": Dataset.from_file(p)}).fingerprint()
    )
    # Other models keep their system message.
    ds2 = Dataset.from_file(p)
    LLM("qwen3-0.6b", session=session).plan("sft", ds2, output_model="q2")
    assert (
        json.loads(ds2._encode(backend, request.technique).splitlines()[0])["messages"][0]["role"]
        == "system"
    )


def test_fold_system_keeps_content_parts() -> None:
    from skyhammer.formats.converters import fold_system

    row = {
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": [{"type": "text", "text": "U"}]},
        ]
    }
    assert fold_system(row)["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "S\n\n"}, {"type": "text", "text": "U"}],
        }
    ]
    plain = {"messages": [{"role": "user", "content": "U"}]}
    assert fold_system(plain) is plain
