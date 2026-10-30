"""Model catalog: which base models can be customized how, where, and served where.

This is data, not logic. AWS support changes often; update the entries here (and the
golden test in tests/integration) rather than special-casing models elsewhere.
Verified against AWS documentation on 2026-10-29.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Technique(StrEnum):
    SFT = "sft"
    DISTILLATION = "distillation"
    DPO = "dpo"
    RFT = "rft"
    CPT = "cpt"


class BackendKind(StrEnum):
    BEDROCK = "bedrock"
    SAGEMAKER = "sagemaker"


class TrainingType(StrEnum):
    LORA = "lora"
    FULL = "full"


class ServingTarget(StrEnum):
    BEDROCK = "bedrock"  # on-demand custom model deployment or Custom Model Import
    SAGEMAKER = "sagemaker"  # SageMaker AI real-time endpoint (SageMaker-trained weights only)


SAGEMAKER_HUB_MODEL_ARN = (
    "arn:aws:sagemaker:{region}:aws:hub-content/SageMakerPublicHub/Model/{model_id}"
)


class Capability(BaseModel):
    """One way to customize a model: a backend, a technique, and where it runs."""

    backend: BackendKind
    technique: Technique
    regions: frozenset[str]
    training_types: frozenset[TrainingType] = frozenset({TrainingType.LORA})
    # The backend's own identifier for the base model.
    backend_model_id: str
    # Bedrock-native only: whether the result can use on-demand deployment. Some native
    # customizations (e.g. Llama 3.1 8B) can only be served with provisioned throughput.
    on_demand: bool = True


class DatasetLimits(BaseModel):
    min_samples: int = 8
    max_samples: int = 20_000
    max_tokens_per_sample: int = 32_000
    # Strings that make training fail if they appear anywhere in prompts (Nova).
    reserved_terms: tuple[str, ...] = ()
    source: str = "generic defaults"


# Amazon Nova Understanding models (Nova user guide, "Dataset constraints", 2026-10-29).
# Nova 2 Lite isn't in that table; it uses the same conversation schema, so we apply the
# same limits until documented otherwise.
NOVA_LIMITS = DatasetLimits(
    min_samples=8,
    max_samples=20_000,
    max_tokens_per_sample=32_768,
    reserved_terms=("User:", "Bot:", "Assistant:", "System:", "<image>", "<video>", "[EOS]"),
    source="Amazon Nova dataset constraints",
)


class ModelEntry(BaseModel):
    alias: str
    family: str
    display_name: str
    capabilities: list[Capability]
    # Regions where the customized model can be served by Bedrock (on-demand deployment or
    # Custom Model Import). Empty means Bedrock can't serve it; SageMaker-trained weights can
    # still be served from a SageMaker endpoint.
    bedrock_serving_regions: frozenset[str] = frozenset()
    # Gated licenses (e.g. Llama, Gemma) need explicit accept_eula=True on SageMaker.
    # TODO(M1): read gating from DescribeHubContent instead of hard-coding.
    requires_eula: bool = False
    # Parameter count in billions; sizes SageMaker endpoint instances. None for Nova.
    params_b: float | None = None
    # Bedrock model ID for serverless calls to the *base* model (base-vs-tuned evals);
    # routing to an inference profile happens in serverless.ModelDirectory.
    inference_id: str | None = None
    # Hugging Face repo with the model's tokenizer.json, for exact token counts (DA-6).
    hf_repo: str | None = None
    limits: DatasetLimits = Field(default_factory=DatasetLimits)

    def capabilities_for(self, technique: Technique) -> list[Capability]:
        return [c for c in self.capabilities if c.technique == technique]


_E1 = frozenset({"us-east-1"})
_W2 = frozenset({"us-west-2"})
# Custom Model Import regions.
_CMI = frozenset({"us-east-1", "us-east-2", "us-west-2", "eu-central-1"})
_BOTH = frozenset({TrainingType.LORA, TrainingType.FULL})
_LORA = frozenset({TrainingType.LORA})


def _bedrock(
    technique: Technique, model_id: str, regions: frozenset[str] = _E1, *, on_demand: bool = True
) -> Capability:
    return Capability(
        backend=BackendKind.BEDROCK,
        technique=technique,
        regions=regions,
        backend_model_id=model_id,
        on_demand=on_demand,
    )


def _sm(
    techniques: list[Technique], model_id: str, training_types: frozenset[TrainingType] = _BOTH
) -> list[Capability]:
    # TODO(M1): confirm SageMaker serverless customization regions; assumed to match the
    # Custom Model Import regions until verified.
    return [
        Capability(
            backend=BackendKind.SAGEMAKER,
            technique=t,
            regions=_CMI,
            training_types=training_types,
            backend_model_id=model_id,
        )
        for t in techniques
    ]


_SM_ALL = [Technique.SFT, Technique.DPO, Technique.RFT]

CATALOG: dict[str, ModelEntry] = {
    e.alias: e
    for e in [
        # ---- Amazon Nova (Bedrock-native) ----
        ModelEntry(
            alias="nova-micro",
            limits=NOVA_LIMITS,
            inference_id="amazon.nova-micro-v1:0",
            family="nova",
            display_name="Amazon Nova Micro",
            capabilities=[
                _bedrock(Technique.SFT, "amazon.nova-micro-v1:0:128k"),
                _bedrock(Technique.DISTILLATION, "amazon.nova-micro-v1:0:128k"),
            ],
            bedrock_serving_regions=_E1,
        ),
        ModelEntry(
            alias="nova-lite",
            limits=NOVA_LIMITS,
            inference_id="amazon.nova-lite-v1:0",
            family="nova",
            display_name="Amazon Nova Lite",
            capabilities=[
                _bedrock(Technique.SFT, "amazon.nova-lite-v1:0:300k"),
                _bedrock(Technique.DISTILLATION, "amazon.nova-lite-v1:0:300k"),
            ],
            bedrock_serving_regions=_E1,
        ),
        ModelEntry(
            alias="nova-pro",
            limits=NOVA_LIMITS,
            inference_id="amazon.nova-pro-v1:0",
            family="nova",
            display_name="Amazon Nova Pro",
            capabilities=[
                _bedrock(Technique.SFT, "amazon.nova-pro-v1:0:300k"),
                _bedrock(Technique.DISTILLATION, "amazon.nova-pro-v1:0:300k"),
            ],
            bedrock_serving_regions=_E1,
        ),
        ModelEntry(
            alias="nova-2-lite",
            limits=NOVA_LIMITS,
            inference_id="amazon.nova-2-lite-v1:0",
            family="nova",
            display_name="Amazon Nova 2 Lite",
            capabilities=[
                _bedrock(Technique.SFT, "amazon.nova-2-lite-v1:0:256k"),
                _bedrock(Technique.RFT, "amazon.nova-2-lite-v1:0:256k"),
            ],
            bedrock_serving_regions=_E1,
        ),
        # ---- Open-weight: Bedrock-native in us-west-2, SageMaker + import elsewhere ----
        ModelEntry(
            alias="llama-3.1-8b",
            hf_repo="meta-llama/Llama-3.1-8B-Instruct",
            inference_id="meta.llama3-1-8b-instruct-v1:0",
            params_b=8,
            family="llama",
            display_name="Llama 3.1 8B Instruct",
            capabilities=[
                _bedrock(
                    Technique.SFT, "meta.llama3-1-8b-instruct-v1:0:128k", _W2, on_demand=False
                ),
                *_sm(_SM_ALL, "meta-textgeneration-llama-3-1-8b-instruct", _LORA),
            ],
            bedrock_serving_regions=_CMI,
            requires_eula=True,
        ),
        ModelEntry(
            alias="llama-3.3-70b",
            hf_repo="meta-llama/Llama-3.3-70B-Instruct",
            inference_id="meta.llama3-3-70b-instruct-v1:0",
            params_b=70,
            family="llama",
            display_name="Llama 3.3 70B Instruct",
            capabilities=[
                _bedrock(Technique.SFT, "meta.llama3-3-70b-instruct-v1:0:128k", _W2),
                *_sm(_SM_ALL, "meta-textgeneration-llama-3-3-70b-instruct"),
            ],
            bedrock_serving_regions=_CMI,
            requires_eula=True,
        ),
        *[
            ModelEntry(
                alias=f"qwen3-{size}",
                hf_repo=f"Qwen/Qwen3-{size.upper()}",
                inference_id="qwen.qwen3-32b-v1:0" if size == "32b" else None,
                params_b=float(size.removesuffix("b")),
                family="qwen",
                display_name=f"Qwen3 {size.upper()}",
                capabilities=[
                    *([_bedrock(Technique.RFT, "qwen.qwen3-32b", _W2)] if size == "32b" else []),
                    *_sm(_SM_ALL, f"huggingface-reasoning-qwen3-{sm_size}"),
                ],
                bedrock_serving_regions=_CMI,
            )
            for size, sm_size in [
                ("0.6b", "06b"),
                ("1.7b", "1-7b"),
                ("4b", "4b"),
                ("8b", "8b"),
                ("14b", "14b"),
                ("32b", "32b"),
            ]
        ],
        ModelEntry(
            alias="qwen2.5-7b",
            hf_repo="Qwen/Qwen2.5-7B-Instruct",
            params_b=7,
            family="qwen",
            display_name="Qwen2.5 7B Instruct",
            capabilities=_sm(_SM_ALL, "huggingface-llm-qwen2-5-7b-instruct"),
            bedrock_serving_regions=_CMI,
        ),
        ModelEntry(
            alias="deepseek-r1-distill-llama-8b",
            hf_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            params_b=8,
            family="deepseek",
            display_name="DeepSeek R1 Distill Llama 8B",
            capabilities=_sm(_SM_ALL, "deepseek-llm-r1-distill-llama-8b"),
            bedrock_serving_regions=_CMI,
        ),
        ModelEntry(
            alias="gpt-oss-20b",
            hf_repo="openai/gpt-oss-20b",
            inference_id="openai.gpt-oss-20b-1:0",
            params_b=21,
            family="gpt-oss",
            display_name="GPT-OSS 20B",
            capabilities=[
                _bedrock(Technique.RFT, "openai.gpt-oss-20b", _W2),
                *_sm(_SM_ALL, "openai-reasoning-gpt-oss-20b"),
            ],
            bedrock_serving_regions=_E1,  # GPT-OSS import is us-east-1 only
        ),
        # Not importable into Bedrock (architecture unsupported): served from SageMaker.
        ModelEntry(
            alias="gemma-4-31b",
            params_b=31,
            family="gemma",
            display_name="Gemma 4 31B",
            capabilities=_sm(_SM_ALL, "huggingface-vlm-gemma-4-31b-it"),
            bedrock_serving_regions=frozenset(),
        ),
    ]
}


def get_model(alias_or_id: str) -> ModelEntry:
    """Look up by SDK alias, or by any backend model ID."""
    from skyhammer.errors import UnsupportedRouteError

    if alias_or_id in CATALOG:
        return CATALOG[alias_or_id]
    for entry in CATALOG.values():
        if any(c.backend_model_id == alias_or_id for c in entry.capabilities):
            return entry
    raise UnsupportedRouteError(
        f"Unknown model {alias_or_id!r}.",
        alternatives=[f"{a} ({e.display_name})" for a, e in sorted(CATALOG.items())],
    )
