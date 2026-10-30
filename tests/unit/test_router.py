import pytest

from skyhammer.catalog import BackendKind, ServingTarget, Technique, TrainingType, get_model
from skyhammer.errors import UnsupportedRouteError
from skyhammer.router import resolve


def test_nova_routes_to_bedrock_in_us_east_1() -> None:
    route = resolve(get_model("nova-micro"), Technique.SFT, "us-east-1")
    assert route.backend is BackendKind.BEDROCK
    assert route.serving is ServingTarget.BEDROCK
    assert route.capability.backend_model_id == "amazon.nova-micro-v1:0:128k"
    assert not route.cross_region


def test_open_weight_routes_to_sagemaker_then_bedrock_in_us_east_1() -> None:
    route = resolve(get_model("llama-3.1-8b"), Technique.SFT, "us-east-1")
    assert (route.backend, route.serving) == (BackendKind.SAGEMAKER, ServingTarget.BEDROCK)
    assert route.region == "us-east-1"


def test_home_region_preferred_over_cross_region_native() -> None:
    # Bedrock-native Qwen3 32B RFT exists in us-west-2, but SageMaker in the home region wins.
    route = resolve(get_model("qwen3-32b"), Technique.RFT, "us-east-1")
    assert (route.backend, route.region) == (BackendKind.SAGEMAKER, "us-east-1")


def test_native_on_demand_preferred_in_us_west_2() -> None:
    route = resolve(get_model("llama-3.3-70b"), Technique.SFT, "us-west-2")
    assert route.backend is BackendKind.BEDROCK


def test_native_without_on_demand_falls_back_to_sagemaker_import() -> None:
    # Native Llama 3.1 8B fine-tunes need provisioned throughput; import is on-demand.
    route = resolve(get_model("llama-3.1-8b"), Technique.SFT, "us-west-2")
    assert (route.backend, route.serving) == (BackendKind.SAGEMAKER, ServingTarget.BEDROCK)


def test_cross_region_by_default_for_nova_from_us_west_2() -> None:
    route = resolve(get_model("nova-lite"), Technique.SFT, "us-west-2")
    assert route.region == "us-east-1"
    assert route.cross_region
    assert "cross-region from us-west-2" in route.describe()


def test_cross_region_disabled() -> None:
    with pytest.raises(UnsupportedRouteError) as e:
        resolve(get_model("nova-lite"), Technique.SFT, "us-west-2", cross_region=False)
    assert any("cross_region" in a for a in e.value.alternatives)


def test_allowed_regions_restricts_cross_region() -> None:
    with pytest.raises(UnsupportedRouteError):
        resolve(
            get_model("nova-lite"),
            Technique.SFT,
            "us-west-2",
            allowed_regions=["us-west-2", "eu-central-1"],
        )


def test_explicit_backend_override() -> None:
    route = resolve(
        get_model("llama-3.3-70b"), Technique.SFT, "us-west-2", backend=BackendKind.SAGEMAKER
    )
    assert route.backend is BackendKind.SAGEMAKER


def test_unsupported_technique() -> None:
    with pytest.raises(UnsupportedRouteError, match="does not support dpo"):
        resolve(get_model("nova-micro"), Technique.DPO, "us-east-1")


def test_not_importable_model_is_served_from_sagemaker() -> None:
    route = resolve(get_model("gemma-4-31b"), Technique.SFT, "us-east-1")
    assert (route.backend, route.serving) == (BackendKind.SAGEMAKER, ServingTarget.SAGEMAKER)
    assert route.region == "us-east-1"


def test_bedrock_serving_preferred_even_cross_region() -> None:
    # GPT-OSS imports are us-east-1 only: from us-east-2, serverless Bedrock serving in
    # us-east-1 beats an always-on SageMaker endpoint in the home region.
    route = resolve(get_model("gpt-oss-20b"), Technique.SFT, "us-east-2")
    assert (route.region, route.serving) == ("us-east-1", ServingTarget.BEDROCK)
    pinned = resolve(get_model("gpt-oss-20b"), Technique.SFT, "us-east-2", cross_region=False)
    assert (pinned.region, pinned.serving) == ("us-east-2", ServingTarget.SAGEMAKER)


def test_explicit_sagemaker_serving() -> None:
    route = resolve(
        get_model("qwen3-8b"), Technique.SFT, "us-east-1", serving=ServingTarget.SAGEMAKER
    )
    assert route.serving is ServingTarget.SAGEMAKER
    with pytest.raises(UnsupportedRouteError, match="No route"):
        resolve(
            get_model("nova-micro"), Technique.SFT, "us-east-1", serving=ServingTarget.SAGEMAKER
        )


def test_full_fine_tuning() -> None:
    with pytest.raises(UnsupportedRouteError, match="full fine-tuning unsupported"):
        resolve(
            get_model("llama-3.1-8b"), Technique.SFT, "us-east-1", training_type=TrainingType.FULL
        )
    route = resolve(
        get_model("qwen3-8b"), Technique.SFT, "us-east-1", training_type=TrainingType.FULL
    )
    assert route.backend is BackendKind.SAGEMAKER


def test_lookup_by_backend_model_id() -> None:
    assert get_model("amazon.nova-pro-v1:0:300k").alias == "nova-pro"
    with pytest.raises(UnsupportedRouteError, match="Unknown model"):
        get_model("nope")
