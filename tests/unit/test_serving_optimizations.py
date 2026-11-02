"""SV-14: quantization and speculative decoding on SageMaker endpoints (LMI vLLM)."""

import json
from typing import Any

import pytest

from skyhammer.backends import DeployOptions
from skyhammer.backends.sagemaker import SageMakerServerlessBackend
from skyhammer.errors import SkyhammerError, UnsupportedParameterError
from skyhammer.hub import HostingConfig
from skyhammer.jobs import ModelRef
from skyhammer.llm import LLM
from skyhammer.serving import EndpointDeployer, EndpointNames, Scaling, optimization_environment
from skyhammer.session import Session

from .test_endpoint_serving import HOSTING, FakeAWS

NGRAM = {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}


@pytest.fixture
def aws(session: Session, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {}
    monkeypatch.setattr(session, "client", lambda svc, region=None: FakeAWS(svc, state))
    return state


def test_options_become_lmi_environment() -> None:
    assert optimization_environment(HOSTING) == {}
    env = optimization_environment(HOSTING, quantize="fp8", speculative=NGRAM)
    assert env["OPTION_QUANTIZE"] == "fp8"
    assert json.loads(env["OPTION_SPECULATIVE_CONFIG"]) == NGRAM
    draft = {"method": "eagle3", "model": "s3://b/eagle3-head/", "num_speculative_tokens": 3}
    assert (
        "eagle3"
        in optimization_environment(HOSTING, speculative=draft)["OPTION_SPECULATIVE_CONFIG"]
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quantize": "awq"}, "choose fp8 or bitsandbytes"),
        ({"speculative": {"method": "medusa", "num_speculative_tokens": 2}}, "choose one of"),
        ({"speculative": {"method": "eagle3", "num_speculative_tokens": 2}}, "needs a draft"),
        ({"speculative": {"method": "ngram"}}, "num_speculative_tokens between 1 and 16"),
        (
            {
                "speculative": {"model": "org/draft", "num_speculative_tokens": 2},
                "network_isolation": True,
            },
            "network isolation, so it can't download 'org/draft'",
        ),
    ],
)
def test_bad_options_fail_before_deploying(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(SkyhammerError, match=message):
        optimization_environment(HOSTING, **kwargs)


def test_only_lmi_containers_take_them() -> None:
    tgi = HostingConfig(
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/tgi-inference:3.3.6",
        instance_type="ml.g6.2xlarge",
        gpus=1,
        memory_mb=8192,
    )
    with pytest.raises(SkyhammerError, match=r"LMI .* tgi-inference"):
        optimization_environment(tgi, quantize="fp8")


def test_model_gets_the_options_and_redeploys_must_match(
    session: Session, aws: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    created: dict[str, Any] = {}
    names = EndpointNames.for_deployment("support-fp8")
    deployer = EndpointDeployer(session, "us-east-1")
    original = deployer.sm.create_model

    def create_model(**kw: Any) -> Any:
        created.update(kw)
        return original(**kw)

    monkeypatch.setattr(deployer.sm, "create_model", create_model, raising=False)
    env = optimization_environment(HOSTING, quantize="fp8")
    deployer.deploy(
        names,
        model_dir="s3://b/m",
        hosting=HOSTING,
        environment=env,
        scaling=Scaling(),
        poll_seconds=0,
    )
    container_env = created["PrimaryContainer"]["Environment"]
    assert container_env["OPTION_QUANTIZE"] == "fp8"
    assert container_env["OPTION_ENTRYPOINT"].endswith("vllm_async_service")  # AWS's kept

    # The model exists now; deploying the same name without fp8 would silently keep it.
    monkeypatch.setattr(deployer, "_model_environment", lambda names: {**container_env})
    with pytest.raises(SkyhammerError, match=r"different serving options \(OPTION_QUANTIZE\)"):
        deployer.deploy(names, model_dir="s3://b/m", hosting=HOSTING, scaling=Scaling())
    deployer.deploy(  # same options: resumes
        names,
        model_dir="s3://b/m",
        hosting=HOSTING,
        environment=env,
        scaling=Scaling(),
        poll_seconds=0,
    )


def test_backend_passes_options_and_refuses_them_for_adapters(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        EndpointDeployer, "deploy", lambda self, names, **kw: seen.update(kw) or "arn:ep"
    )
    backend = SageMakerServerlessBackend(session)
    backend.deploy_base(
        "qwen3-0.6b", DeployOptions(name="b", tags={}, quantize="fp8", speculative=NGRAM)
    )
    assert seen["environment"]["OPTION_QUANTIZE"] == "fp8"
    ref = ModelRef(
        "lora",
        "s3://b/x",
        "sagemaker",
        "qwen3-0.6b",
        "us-east-1",
        api="openai_chat",
        serving="sagemaker",
        artifacts_uri="s3://b/x/",
    )
    with pytest.raises(SkyhammerError, match="shared endpoint's settings"):
        backend.deploy(ref, DeployOptions(name="lora", on="shared", quantize="fp8"))


def test_bedrock_serving_refuses_them(session: Session) -> None:
    ref = ModelRef(
        "support-nova",
        "arn:aws:bedrock:us-east-1:1:custom-model/x",
        "bedrock",
        "nova-micro",
        "us-east-1",
    )
    with pytest.raises(UnsupportedParameterError, match="Bedrock serving doesn't expose"):
        LLM.from_model_ref(ref, session=session).deploy(quantize="fp8")
    with pytest.raises(UnsupportedParameterError, match="Bedrock serving doesn't expose"):
        LLM("qwen3-0.6b", session=session).deploy(type="provisioned", speculative=NGRAM)
