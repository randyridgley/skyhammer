"""SageMaker endpoint serving: deployer orchestration, chat over sagemaker-runtime, streaming."""

import io
import json
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.validate import validate_parameters

from skyhammer.hub import HostingConfig
from skyhammer.inference import Chat, Stream, is_capacity_cold_start, sse_stream_chunks
from skyhammer.serving import EndpointDeployer, EndpointNames, Scaling, serving_environment
from skyhammer.session import Session

HOSTING = HostingConfig(
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.34.0-lmi16.0.0-cu128",
    instance_type="ml.g7.2xlarge",
    gpus=1,
    memory_mb=8192,
    cpus=2,
    environment={
        "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
        "OPTION_ENABLE_LORA": "true",
        "OPTION_MAX_LORAS": "4",
    },
)
NAMES = EndpointNames.for_deployment("support-gemma")


def _missing(op: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Could not find resource"}}, op
    )


class FakeAWS:
    """One object standing in for sagemaker / application-autoscaling / cloudwatch."""

    def __init__(self, service: str, state: dict[str, Any]) -> None:
        self.client = boto3.client(service, region_name="us-east-1")
        self.state = state

    def __getattr__(self, op: str) -> Any:
        def call(*args: Any, **kw: Any) -> Any:
            if op != "get_waiter":
                api = self.client.meta.method_to_api_mapping[op]
                validate_parameters(
                    kw, self.client.meta.service_model.operation_model(api).input_shape
                )
            self.state.setdefault("calls", []).append(op)
            res = self.state.setdefault("resources", set())
            if op.startswith("create_"):
                res.add(op.removeprefix("create_"))
                return {"EndpointArn": "arn:aws:sagemaker:us-east-1:1:endpoint/support-gemma"}
            if op.startswith("describe_"):
                kind = op.removeprefix("describe_")
                if kind not in res:
                    raise _missing(op)
                return {
                    "EndpointArn": "arn:aws:sagemaker:us-east-1:1:endpoint/support-gemma",
                    "InferenceComponentStatus": "InService",
                    "RuntimeConfig": {"CurrentCopyCount": 1},
                }
            if op.startswith("delete_") and op != "delete_alarms":
                res.discard(op.removeprefix("delete_"))
            if op == "put_scaling_policy":
                return {"PolicyARN": "arn:aws:autoscaling:us-east-1:1:scalingPolicy:x"}
            if op == "get_waiter":
                return type("W", (), {"wait": staticmethod(lambda **_: None)})()
            return {}

        return call


@pytest.fixture
def aws(session: Session, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {}
    monkeypatch.setattr(session, "client", lambda svc, region=None: FakeAWS(svc, state))
    return state


def test_merged_weights_drop_lora_env() -> None:
    env = serving_environment(HOSTING)
    assert "OPTION_ENABLE_LORA" not in env and "OPTION_MAX_LORAS" not in env
    assert env["OPTION_ENTRYPOINT"].endswith("vllm_async_service")


def test_deploy_creates_everything_validated_and_scales_to_zero(
    session: Session, aws: dict[str, Any]
) -> None:
    arn = EndpointDeployer(session, "us-east-1").deploy(
        NAMES,
        model_dir="s3://b/jobs/x/model/checkpoints/hf_merged",
        hosting=HOSTING,
        scaling=Scaling(),
        tags={"skyhammer:name": "support-gemma"},
        poll_seconds=0,
    )
    assert arn.endswith(":endpoint/support-gemma")
    calls = aws["calls"]
    order = [c for c in calls if c.startswith("create_")]
    assert order == [
        "create_model",
        "create_endpoint_config",
        "create_endpoint",
        "create_inference_component",
    ]
    assert calls.count("put_scaling_policy") == 2 and "put_metric_alarm" in calls  # from-zero


def test_deploy_resumes_without_recreating(session: Session, aws: dict[str, Any]) -> None:
    aws["resources"] = {"model", "endpoint_config", "endpoint"}  # interrupted mid-deploy
    EndpointDeployer(session, "us-east-1").deploy(
        NAMES, model_dir="s3://b/m", hosting=HOSTING, scaling=Scaling(min_copies=1), poll_seconds=0
    )
    assert [c for c in aws["calls"] if c.startswith("create_")] == ["create_inference_component"]
    assert "put_metric_alarm" not in aws["calls"]  # warm (min 1) needs no scale-from-zero


def test_delete_removes_in_dependency_order(session: Session, aws: dict[str, Any]) -> None:
    aws["resources"] = {"model", "endpoint_config", "endpoint", "inference_component"}
    EndpointDeployer(session, "us-east-1").delete(NAMES)
    deletes = [c for c in aws["calls"] if c.startswith(("delete_", "deregister_"))]
    assert deletes == [
        "deregister_scalable_target",
        "delete_alarms",
        "delete_inference_component",
        "delete_endpoint",
        "delete_endpoint_config",
        "delete_model",
    ]
    assert not aws["resources"]


class FakeSMRuntime:
    def __init__(self, cold: int = 0) -> None:
        self.cold = cold
        self.kw: dict[str, Any] = {}

    def invoke_endpoint(self, **kw: Any) -> dict[str, Any]:
        if self.cold:
            self.cold -= 1
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationError",
                        "Message": "Inference Component has no capacity to process this request.",
                    }
                },
                "InvokeEndpoint",
            )
        self.kw = kw
        return {
            "Body": io.BytesIO(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "<think>hm</think>hello",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                    }
                ).encode()
            )
        }


def test_chat_over_endpoint_with_scale_from_zero_retry() -> None:
    rt, sleeps = FakeSMRuntime(cold=2), []
    chat = Chat(
        rt, "support-gemma", api="sagemaker", component="support-gemma-ic", sleep=sleeps.append
    )
    r = chat.completions.create(messages=[{"role": "user", "content": "hi"}], max_tokens=5)
    assert rt.kw["InferenceComponentName"] == "support-gemma-ic"
    assert json.loads(rt.kw["Body"])["max_tokens"] == 5
    assert (r.choices[0].message.content, r.choices[0].message.reasoning_content) == ("hello", "hm")
    assert len(sleeps) == 2


def test_capacity_error_detection() -> None:
    def err(code: str, msg: str) -> ClientError:
        return ClientError({"Error": {"Code": code, "Message": msg}}, "InvokeEndpoint")

    assert is_capacity_cold_start(err("ValidationError", "Inference Component has no capacity"))
    assert not is_capacity_cold_start(err("ValidationError", "Invalid JSON body"))


def test_sse_stream_across_split_payloads() -> None:
    data = (
        b'data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"he"}}]}\n\n'
        b'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"llo"},'
        b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    )
    parts = [{"PayloadPart": {"Bytes": data[i : i + 17]}} for i in range(0, len(data), 17)]
    final = Stream("e", sse_stream_chunks("e", iter(parts))).final()
    assert final.choices[0].message.content == "hello" and final.choices[0].finish_reason == "stop"


def test_hub_base_hosting_parsed_from_real_metadata() -> None:
    from skyhammer.hub import parse_hub_model
    from tests.conftest import FIXTURES

    m = parse_hub_model(json.loads((FIXTURES / "hub_qwen3_06b.json").read_text()))
    assert m.base and m.base.artifact_uri.startswith("s3://jumpstart-cache-prod-us-east-1/")
    assert m.base.hosting.image_uri.endswith("lmi22.0.0-cu129")
    assert m.base.hosting.instance_type == "ml.g6e.2xlarge" and m.base.hosting.gpus == 1
    assert m.base.hosting.environment["HF_MODEL_ID"] == "/opt/ml/model"
    assert m.base.network_isolation is True


def test_base_model_request_uses_hub_access_and_eula() -> None:
    from skyhammer.serving import model_request

    req = model_request(
        NAMES,
        image_uri="img",
        environment={},
        model_dir="s3://js/prepack",
        role_arn="arn:aws:iam::1:role/r",
        tags=[],
        hub_content_arn="arn:aws:sagemaker:us-east-1:aws:hub-content/x",
        accept_eula=True,
        network_isolation=True,
    )
    src = req["PrimaryContainer"]["ModelDataSource"]["S3DataSource"]
    assert src["HubAccessConfig"]["HubContentArn"].endswith("hub-content/x")
    assert src["ModelAccessConfig"] == {"AcceptEula": True} and req["EnableNetworkIsolation"]
    client = boto3.client("sagemaker", region_name="us-east-1")
    validate_parameters(req, client.meta.service_model.operation_model("CreateModel").input_shape)


def test_deployment_types(session: Session, aws: dict[str, Any]) -> None:
    from skyhammer.errors import SkyhammerError
    from skyhammer.llm import LLM

    assert LLM("nova-micro", session=session).deploy().model_id == "us.amazon.nova-micro-v1:0"
    with pytest.raises(SkyhammerError, match="dedicated deployment"):
        _ = LLM("qwen3-8b", session=session, deployment_type="dedicated").model_id
    ded = LLM("qwen3-8b", session=session).deploy(name="qwen-ded")  # auto -> not serverless
    assert ded.deployment and ded.deployment.name == "qwen-ded"
    assert ded.model_ref and ded.model_ref.serving == "sagemaker"
    assert "create_model" in aws["calls"]
    with pytest.raises(SkyhammerError, match="deployment_type must be"):
        LLM("nova-micro", session=session, deployment_type="turbo")


def test_real_lmi_jsonlines_stream_parses() -> None:
    """Regression (live 2026-10-29): LMI streams JSON Lines split across PayloadParts."""
    import base64

    from tests.conftest import FIXTURES

    fx = json.loads((FIXTURES / "sagemaker_lmi_stream.json").read_text())
    assert fx["content_type"] == "application/jsonlines"
    parts = [{"PayloadPart": {"Bytes": base64.b64decode(p)}} for p in fx["parts_b64"]]
    stream = Stream("e", sse_stream_chunks("e", iter(parts)))
    chunks = list(stream)
    assert len(chunks) > 5
    final = stream.final()
    msg = final.choices[0].message
    assert msg.reasoning_content  # Qwen3's streamed <think> text is split out
    assert "<think>" not in (msg.content or "")


def test_base_deploy_omits_hub_access_config(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Public-hub models: CreateModel rejects HubAccessConfig (live, 2026-10-29)."""
    from skyhammer.backends import DeployOptions
    from skyhammer.backends.sagemaker import SageMakerServerlessBackend
    from skyhammer.serving import EndpointDeployer

    seen: dict = {}
    monkeypatch.setattr(
        EndpointDeployer, "deploy", lambda self, names, **kw: seen.update(kw) or "arn:ep"
    )
    SageMakerServerlessBackend(session).deploy_base("qwen3-0.6b", DeployOptions(name="b", tags={}))
    assert "hub_content_arn" not in seen
    assert seen["model_dir"].startswith("s3://jumpstart-cache-prod-us-east-1/")


def test_endpoint_failure_reports_sagemaker_reason(session: Session) -> None:
    """Seen live: botocore's WaiterError hid InsufficientInstanceCapacity."""
    from types import SimpleNamespace

    from botocore.exceptions import WaiterError

    from skyhammer.errors import JobFailedError

    def fail(**_: Any) -> None:
        raise WaiterError("EndpointInService", "terminal failure", {})

    reason = "Unable to provision requested ML compute capacity due to InsufficientInstanceCapacity"
    deployer = EndpointDeployer(session, "us-east-1")
    deployer.__dict__["sm"] = SimpleNamespace(
        get_waiter=lambda name: SimpleNamespace(wait=fail),
        describe_endpoint=lambda EndpointName: {
            "EndpointStatus": "Failed",
            "FailureReason": reason,
        },
    )
    with pytest.raises(JobFailedError, match="InsufficientInstanceCapacity"):
        deployer._wait_in_service(EndpointNames.for_deployment("x"))
