"""TR-12: multi-turn (agentic) RL, the job side (CreateJob) and the agent side (rollouts)."""

import io
import json
from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from botocore.validate import validate_parameters

from skyhammer import Dataset, iam
from skyhammer.agent_rl import Rollout, parse_job, rollout_handler
from skyhammer.backends.base import DeployOptions
from skyhammer.backends.sagemaker import SageMakerServerlessBackend
from skyhammer.config import Settings
from skyhammer.errors import (
    ConfigError,
    SkyhammerError,
    UnsupportedParameterError,
    UnsupportedRouteError,
)
from skyhammer.jobs import JobHandle, JobState
from skyhammer.llm import LLM
from skyhammer.session import Session

ACCOUNT = "123456789012"
RUNTIME = f"arn:aws:bedrock-agentcore:us-east-1:{ACCOUNT}:runtime/skyhammer_agent_wordle-AbC123"
MLFLOW = f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:mlflow-app/app-XYZ"
JOB_ARN = f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:job/AgentRFT/wordle-gptoss-0a1b2c3d"


@pytest.fixture
def agent_session(settings: Settings) -> Session:
    settings.agent_job_role_arn = f"arn:aws:iam::{ACCOUNT}:role/skyhammer-bootstrap-AgentJobRole"
    settings.mlflow_arn = MLFLOW
    return Session(settings)


def tasks(n: int = 64) -> Dataset:
    words = ["beach", "riser", "swoon", "dirty"]
    return Dataset.from_records(
        [{"prompt": "Guess the 5-letter word", "answer": words[i % 4], "id": i} for i in range(n)],
        "wordle",
    )


def valid(service: str, op: str, kwargs: dict[str, Any]) -> None:
    model = boto3.client(service, region_name="us-east-1").meta.service_model
    validate_parameters(kwargs, model.operation_model(op).input_shape)


class FakeSageMaker:
    def __init__(self) -> None:
        self.created: dict[str, Any] = {}
        self.groups: list[str] = []

    def describe_job(self, **kw: Any) -> dict[str, Any]:
        raise ClientError({"Error": {"Code": "ResourceNotFound", "Message": "nope"}}, "DescribeJob")

    def describe_model_package_group(self, ModelPackageGroupName: str) -> dict[str, str]:
        raise ClientError(
            {"Error": {"Code": "ValidationException", "Message": "does not exist"}}, "Describe"
        )

    def create_model_package_group(self, **kw: Any) -> dict[str, str]:
        valid("sagemaker", "CreateModelPackageGroup", kw)
        assert kw["ManagedConfiguration"] == {"ManagedStorageType": "Restricted"}
        self.groups.append(kw["ModelPackageGroupName"])
        return {
            "ModelPackageGroupArn": f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:model-package-group/"
            + kw["ModelPackageGroupName"]
        }

    def create_job(self, **kw: Any) -> dict[str, str]:
        valid("sagemaker", "CreateJob", kw)
        self.created = kw
        return {"JobArn": JOB_ARN}


def test_job_document(agent_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = LLM("gpt-oss-20b", session=agent_session)
    request, estimate = llm.plan(
        "agentic",
        tasks(),
        output_model="wordle-gptoss",
        agent=RUNTIME,
        batch_size=32,
        hyperparameters_extra={"max_steps": 20, "group_size": 8, "sampling_max_tokens": 8192},
    )
    assert estimate.usd is None and "billed per token" in estimate.notes[0]
    backend = SageMakerServerlessBackend(agent_session)
    fake = FakeSageMaker()
    backend.sagemaker = fake
    monkeypatch.setattr(Dataset, "upload", lambda self, b, t: "s3://bucket/datasets/wordle.jsonl")
    handle = backend.submit(request)
    assert handle.job_id == JOB_ARN

    kw = fake.created
    assert (kw["JobCategory"], kw["JobConfigSchemaVersion"]) == ("AgentRFT", "1.0.0")
    assert kw["RoleArn"].endswith("AgentJobRole")
    doc = json.loads(kw["JobConfigDocument"])
    assert doc["AgentConfig"] == {
        "BedrockAgentCoreConfig": {"AgentRuntimeArn": RUNTIME, "Qualifier": "DEFAULT"}
    }
    assert doc["InputDataConfig"] == [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://bucket/datasets/wordle.jsonl",
                }
            },
        }
    ]
    assert doc["OutputDataConfig"]["S3OutputPath"].endswith(f"/jobs/{request.job_name}/output/")
    assert doc["OutputDataConfig"]["KmsKeyArn"] == agent_session.settings.kms_key_arn
    assert fake.groups == ["wordle-gptoss", "wordle-gptoss-checkpoints"]
    assert doc["ModelPackageConfig"]["IntermediateCheckpointModelPackageGroupArn"].endswith(
        "/wordle-gptoss-checkpoints"
    )
    training = doc["TrainingConfig"]
    assert training["BaseModelArn"].endswith("/Model/openai-reasoning-gpt-oss-20b")
    assert training["MlflowConfig"] == {
        "MlflowResourceArn": MLFLOW,
        "MlflowExperimentName": "wordle-gptoss",
        "MlflowRunName": request.job_name,
    }
    # Only what you set, typed as the recipe declares (the service fills in the rest).
    assert training["HyperParameters"] == {
        "global_batch_size": 32,
        "max_steps": 20,
        "group_size": 8,
        "sampling_max_tokens": 8192,
    }
    assert {"Key": "skyhammer:technique", "Value": "agentic"} in kw["Tags"]


def test_rows_reach_the_agent_as_prompts(agent_session: Session) -> None:
    backend = SageMakerServerlessBackend(agent_session)
    from skyhammer.catalog import Technique

    assert backend.format_record(Technique.AGENTIC, {"prompt": "Guess"}) == {"prompt": "Guess"}
    row = {"prompt": "Guess", "answer": "beach"}
    assert json.loads(backend.format_record(Technique.AGENTIC, row)["prompt"]) == row


def test_checks_before_spending(agent_session: Session) -> None:
    llm = LLM("gpt-oss-20b", session=agent_session)

    def plan(data: Dataset | None = None, **kw: Any) -> Any:
        base: dict[str, Any] = {"agent": RUNTIME, "hyperparameters_extra": {"max_steps": 5}}
        return llm.plan("agentic", data or tasks(), output_model="m", **{**base, **kw})

    with pytest.raises(UnsupportedParameterError, match="Set max_steps"):
        plan(hyperparameters_extra={})
    with pytest.raises(UnsupportedParameterError, match=r"takes 128 tasks .* has 64"):
        plan()  # the recipe's default batch is 128
    with pytest.raises(UnsupportedParameterError, match=r"global_batch_size=48 not allowed"):
        plan(batch_size=48)
    with pytest.raises(UnsupportedParameterError, match="'warmup' isn't tunable"):
        plan(batch_size=32, hyperparameters_extra={"max_steps": 5, "warmup": 3})
    with pytest.raises(UnsupportedParameterError, match="billed per token, not by time"):
        plan(batch_size=32, max_hours=2)
    with pytest.raises(SkyhammerError, match="is in us-west-2 but the job runs in us-east-1"):
        plan(batch_size=32, agent=RUNTIME.replace("us-east-1", "us-west-2"))
    with pytest.raises(SkyhammerError, match=r"skyhammer_agent\* .* not 'wordle_agent'"):
        plan(
            batch_size=32,
            agent=f"arn:aws:bedrock-agentcore:us-east-1:{ACCOUNT}:runtime/wordle_agent-AbC123",
        )
    with pytest.raises(SkyhammerError, match="skyhammer-agent-"):
        plan(batch_size=32, agent=f"arn:aws:lambda:us-east-1:{ACCOUNT}:function:wordle")
    with pytest.raises(SkyhammerError, match="AgentCore runtime ARN"):
        plan(batch_size=32, agent="my-agent")
    with pytest.raises(SkyhammerError, match="needs agent="):
        plan(batch_size=32, agent=None)
    with pytest.raises(UnsupportedParameterError, match="max_cost_usd can't be enforced"):
        llm.create_agent_rl_job(
            tasks(), agent=RUNTIME, output_model="m", max_steps=5, max_cost_usd=10
        )
    with pytest.raises(UnsupportedRouteError):  # no multi-turn RL route for Nova Micro
        LLM("nova-micro", session=agent_session).plan(
            "agentic", tasks(), output_model="m", agent=RUNTIME
        )
    lam = f"arn:aws:lambda:us-east-1:{ACCOUNT}:function:skyhammer-agent-wordle"
    request, _ = plan(batch_size=32, agent=lam)
    assert request.agent == lam
    # The agent is part of what makes a job: another agent, another job.
    assert plan(batch_size=32)[0].job_name != request.job_name


def test_submit_needs_mlflow_app_and_role(settings: Settings, monkeypatch: Any) -> None:
    request, _ = LLM("gpt-oss-20b", session=Session(settings)).plan(
        "agentic",
        tasks(),
        output_model="m",
        agent=RUNTIME,
        batch_size=32,
        hyperparameters_extra={"max_steps": 5},
    )
    monkeypatch.setattr(Dataset, "upload", lambda self, b, t: "s3://b/k.jsonl")
    with pytest.raises(SkyhammerError, match="bootstrap AgentJobRole"):
        SageMakerServerlessBackend(Session(settings)).submit(request)
    settings.agent_job_role_arn = "arn:aws:iam::1:role/x"
    with pytest.raises(ConfigError, match="SageMaker MLflow App"):
        SageMakerServerlessBackend(Session(settings)).submit(request)
    settings.mlflow_arn = f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:mlflow-tracking-server/old"
    with pytest.raises(ConfigError, match="isn't an MLflow App ARN"):
        SageMakerServerlessBackend(Session(settings)).submit(request)


def describe_job(status: str, secondary: str, service: dict[str, Any] | None = None) -> dict:
    doc = {"AgentConfig": {}, **({"ServiceOutput": service} if service else {})}
    return {
        "JobName": "wordle-gptoss-0a1b2c3d",
        "JobArn": JOB_ARN,
        "RoleArn": "arn:aws:iam::1:role/x",
        "JobCategory": "AgentRFT",
        "JobConfigSchemaVersion": "1.0.0",
        "JobConfigDocument": json.dumps(doc),
        "CreationTime": 0,
        "LastModifiedTime": 0,
        "JobStatus": status,
        "SecondaryStatus": secondary,
        "SecondaryStatusTransitions": [],
    }


def test_status_reports_progress_tokens_and_output(agent_session: Session) -> None:
    info = parse_job(describe_job("InProgress", "Downloading"))
    assert info.state == "pending"
    info = parse_job(
        describe_job(
            "InProgress",
            "Training",
            {
                "ProgressInfo": {"MaxSteps": 20, "CurrentStep": 7},
                "BillableTokenUsage": {
                    "PrefillTokenCount": 4_916_168,
                    "SampleTokenCount": 1_597_370,
                    "TrainTokenCount": 6_433_085,
                },
            },
        )
    )
    assert info.state == "training"
    assert info.message == (
        "Training: step 7/20; billed tokens: prefill 4,916,168, sample 1,597,370, train 6,433,085"
    )

    class SM:
        def describe_job(self, **kw: Any) -> dict[str, Any]:
            assert kw == {"JobName": "wordle-gptoss-0a1b2c3d", "JobCategory": "AgentRFT"}
            pkg = f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:model-package/wordle-gptoss/1"
            return describe_job("Completed", "Completed", {"OutputModelPackageArn": pkg})

        def stop_job(self, **kw: Any) -> None:
            self.stopped = kw

    backend = SageMakerServerlessBackend(agent_session)
    backend.sagemaker = sm = SM()
    handle = JobHandle("sagemaker", "wordle-gptoss-0a1b2c3d", JOB_ARN, "us-east-1")
    st = backend.describe(handle)
    assert st.state is JobState.COMPLETED
    pkg = f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:model-package/wordle-gptoss/1"
    # A restricted model package: the ARN is the output, with no S3 weights to import.
    assert st.output_artifacts == pkg
    assert "/aws/sagemaker/Job/AgentRFT" in (st.log_url or "")
    record: Any = SimpleNamespace(
        handle=handle,
        job_id=JOB_ARN,
        output_model="wordle-gptoss",
        base_model="gpt-oss-20b",
        serving="bedrock",
    )
    assert backend.poll(record).state is JobState.COMPLETED  # no Bedrock import attempted
    ref = backend.finalize(record)
    assert (ref.model_arn, ref.artifacts_uri, ref.serving) == (pkg, None, "sagemaker")
    with pytest.raises(SkyhammerError, match="restricted model package"):
        backend.deploy(ref, DeployOptions(name="wordle-gptoss", tags={}))
    backend.cancel(handle)
    assert sm.stopped == {"JobName": "wordle-gptoss-0a1b2c3d", "JobCategory": "AgentRFT"}


# ---- the agent side ----


def payload(**extra: Any) -> dict[str, Any]:
    """The shape the training service sends an AgentCore runtime (AWS's Wordle sample)."""
    return {
        "prompt": json.dumps({"prompt": "Guess the 5-letter word", "answer": "beach"}),
        "metadata": {"jobArn": JOB_ARN, "rolloutId": "r1"},
        "inferenceParams": {"temperature": 1.0, "maxTokens": 4096},
        **extra,
    }


COMPLETION = {
    "id": "c1",
    "model": "default",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "guess_word", "arguments": '{"guess": "crane"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
}


def runtime() -> tuple[Any, Stubber]:
    client = boto3.client(
        "sagemakerjobruntime",
        region_name="us-east-1",
        aws_access_key_id="x",
        aws_secret_access_key="y",
    )
    return client, Stubber(client)


def test_rollout_samples_the_policy_and_reports_the_reward() -> None:
    client, stub = runtime()
    tools = [{"type": "function", "function": {"name": "guess_word", "parameters": {}}}]
    expected_body = {
        "model": "default",
        "temperature": 1.0,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Guess the 5-letter word"}],
        "tools": tools,
    }
    stub.add_response(
        "sample",
        {"Body": json.dumps(COMPLETION).encode(), "ContentType": "application/json"},
        {"JobArn": JOB_ARN, "TrajectoryId": "r1", "Body": json.dumps(expected_body)},
    )
    stub.add_response(
        "complete_rollout", {}, {"JobArn": JOB_ARN, "TrajectoryId": "r1", "Status": "ready"}
    )
    stub.add_response(
        "update_reward", {}, {"JobArn": JOB_ARN, "TrajectoryId": "r1", "Rewards": [1.35]}
    )

    @rollout_handler
    def play(rollout: Rollout) -> float:
        assert rollout.task["answer"] == "beach"
        reply = rollout.chat(
            messages=[{"role": "user", "content": rollout.task["prompt"]}], tools=tools
        )
        call = reply.choices[0].message.tool_calls[0]  # type: ignore[index]
        assert json.loads(call.function.arguments) == {"guess": "crane"}
        return 1.35

    original = Rollout.__init__

    def with_client(self: Rollout, p: dict[str, Any], *, client: Any = None) -> None:
        original(self, p, client=client or client_ref)

    client_ref = client
    with stub, pytest.MonkeyPatch.context() as mp:
        mp.setattr(Rollout, "__init__", with_client)
        assert play(payload()) == {"reward": 1.35}
        stub.assert_no_pending_responses()


def test_failed_rollouts_report_zero_and_reraise() -> None:
    client, stub = runtime()
    stub.add_response(
        "complete_rollout", {}, {"JobArn": JOB_ARN, "TrajectoryId": "r1", "Status": "failed"}
    )
    stub.add_response(
        "update_reward", {}, {"JobArn": JOB_ARN, "TrajectoryId": "r1", "Rewards": [0.0]}
    )

    def boom(rollout: Rollout) -> float:
        rollout._client = client
        raise RuntimeError("environment crashed")

    with stub, pytest.raises(RuntimeError, match="environment crashed"):
        rollout_handler(boom)(payload())
    stub.assert_no_pending_responses()


def test_rollout_payload_shapes() -> None:
    # The SDK's rollout-server shape: instance + snake_case metadata.
    r = Rollout(
        {
            "instance": {"prompt": "Plan a trip"},
            "metadata": {"job_arn": JOB_ARN, "trajectory_id": "t9", "region": "us-west-2"},
            "inference_params": {"top_p": 0.9},
        }
    )
    assert (r.prompt, r.task, r.trajectory_id, r.region, r.sampling) == (
        "Plan a trip",
        "Plan a trip",
        "t9",
        "us-west-2",
        {"top_p": 0.9},
    )
    assert Rollout(payload()).region == "us-east-1"  # from the job ARN
    with pytest.raises(SkyhammerError, match="no job ARN"):
        Rollout({"prompt": "x", "metadata": {}})
    body = io.BytesIO(json.dumps(COMPLETION).encode())

    class Client:
        def sample(self, **kw: Any) -> dict[str, Any]:
            self.body = json.loads(kw["Body"])
            return {"Body": body}

    c = Client()
    reply = Rollout(payload(), client=c).chat(messages=[], temperature=0.0)
    assert c.body["temperature"] == 0.0  # yours wins over the job's
    assert reply.usage.total_tokens == 52


def test_iam_policy_and_bootstrap_role() -> None:
    import yaml

    from skyhammer.bootstrap import template_body

    policy = iam.agent_rl_policy(ACCOUNT)
    by_sid = {s["Sid"]: s for s in policy["Statement"]}
    assert by_sid["AgentRLJobs"]["Action"] == [
        "sagemaker:CreateJob",
        "sagemaker:DescribeJob",
        "sagemaker:StopJob",
    ]
    assert by_sid["PassAgentJobRole"]["Condition"] == {
        "StringEquals": {"iam:PassedToService": "job.sagemaker.amazonaws.com"}
    }
    assert by_sid["CheckAgents"]["Resource"].endswith(":runtime/skyhammer_agent*")
    assert len(json.dumps(policy)) < 6144

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor(
        "!",
        lambda loader, suffix, node: {
            suffix: loader.construct_sequence(node)
            if isinstance(node, yaml.SequenceNode)
            else loader.construct_scalar(node)
            if isinstance(node, yaml.ScalarNode)
            else loader.construct_mapping(node)
        },
    )
    t = yaml.load(template_body(), Loader=Loader)
    role = t["Resources"]["AgentJobRole"]["Properties"]
    trust = role["AssumeRolePolicyDocument"]["Statement"][0]
    assert trust["Principal"] == {"Service": "job.sagemaker.amazonaws.com"}
    assert trust["Action"] == ["sts:AssumeRole", "sts:TagSession"]
    # Scoped to Skyhammer's bucket, not AmazonSageMakerJobFullAccess's every bucket.
    assert role["ManagedPolicyArns"] == [{"Ref": "DataAccessPolicy"}]
    statements = role["Policies"][0]["PolicyDocument"]["Statement"]
    invoke = next(st for st in statements if "bedrock-agentcore:InvokeAgentRuntime" in st["Action"])
    assert invoke["Resource"]["Sub"].endswith(":runtime/skyhammer_agent*")
    assert "AgentJobRoleArn" in t["Outputs"]


def test_nova_2_lite_job(agent_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    # Real Hub metadata and recipe parameters (2026-11-03): Nova 2 Lite's MTRL recipe takes
    # global_batch_size 8-64 and has its own hyperparameter set (e.g. save_every).
    llm = LLM("nova-2-lite", session=agent_session)
    request, _ = llm.plan(
        "agentic",
        tasks(16),
        output_model="wordle-nova",
        agent=RUNTIME,
        batch_size=8,
        hyperparameters_extra={"max_steps": 5, "save_every": 0},
    )
    assert request.route.backend.value == "sagemaker"
    backend = SageMakerServerlessBackend(agent_session)
    fake = FakeSageMaker()
    backend.sagemaker = fake
    monkeypatch.setattr(Dataset, "upload", lambda self, b, t: "s3://bucket/datasets/wordle.jsonl")
    backend.submit(request)
    training = json.loads(fake.created["JobConfigDocument"])["TrainingConfig"]
    assert training["BaseModelArn"].endswith("/Model/nova-textgeneration-lite-v2")
    assert training["HyperParameters"] == {"global_batch_size": 8, "max_steps": 5, "save_every": 0}
    with pytest.raises(UnsupportedParameterError, match="global_batch_size"):
        llm.plan(
            "agentic",
            tasks(),
            output_model="wordle-nova",
            agent=RUNTIME,
            batch_size=128,  # GPT-OSS allows it; Nova 2 Lite's recipe doesn't
            hyperparameters_extra={"max_steps": 5},
        )


def test_output_group_must_be_restricted(agent_session: Session) -> None:
    llm = LLM("gpt-oss-20b", session=agent_session)
    request, _ = llm.plan(
        "agentic",
        tasks(),
        output_model="support",
        agent=RUNTIME,
        batch_size=32,
        hyperparameters_extra={"max_steps": 5},
    )
    backend = SageMakerServerlessBackend(agent_session)
    fake = FakeSageMaker()
    arn = f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:model-package-group/support"
    fake.describe_model_package_group = lambda **kw: {"ModelPackageGroupArn": arn}  # type: ignore[method-assign]
    backend.sagemaker = fake
    with pytest.raises(SkyhammerError, match="can't share a group"):
        backend.ensure_model_package_group(request, restricted=True)
    fake.describe_model_package_group = lambda **kw: {  # type: ignore[method-assign]
        "ModelPackageGroupArn": arn,
        "ManagedConfiguration": {"ManagedStorageType": "Restricted"},
    }
    assert backend.ensure_model_package_group(request, restricted=True) == arn
    with pytest.raises(SkyhammerError, match="can't share a group"):
        backend.ensure_model_package_group(request)


# ---- evaluation ----

PACKAGE = f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:model-package/wordle-gptoss/1"


class FakeEvalSageMaker:
    def __init__(self) -> None:
        self.created: dict[str, Any] = {}

    def describe_job(self, **kw: Any) -> dict[str, Any]:
        assert kw["JobCategory"] == "AgentRFTEvaluation"
        if not self.created:
            raise ClientError({"Error": {"Code": "ResourceNotFound", "Message": "not found"}}, "D")
        return {"JobArn": JOB_ARN, "JobStatus": "Completed", "JobConfigDocument": "{}"}

    def create_job(self, **kw: Any) -> dict[str, str]:
        valid("sagemaker", "CreateJob", kw)
        self.created = kw
        return {"JobArn": JOB_ARN}


def eval_llm(
    agent_session: Session, monkeypatch: pytest.MonkeyPatch, llm: LLM
) -> FakeEvalSageMaker:
    fake = FakeEvalSageMaker()
    real = agent_session.client
    monkeypatch.setattr(
        agent_session,
        "client",
        lambda svc, region=None: fake if svc == "sagemaker" else real(svc, region=region),
    )
    monkeypatch.setattr(Dataset, "upload", lambda self, b, t: "s3://bucket/datasets/held.jsonl")
    return fake


def test_eval_job_on_trained_output(
    agent_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skyhammer.backends.base import ModelRef

    ref = ModelRef(
        name="wordle-gptoss",
        model_arn=PACKAGE,
        backend="sagemaker",
        base_alias="gpt-oss-20b",
        region="us-east-1",
        api="openai_chat",
        serving="sagemaker",
        artifacts_uri=None,
    )
    llm = LLM.from_model_ref(ref, session=agent_session)
    fake = eval_llm(agent_session, monkeypatch, llm)
    job = llm.create_agent_eval_job(
        tasks(16), agent=RUNTIME, group_size=4, temperature=0.7, pass_k=[1, 4]
    )
    kw = fake.created
    assert kw["JobCategory"] == "AgentRFTEvaluation" and kw["RoleArn"].endswith("AgentJobRole")
    assert job.name.startswith("wordle-gptoss-eval-") and job.output_uri.endswith(f"/{job.name}/")
    doc = json.loads(kw["JobConfigDocument"])
    assert doc["ModelPackageConfig"] == {"InputModelPackageArn": PACKAGE}
    assert doc["InputDataConfig"][0]["ChannelName"] == "evaluation"
    assert doc["OutputDataConfig"]["MlflowConfig"] == {"MlflowResourceArn": MLFLOW}
    ev = doc["EvaluationConfig"]
    assert ev["BaseModelArn"].endswith("/Model/openai-reasoning-gpt-oss-20b")
    assert ev["HyperParameters"] == {
        "batch": {"eval_group_size": 4},
        "eval_metrics_config": {"pass_k_values": [1, 4]},
        "eval_sampling_params": {"temperature": 0.7},
    }
    # Same inputs reattach instead of starting (and billing) a second run.
    again = llm.create_agent_eval_job(
        tasks(16), agent=RUNTIME, group_size=4, temperature=0.7, pass_k=[1, 4]
    )
    assert again.name == job.name and again.status().state == "completed"


def test_eval_job_on_base_model_and_checks(
    agent_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = LLM("gpt-oss-20b", session=agent_session)
    fake = eval_llm(agent_session, monkeypatch, llm)
    with pytest.raises(UnsupportedParameterError, match="pass@8 needs at least 8"):
        llm.create_agent_eval_job(tasks(), agent=RUNTIME, pass_k=[1, 8], group_size=4)
    other = f"arn:aws:bedrock-agentcore:us-east-1:{ACCOUNT}:runtime/someone_elses-AbC"
    with pytest.raises(SkyhammerError):
        llm.create_agent_eval_job(tasks(), agent=other)
    assert not fake.created  # nothing started
    llm.create_agent_eval_job(tasks(), agent=RUNTIME)
    doc = json.loads(fake.created["JobConfigDocument"])
    assert "ModelPackageConfig" not in doc
    assert doc["EvaluationConfig"]["HyperParameters"] == {"batch": {"eval_group_size": 1}}


# ---- checkpoints and resume (TR-15) ----

CKPT_GROUP = "wordle-gptoss-checkpoints"


def ckpt_arn(version: int) -> str:
    return f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:model-package/{CKPT_GROUP}/{version}"


class FakeCheckpoints(FakeSageMaker):
    """A job that registered a resumable and a weights-only package at each of 2 steps."""

    def __init__(self) -> None:
        super().__init__()
        self.listed: dict[str, Any] = {}
        self.meta = {
            ckpt_arn(1): {"IsCheckpoint": "true", "Step": "1"},
            ckpt_arn(2): {"Step": "1"},
            ckpt_arn(3): {"IsCheckpoint": "true", "Step": "2"},
            ckpt_arn(4): {},
        }

    def describe_job(self, **kw: Any) -> dict[str, Any]:
        return {
            "JobName": "wordle-gptoss-0a1b2c3d",
            "JobArn": JOB_ARN,
            "JobStatus": "Stopped",
            "CreationTime": 100.0,
            "EndTime": 200.0,
        }

    def get_paginator(self, name: str) -> Any:
        fake = self

        class P:
            def paginate(self, **kw: Any) -> list[dict[str, Any]]:
                valid("sagemaker", "ListModelPackages", kw)
                fake.listed = kw
                return [
                    {"ModelPackageSummaryList": [{"ModelPackageArn": a} for a in fake.meta][:2]},
                    {"ModelPackageSummaryList": [{"ModelPackageArn": a} for a in fake.meta][2:]},
                ]

        return P()

    def describe_model_package(self, ModelPackageName: str) -> dict[str, Any]:
        meta = self.meta[ModelPackageName]
        return {"CustomerMetadataProperties": meta} if meta else {}


def test_agent_rl_checkpoints(agent_session: Session) -> None:
    backend = SageMakerServerlessBackend(agent_session)
    fake = FakeCheckpoints()
    backend.sagemaker = fake
    record = SimpleNamespace(
        job_id=JOB_ARN, job_name="wordle-gptoss-0a1b2c3d", output_model="wordle-gptoss"
    )
    ckpts = backend.checkpoints(record)  # type: ignore[arg-type]
    # Only packages created while this job ran (other runs share the group).
    assert fake.listed["ModelPackageGroupName"] == CKPT_GROUP
    assert (fake.listed["CreationTimeAfter"], fake.listed["CreationTimeBefore"]) == (100.0, 200.0)
    assert [(c.step, c.uri, c.resumable) for c in ckpts] == [
        (1, ckpt_arn(1), True),
        (1, ckpt_arn(2), False),
        (2, ckpt_arn(3), True),
        (4, ckpt_arn(4), False),  # no step recorded: its position
    ]
    assert {c.job_name for c in ckpts} == {"wordle-gptoss-0a1b2c3d"}


def plan_rl(llm: LLM, **kw: Any) -> Any:
    request, _ = llm.plan(
        "agentic",
        kw.pop("data", None) or tasks(),
        output_model="wordle-gptoss",
        agent=RUNTIME,
        hyperparameters_extra={"max_steps": 20, "group_size": 8},
        **{"batch_size": 32, **kw},
    )
    return request


def test_resume_from_checkpoint(agent_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    from skyhammer import agent_rl
    from skyhammer.jobs import Checkpoint, JobRecord

    llm = LLM("gpt-oss-20b", session=agent_session)
    first = plan_rl(llm)
    resumed = plan_rl(llm, resume_from=ckpt_arn(3))
    # A new job (the stopped one keeps its name), continuing from the checkpoint.
    assert resumed.job_name != first.job_name and resumed.source_model == ckpt_arn(3)
    backend = SageMakerServerlessBackend(agent_session)
    fake = FakeSageMaker()
    backend.sagemaker = fake
    monkeypatch.setattr(Dataset, "upload", lambda self, b, t: "s3://bucket/datasets/wordle.jsonl")
    backend.submit(resumed)
    config = json.loads(fake.created["JobConfigDocument"])["ModelPackageConfig"]
    assert config["InputModelPackageArn"] == ckpt_arn(3)
    assert config["IntermediateCheckpointModelPackageGroupArn"].endswith("/" + CKPT_GROUP)

    # AWS resumes only with the same base model, tasks and hyperparameters.
    record = JobRecord.from_request(
        first, JobHandle("sagemaker", first.job_name, JOB_ARN, "us-east-1")
    )
    agent_rl.check_resume(record, resumed)
    other = plan_rl(llm, data=tasks(70), batch_size=64, resume_from=ckpt_arn(3))
    with pytest.raises(SkyhammerError, match="changes its tasks, hyperparameter batch_size"):
        agent_rl.check_resume(record, other)

    weights = Checkpoint(1, uri=ckpt_arn(2), job_name=first.job_name)
    with pytest.raises(SkyhammerError, match="weights only"):
        llm.create_agent_rl_job(
            tasks(),
            agent=RUNTIME,
            output_model="wordle-gptoss",
            max_steps=20,
            batch_size=32,
            resume_from=weights,
        )
    with pytest.raises(SkyhammerError, match="model package"):
        agent_rl.resume_arn("s3://bucket/jobs/x/checkpoints/step_5/")
    # The mismatch is caught before anything is submitted.
    monkeypatch.setattr("skyhammer.llm.registry.load_job", lambda s, name: record)
    ckpt = Checkpoint(2, uri=ckpt_arn(3), resumable=True, job_name=first.job_name)
    with pytest.raises(SkyhammerError, match="changes its hyperparameter max_steps"):
        llm.create_agent_rl_job(
            tasks(),
            agent=RUNTIME,
            output_model="wordle-gptoss",
            max_steps=40,
            group_size=8,
            batch_size=32,
            resume_from=ckpt,
        )
    with pytest.raises(UnsupportedParameterError, match="multi-turn RL jobs only"):
        llm.plan("sft", tasks(), output_model="x", resume_from=ckpt_arn(3))


def test_train_rl_output_further(agent_session: Session) -> None:
    from skyhammer.backends.base import ModelRef

    ref = ModelRef(
        name="wordle-gptoss",
        model_arn=PACKAGE,
        backend="sagemaker",
        base_alias="gpt-oss-20b",
        region="us-east-1",
        api="openai_chat",
        serving="sagemaker",
        artifacts_uri=None,
    )
    llm = LLM.from_model_ref(ref, session=agent_session)
    # Iterative training starts from the output package's weights (no AWS lookups).
    assert plan_rl(llm).source_model == PACKAGE
    with pytest.raises(SkyhammerError, match="only be trained further with create_agent_rl_job"):
        llm.plan("sft", tasks(), output_model="wordle-sft")
    with pytest.raises(SkyhammerError, match="resume_from continues a stopped run"):
        plan_rl(llm, resume_from=ckpt_arn(3))
