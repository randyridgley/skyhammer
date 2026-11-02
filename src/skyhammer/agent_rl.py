"""Multi-turn (agentic) RL (TR-12): train a model on whole episodes of your agent's work.

SageMaker AI multi-turn RL ("MTRL") runs the training loop; your agent runs the episodes.
For every rollout the job invokes your agent (an Amazon Bedrock AgentCore runtime named
``skyhammer_agent*``, or a Lambda named ``skyhammer-agent-*``) with one dataset row. The
agent calls the model being trained as often as the task needs, acts in its environment
(tools, APIs, code), and reports one reward at the end (or one per turn). The job compares
rewards across a group of rollouts of the same task (GRPO by default) and updates a LoRA
adapter.

Two halves:

- **Training** (your machine): ``LLM("gpt-oss-20b").create_agent_rl_job(tasks,
  agent=<runtime ARN>, max_steps=..., output_model=...)``. Rows are tasks: any JSON
  objects; a ``prompt`` string is passed to the agent as is, other rows are passed as
  JSON. Needs a SageMaker MLflow App (settings ``mlflow_arn``), where the service logs
  rewards and trajectories, and the bootstrap ``AgentJobRole``.
- **The agent** (AgentCore or Lambda): wrap the entrypoint with ``rollout_handler``::

      from skyhammer.agent_rl import Rollout, rollout_handler

      @app.entrypoint
      @rollout_handler
      def handle(rollout: Rollout) -> float:
          reply = rollout.chat(messages=[{"role": "user", "content": rollout.prompt}])
          return score(reply.choices[0].message.content, rollout.task)

  ``rollout.chat`` takes the OpenAI chat parameters (tools included) and returns a
  ``ChatCompletion``; every call is recorded as a turn of the trajectory. The handler
  completes the rollout and reports the reward, and always reports one: a rollout that
  raises scores 0.0 and is marked failed. The agent's role needs the AWS managed policy
  ``AmazonSageMakerJobRuntimeAccess``.

Billing is per token (prefill, sample, train) and depends on how long your episodes are,
so there is no estimate up front: ``max_steps`` is required and bounds the run, and the
finished job reports the tokens it billed. For scale: AWS's Wordle walkthrough (GPT-OSS
20B, 100 steps of 128 tasks x 8 rollouts) billed about 1.3 billion tokens, $339.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skyhammer.catalog import DatasetLimits
from skyhammer.errors import ConfigError, SkyhammerError, UnsupportedParameterError

if TYPE_CHECKING:
    from skyhammer.hub import HubModel, ParamSpec, Recipe
    from skyhammer.inference import ChatCompletion
    from skyhammer.jobs import TrainingRequest
    from skyhammer.session import Session

log = logging.getLogger("skyhammer")

JOB_CATEGORY = "AgentRFT"
SCHEMA_VERSION = "1.0.0"
LOG_GROUP = "/aws/sagemaker/Job/AgentRFT"
# Tasks go to your agent unchanged; only size bounds apply.
LIMITS = DatasetLimits(
    min_samples=1,
    max_samples=1_000_000,
    max_tokens_per_sample=1_000_000,
    source="SageMaker multi-turn RL (your agent reads the tasks)",
)
_AGENTCORE_ARN = re.compile(
    r"arn:aws[\w-]*:bedrock-agentcore:(?P<region>[\w-]+):(?P<account>\d{12}):runtime/(?P<name>[\w-]+)"
)
_LAMBDA_ARN = re.compile(
    r"arn:aws[\w-]*:lambda:(?P<region>[\w-]+):(?P<account>\d{12}):function:(?P<name>[\w-]+)"
)
_MLFLOW_APP = re.compile(r"arn:aws[\w-]*:sagemaker:[\w-]+:\d{12}:mlflow-app/[\w-]+")
# SDK hyperparameter -> MTRL recipe key.
_SDK_TO_RECIPE = {
    "epochs": "max_epochs",
    "learning_rate": "learning_rate",
    "batch_size": "global_batch_size",
    "lora_rank": "lora_rank",
    "lora_alpha": "lora_alpha",
}
_STATE = {
    "InProgress": "training",
    "Completed": "completed",
    "Failed": "failed",
    "Stopping": "training",
    "Stopped": "stopped",
    "Deleting": "stopped",
    "DeleteFailed": "stopped",
}


# ---- training side ----


def format_record(record: dict[str, Any]) -> dict[str, Any]:
    """The service passes the ``prompt`` column to your agent verbatim."""
    prompt = record.get("prompt")
    if isinstance(prompt, str) and len(record) == 1:
        return {"prompt": prompt}
    return {"prompt": json.dumps(record, ensure_ascii=False)}


def agent_config(agent: str, region: str) -> dict[str, Any]:
    """AgentConfig for an AgentCore runtime or Lambda ARN in ``region``."""
    if m := _AGENTCORE_ARN.fullmatch(agent):
        if not m["name"].startswith("skyhammer_agent"):
            raise SkyhammerError(
                "Name the AgentCore runtime skyhammer_agent* (the bootstrap job role may "
                f"invoke only those), not {m['name'].rsplit('-', 1)[0]!r}"
            )
        config: dict[str, Any] = {
            "BedrockAgentCoreConfig": {"AgentRuntimeArn": agent, "Qualifier": "DEFAULT"}
        }
    elif m := _LAMBDA_ARN.fullmatch(agent):
        if not m["name"].startswith("skyhammer-agent-"):
            raise SkyhammerError(
                f"Name the agent Lambda skyhammer-agent-* (the bootstrap job role may invoke "
                f"only those), not {m['name']!r}"
            )
        config = {"CustomAgentLambdaConfig": {"LambdaArn": agent}}
    else:
        raise SkyhammerError(
            f"agent must be an AgentCore runtime ARN (arn:aws:bedrock-agentcore:<region>:"
            f"<account>:runtime/<id>) or a Lambda ARN, not {agent!r}"
        )
    if m["region"] != region:
        raise SkyhammerError(
            f"The agent is in {m['region']} but the job runs in {region}; multi-turn RL "
            "needs both in one region"
        )
    return config


def _typed(spec: ParamSpec, value: Any) -> Any:
    if spec.type == "array":
        if not isinstance(value, list | tuple):
            raise UnsupportedParameterError(f"{spec.name} must be a list, got {value!r}")
        return list(value)
    checked = spec.check(value)
    if spec.type == "integer":
        return int(checked)
    if spec.type == "float":
        return float(checked)
    if spec.type == "boolean":
        return checked == "true"
    return checked


def hyperparameters(
    request: TrainingRequest, recipe: Recipe, params: dict[str, ParamSpec]
) -> dict[str, Any]:
    """The values you set, checked against the recipe (the service fills in the rest).
    ``max_steps`` is required: it is what bounds a token-billed run."""
    set_fields = request.hyperparameters.set_fields()
    unknown = sorted(k for k in set_fields if k not in _SDK_TO_RECIPE)
    if unknown:
        raise UnsupportedParameterError(
            f"Multi-turn RL doesn't take: {', '.join(unknown)}. Pass recipe keys with "
            "hyperparameters_extra={...}; tunable: " + ", ".join(sorted(params))
        )
    values = {_SDK_TO_RECIPE[k]: v for k, v in set_fields.items()}
    values.update(request.hyperparameters.extra)
    out = {}
    for key, value in values.items():
        if key not in params:
            raise UnsupportedParameterError(
                f"{key!r} isn't tunable in recipe {recipe.name}; tunable: "
                + ", ".join(sorted(params))
            )
        out[key] = _typed(params[key], value)
    if "max_steps" not in out:
        raise UnsupportedParameterError(
            "Set max_steps: multi-turn RL bills every token your episodes use, and the "
            f"recipe default ({params['max_steps'].default if 'max_steps' in params else '?'} "
            "steps) can cost hundreds of dollars"
        )
    batch_spec = params.get("global_batch_size")
    batch = out.get("global_batch_size", batch_spec.default if batch_spec else None)
    if batch and request.dataset.num_records < int(batch):
        raise UnsupportedParameterError(
            f"Each step takes {batch} tasks (global_batch_size) but the dataset has "
            f"{request.dataset.num_records}; add tasks or lower batch_size"
            + (f" ({list(batch_spec.enum)})" if batch_spec and batch_spec.enum else "")
        )
    return out


def check(
    request: TrainingRequest, hub_model: HubModel, params_for: Callable[..., Any]
) -> dict[str, Any]:
    """Everything that can fail before spending; returns the hyperparameters to send."""
    if not request.agent:
        raise SkyhammerError("Multi-turn RL needs agent=<AgentCore runtime or Lambda ARN>")
    agent_config(request.agent, request.route.region)
    if request.validation_dataset is None and request.hyperparameters.extra.get("val_every"):
        log.info("val_every has no effect without a validation dataset")
    if hub_model.gated and not request.accept_eula:
        raise SkyhammerError(
            f"{request.route.model.display_name} has a gated license ({hub_model.license}). "
            "Review it, then pass accept_eula=True."
        )
    recipe = hub_model.recipe("MTRL", request.training_type)
    return hyperparameters(request, recipe, params_for(recipe))


def mlflow_app(session: Session) -> str:
    arn = session.settings.mlflow_arn
    if not arn or not _MLFLOW_APP.fullmatch(arn):
        raise ConfigError(
            "Multi-turn RL logs rewards and trajectories to a SageMaker MLflow App, which "
            "the service requires. Create one (aws sagemaker create-mlflow-app) and set "
            "mlflow_arn to its ARN (arn:aws:sagemaker:<region>:<account>:mlflow-app/<id>)"
            + (f"; {arn!r} isn't an MLflow App ARN" if arn else "")
        )
    return arn


def build_job(
    request: TrainingRequest,
    hyperparameters: dict[str, Any],
    *,
    role_arn: str,
    base_model_arn: str,
    mlflow_arn: str,
    training_uri: str,
    validation_uri: str | None,
    output_uri: str,
    output_group_arn: str,
    checkpoint_group_arn: str,
    kms_key_arn: str | None,
    tags: list[dict[str, str]],
) -> dict[str, Any]:
    """CreateJob kwargs (no AWS calls)."""
    assert request.agent is not None
    channels = [("train", training_uri)] + (
        [("validation", validation_uri)] if validation_uri else []
    )
    output: dict[str, Any] = {"S3OutputPath": output_uri}
    if kms_key_arn:
        output["KmsKeyArn"] = kms_key_arn
    training: dict[str, Any] = {
        "BaseModelArn": base_model_arn,
        "MlflowConfig": {
            "MlflowResourceArn": mlflow_arn,
            "MlflowExperimentName": request.output_model,
            "MlflowRunName": request.job_name,
        },
        "AcceptEula": request.accept_eula,
    }
    if hyperparameters:
        training["HyperParameters"] = hyperparameters
    document = {
        "AgentConfig": agent_config(request.agent, request.route.region),
        "InputDataConfig": [
            {
                "ChannelName": name,
                "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": uri}},
            }
            for name, uri in channels
        ],
        "OutputDataConfig": output,
        "ModelPackageConfig": {
            "OutputModelPackageGroupArn": output_group_arn,
            "IntermediateCheckpointModelPackageGroupArn": checkpoint_group_arn,
        },
        "TrainingConfig": training,
    }
    return {
        "JobName": request.job_name,
        "RoleArn": role_arn,
        "JobCategory": JOB_CATEGORY,
        "JobConfigSchemaVersion": SCHEMA_VERSION,
        "JobConfigDocument": json.dumps(document),
        "Tags": tags,
    }


@dataclass(frozen=True)
class JobInfo:
    """What DescribeJob says about a multi-turn RL job."""

    state: str  # a JobState value
    message: str | None
    output_package_arn: str | None
    billable_tokens: dict[str, int]
    progress: dict[str, Any]

    @property
    def summary(self) -> str | None:
        parts = []
        p = self.progress
        if p.get("CurrentStep") is not None:
            total = p.get("MaxSteps") or (
                (p.get("MaxEpoch") or 0) * (p.get("StepsPerEpoch") or 0) or None
            )
            parts.append(f"step {p['CurrentStep']}" + (f"/{total}" if total else ""))
        if self.billable_tokens:
            t = self.billable_tokens
            parts.append(
                "billed tokens: "
                + ", ".join(
                    f"{k.removesuffix('TokenCount').lower()} {v:,}" for k, v in sorted(t.items())
                )
            )
        return "; ".join(parts) or None


def parse_job(job: dict[str, Any]) -> JobInfo:
    doc = json.loads(job.get("JobConfigDocument") or "{}")
    service = doc.get("ServiceOutput") or {}
    state = _STATE.get(job["JobStatus"], "pending")
    if state == "training" and job.get("SecondaryStatus") in ("Starting", "Downloading", "Pending"):
        state = "pending"
    info = JobInfo(
        state=state,
        message=job.get("FailureReason") or job.get("SecondaryStatus"),
        output_package_arn=service.get("OutputModelPackageArn"),
        billable_tokens={k: int(v) for k, v in (service.get("BillableTokenUsage") or {}).items()},
        progress=service.get("ProgressInfo") or {},
    )
    if not job.get("FailureReason") and info.summary:
        info = JobInfo(
            info.state,
            f"{info.message}: {info.summary}" if info.message else info.summary,
            info.output_package_arn,
            info.billable_tokens,
            info.progress,
        )
    return info


# ---- agent side ----


class Rollout:
    """One episode, as your agent sees it: the task, and the model being trained.

    ``payload`` is what the training job sends the agent: the task row (``prompt``) and
    rollout metadata (job ARN, trajectory ID, sampling settings)."""

    def __init__(self, payload: dict[str, Any], *, client: Any = None) -> None:
        meta = payload.get("metadata") or {}
        self.payload = payload
        self.job_arn: str = meta.get("job_arn") or meta.get("jobArn") or ""
        self.trajectory_id: str = (
            meta.get("trajectory_id") or meta.get("trajectoryId") or meta.get("rolloutId") or ""
        )
        if not self.job_arn or not self.trajectory_id:
            raise SkyhammerError("Rollout payload has no job ARN / trajectory ID in metadata")
        self.region: str = meta.get("region") or self.job_arn.split(":")[3]
        instance = payload.get("instance")
        raw = payload.get("prompt", instance.get("prompt") if isinstance(instance, dict) else None)
        #: The task row's ``prompt`` string (a JSON object for rows with other fields).
        self.prompt: str = raw if isinstance(raw, str) else json.dumps(raw)
        params = payload.get("inferenceParams") or payload.get("inference_params") or {}
        #: The job's sampling settings (temperature, max_tokens, top_p), applied by ``chat``.
        self.sampling: dict[str, Any] = {
            k: params.get(k, params.get(camel))
            for k, camel in (
                ("temperature", "temperature"),
                ("max_tokens", "maxTokens"),
                ("top_p", "topP"),
            )
            if params.get(k, params.get(camel)) is not None
        }
        self._client = client
        self.turns = 0

    @property
    def task(self) -> Any:
        """The task row: ``prompt`` parsed as JSON when it is JSON, else the string."""
        try:
            return json.loads(self.prompt)
        except ValueError:
            return self.prompt

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("sagemakerjobruntime", region_name=self.region)
        return self._client

    def chat(self, **params: Any) -> ChatCompletion:
        """Call the model being trained (OpenAI chat parameters: ``messages``, ``tools``,
        ...). The job's sampling settings apply unless you pass them. Each call is one
        turn of the trajectory."""
        from skyhammer.inference import from_openai_chat_response

        body = {"model": "default", **self.sampling, **params}
        resp = self.client.sample(
            JobArn=self.job_arn, TrajectoryId=self.trajectory_id, Body=json.dumps(body)
        )
        raw = resp["Body"]
        data = json.loads(raw.read() if hasattr(raw, "read") else raw)
        self.turns += 1
        return from_openai_chat_response(str(body["model"]), data)

    def complete(self, reward: float | list[float]) -> None:
        """Seal the trajectory and report its reward (or one reward per turn)."""
        rewards = [float(reward)] if isinstance(reward, int | float) else [float(r) for r in reward]
        self.client.complete_rollout(
            JobArn=self.job_arn, TrajectoryId=self.trajectory_id, Status="ready"
        )
        self.client.update_reward(
            JobArn=self.job_arn, TrajectoryId=self.trajectory_id, Rewards=rewards
        )

    def fail(self) -> None:
        """Mark the trajectory failed (reward 0.0)."""
        self.client.complete_rollout(
            JobArn=self.job_arn, TrajectoryId=self.trajectory_id, Status="failed"
        )
        self.client.update_reward(
            JobArn=self.job_arn, TrajectoryId=self.trajectory_id, Rewards=[0.0]
        )


def rollout_handler(
    fn: Callable[[Rollout], float | list[float] | dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Turn ``fn(rollout) -> reward`` into an agent entrypoint for training payloads.

    ``fn`` returns a reward, a list of per-turn rewards, or a dict with ``"reward"``.
    The handler reports it and returns ``{"reward": ...}``. If ``fn`` raises, the
    trajectory is marked failed with reward 0.0 and the error is re-raised. A rollout
    that never calls ``rollout.chat`` fails the whole job (the service needs at least one
    sample per rollout), so the handler warns about it."""

    @functools.wraps(fn)
    def handler(payload: dict[str, Any]) -> dict[str, Any]:
        rollout = Rollout(payload)
        try:
            result = fn(rollout)
        except Exception:
            log.exception("rollout %s failed", rollout.trajectory_id)
            try:
                rollout.fail()
            except Exception:
                log.exception("could not report the failed rollout")
            raise
        reward = result.get("reward") if isinstance(result, dict) else result
        if reward is None:
            raise SkyhammerError("The rollout function returned no reward")
        if rollout.turns == 0:
            log.warning(
                "rollout %s never called rollout.chat; the job fails rollouts without a "
                "sample from the model",
                rollout.trajectory_id,
            )
        rollout.complete(reward)
        return {"reward": reward}

    return handler
