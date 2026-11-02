"""Find and delete leftovers from tests and smoke runs (PK-5).

A resource is a leftover only when all of these hold, so the sweeper can't touch anything
else in a shared account:

- its name starts with a sweep prefix (default ``skyhammer-it-`` and ``skyhammer-smoke-``),
- it carries the ``skyhammer:name`` tag (for services that tag the resource),
- it is older than ``older_than_hours`` (default 6), so runs in progress are left alone.

Grader Lambdas (``skyhammer-grader-*``) are shared by every job with the same reward code,
so they're only swept with ``include_graders=True`` and only in regions with no RL job
running. ``sweep`` is a dry run unless ``dry_run=False``.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError

from skyhammer.serving import EndpointDeployer, EndpointNames

if TYPE_CHECKING:
    from skyhammer.session import Session

log = logging.getLogger(__name__)

TAG = "skyhammer:name"
DEFAULT_PREFIXES = ("skyhammer-it-", "skyhammer-smoke-")
GRADER_PREFIX = "skyhammer-grader-"


@dataclass
class Leftover:
    kind: str
    name: str
    region: str
    created: dt.datetime
    delete: Callable[[], None] = field(repr=False, compare=False)

    @property
    def age_hours(self) -> float:
        return (dt.datetime.now(dt.UTC) - self.created).total_seconds() / 3600


def _utc(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.UTC)
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


class _Finder:
    def __init__(self, session: Session, region: str, prefixes: tuple[str, ...]) -> None:
        self.session = session
        self.region = region
        self.prefixes = prefixes

    def client(self, service: str) -> Any:
        return self.session.client(service, region=self.region)

    def ours(self, name: str) -> bool:
        return name.startswith(self.prefixes)

    # ---- SageMaker ----

    def _delete_endpoint(self, names: EndpointNames) -> None:
        EndpointDeployer(self.session, self.region).delete(names)  # IC, scaling, alarm too

    def sagemaker(self) -> Iterable[Leftover]:
        sm = self.client("sagemaker")

        def tagged(arn: str) -> bool:
            return any(t["Key"] == TAG for t in sm.list_tags(ResourceArn=arn).get("Tags", []))

        endpoints = set()
        for page in sm.get_paginator("list_endpoints").paginate():
            for ep in page["Endpoints"]:
                name = ep["EndpointName"]
                if self.ours(name) and tagged(ep["EndpointArn"]):
                    endpoints.add(name)
                    names = EndpointNames.for_deployment(name)
                    yield Leftover(
                        "sagemaker-endpoint",
                        name,
                        self.region,
                        _utc(ep["CreationTime"]),
                        lambda n=names: self._delete_endpoint(n),  # type: ignore[misc]
                    )
        # Configs and models left by deploys that failed before the endpoint existed.
        for page in sm.get_paginator("list_endpoint_configs").paginate():
            for cfg in page["EndpointConfigs"]:
                name = cfg["EndpointConfigName"]
                if self.ours(name) and name.removesuffix("-cfg") not in endpoints:
                    yield Leftover(
                        "sagemaker-endpoint-config",
                        name,
                        self.region,
                        _utc(cfg["CreationTime"]),
                        lambda n=name: sm.delete_endpoint_config(EndpointConfigName=n),  # type: ignore[misc]
                    )
        for page in sm.get_paginator("list_models").paginate():
            for m in page["Models"]:
                name = m["ModelName"]
                if self.ours(name) and name.removesuffix("-model") not in endpoints:
                    yield Leftover(
                        "sagemaker-model",
                        name,
                        self.region,
                        _utc(m["CreationTime"]),
                        lambda n=name: sm.delete_model(ModelName=n),  # type: ignore[misc]
                    )
        for page in sm.get_paginator("list_training_jobs").paginate(StatusEquals="InProgress"):
            for job in page["TrainingJobSummaries"]:
                name = job["TrainingJobName"]
                if self.ours(name):
                    yield Leftover(
                        "sagemaker-training-job",
                        name,
                        self.region,
                        _utc(job["CreationTime"]),
                        lambda n=name: sm.stop_training_job(TrainingJobName=n),  # type: ignore[misc]
                    )

    # ---- Bedrock ----

    def bedrock(self) -> Iterable[Leftover]:
        br = self.client("bedrock")

        def tagged(arn: str) -> bool:
            tags = br.list_tags_for_resource(resourceARN=arn).get("tags", [])
            return any(t["key"] == TAG for t in tags)

        for d in br.list_custom_model_deployments().get("modelDeploymentSummaries", []):
            name, arn = d["customModelDeploymentName"], d["customModelDeploymentArn"]
            if self.ours(name) and tagged(arn):
                yield Leftover(
                    "bedrock-custom-deployment",
                    name,
                    self.region,
                    _utc(d["createdAt"]),
                    lambda a=arn: br.delete_custom_model_deployment(  # type: ignore[misc]
                        customModelDeploymentIdentifier=a
                    ),
                )
        for m in br.list_custom_models().get("modelSummaries", []):
            name, arn = m["modelName"], m["modelArn"]
            if self.ours(name) and tagged(arn):
                yield Leftover(
                    "bedrock-custom-model",
                    name,
                    self.region,
                    _utc(m["creationTime"]),
                    lambda a=arn: br.delete_custom_model(modelIdentifier=a),  # type: ignore[misc]
                )
        for m in br.list_imported_models().get("modelSummaries", []):
            name, arn = m["modelName"], m["modelArn"]
            if self.ours(name) and tagged(arn):
                yield Leftover(
                    "bedrock-imported-model",
                    name,
                    self.region,
                    _utc(m["creationTime"]),
                    lambda a=arn: br.delete_imported_model(modelIdentifier=a),  # type: ignore[misc]
                )
        for job in br.list_model_customization_jobs(statusEquals="InProgress").get(
            "modelCustomizationJobSummaries", []
        ):
            name = job["jobName"]
            if self.ours(name):
                yield Leftover(
                    "bedrock-customization-job",
                    name,
                    self.region,
                    _utc(job["creationTime"]),
                    lambda a=job["jobArn"]: br.stop_model_customization_job(  # type: ignore[misc]
                        jobIdentifier=a
                    ),
                )
        for p in br.list_provisioned_model_throughputs().get("provisionedModelSummaries", []):
            name, arn = p["provisionedModelName"], p["provisionedModelArn"]
            if self.ours(name) and not p.get("commitmentDuration") and tagged(arn):
                yield Leftover(  # hourly billing: the costliest thing a test can leave behind
                    "bedrock-provisioned",
                    name,
                    self.region,
                    _utc(p["creationTime"]),
                    lambda a=arn: br.delete_provisioned_model_throughput(  # type: ignore[misc]
                        provisionedModelId=a
                    ),
                )
        for g in br.list_guardrails().get("guardrails", []):
            if self.ours(g["name"]) and tagged(g["arn"]):
                yield Leftover(
                    "bedrock-guardrail",
                    g["name"],
                    self.region,
                    _utc(g["createdAt"]),
                    lambda i=g["id"]: br.delete_guardrail(guardrailIdentifier=i),  # type: ignore[misc]
                )

    def rl_jobs_running(self) -> bool:
        """True if any RL job that could still call a grader Lambda is running here."""
        br = self.client("bedrock")
        bedrock_jobs = br.list_model_customization_jobs(statusEquals="InProgress").get(
            "modelCustomizationJobSummaries", []
        )
        if any(j.get("customizationType") == "REINFORCEMENT_FINE_TUNING" for j in bedrock_jobs):
            return True
        sm = self.client("sagemaker")
        jobs = sm.list_training_jobs(StatusEquals="InProgress").get("TrainingJobSummaries", [])
        return bool(jobs)  # serverless RLVR jobs don't say their technique in the summary

    def graders(self) -> Iterable[Leftover]:
        lam = self.client("lambda")
        for page in lam.get_paginator("list_functions").paginate():
            for fn in page["Functions"]:
                name = fn["FunctionName"]
                if not name.startswith(GRADER_PREFIX):
                    continue
                tags = lam.list_tags(Resource=fn["FunctionArn"]).get("Tags", {})
                if TAG not in tags:
                    continue
                yield Leftover(
                    "grader-lambda",
                    name,
                    self.region,
                    _utc(fn["LastModified"]),
                    lambda n=name: lam.delete_function(FunctionName=n),  # type: ignore[misc]
                )


def find_leftovers(
    session: Session,
    regions: Iterable[str],
    *,
    prefixes: tuple[str, ...] = DEFAULT_PREFIXES,
    older_than_hours: float = 6,
    include_graders: bool = False,
) -> list[Leftover]:
    out: list[Leftover] = []
    for region in regions:
        finder = _Finder(session, region, prefixes)
        sources: list[Callable[[], Iterable[Leftover]]] = [finder.sagemaker, finder.bedrock]
        if include_graders:
            if finder.rl_jobs_running():
                log.warning("%s: RL jobs running, keeping grader Lambdas", region)
            else:
                sources.append(finder.graders)
        for source in sources:
            try:
                out += [x for x in source() if x.age_hours >= older_than_hours]
            except ClientError as e:  # a service not enabled in a region, missing permission
                log.warning("%s: skipped %s: %s", region, source.__name__, e)
    return out


def sweep(
    session: Session,
    regions: Iterable[str],
    *,
    dry_run: bool = True,
    **kwargs: Any,
) -> list[Leftover]:
    """Delete leftovers (``find_leftovers`` arguments); returns what was (or would be) removed.

    Deployments go before the models they serve, so dependent deletes succeed.
    """
    order = [
        "bedrock-customization-job",
        "sagemaker-training-job",
        "sagemaker-endpoint",
        "bedrock-custom-deployment",
        "bedrock-provisioned",  # before the custom model it serves
        "sagemaker-endpoint-config",
        "sagemaker-model",
        "bedrock-custom-model",
        "bedrock-imported-model",
        "bedrock-guardrail",
        "grader-lambda",
    ]
    leftovers = sorted(
        find_leftovers(session, regions, **kwargs), key=lambda x: order.index(x.kind)
    )
    for item in leftovers:
        if dry_run:
            continue
        try:
            item.delete()
            log.info("deleted %s %s (%s)", item.kind, item.name, item.region)
        except ClientError as e:
            log.warning("could not delete %s %s: %s", item.kind, item.name, e)
    return leftovers
