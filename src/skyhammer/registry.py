"""Manifest store: the SDK's only state, as JSON in the home-region bootstrap bucket.

    manifests/jobs/<job_name>.json      JobRecord (one per training job)
    manifests/models/<model_name>.json  {latest_job, jobs[], deployment, deleted, eval,
                                         model_ref (promoted models, no job here)}
    manifests/envs/<app>.json           environments: {envs: {prod: {targets, history}}}

Model manifests map the user's name ("support-qwen-v1") to its latest job, so jobs can be
reattached by name from any process, and remember deployments so ``LLM.get(name)`` can
chat without redeploying. Model and environment manifests are updated with S3 conditional
writes, so processes changing the same model at once don't lose each other's changes.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any

from skyhammer import tags as tagging
from skyhammer.errors import SkyhammerError
from skyhammer.jobs import JobRecord, ModelRef

if TYPE_CHECKING:
    from skyhammer.session import Session

_JOBS = "jobs"
_MODELS = "models"


def _s3(session: Session) -> Any:
    return session.client("s3")


def _update(session: Session, kind: str, name: str, change: Any) -> dict[str, Any] | None:
    res = session.resources
    return tagging.update_manifest(_s3(session), res.bucket, kind, name, change, res.kms_key_arn)


def save_job(session: Session, record: JobRecord) -> None:
    res = session.resources
    tagging.write_manifest(
        _s3(session), res.bucket, _JOBS, record.job_name, record.to_json(), res.kms_key_arn
    )

    def change(model: dict[str, Any] | None) -> dict[str, Any]:
        model = model or {"name": record.output_model, "base_model": record.base_model, "jobs": []}
        if record.job_name not in model["jobs"]:
            model["jobs"].append(record.job_name)
        model.update(latest_job=record.job_name, deleted=False, updated_at=time.time())
        return model

    _update(session, _MODELS, record.output_model, change)


def save_model_ref(session: Session, ref: ModelRef, **extra: Any) -> None:
    """Record a model that has no training job in this account (promoted from another
    account), so ``LLM.get(name)`` finds it."""

    def change(model: dict[str, Any] | None) -> dict[str, Any]:
        model = model or {"name": ref.name, "base_model": ref.base_alias, "jobs": []}
        model.update(
            model_ref=dataclasses.asdict(ref), deleted=False, updated_at=time.time(), **extra
        )
        return model

    _update(session, _MODELS, ref.name, change)


def load_job(session: Session, job_name: str) -> JobRecord | None:
    data = tagging.read_manifest(_s3(session), session.resources.bucket, _JOBS, job_name)
    return JobRecord.from_json(data) if data else None


def load_model(session: Session, name: str) -> dict[str, Any] | None:
    return tagging.read_manifest(_s3(session), session.resources.bucket, _MODELS, name)


def update_model(session: Session, name: str, **changes: Any) -> None:
    def change(model: dict[str, Any] | None) -> dict[str, Any]:
        if model is None:
            raise SkyhammerError(f"No skyhammer model named {name!r}")
        model.update(changes, updated_at=time.time())
        return model

    _update(session, _MODELS, name, change)


def find_job(session: Session, name: str) -> JobRecord:
    """Resolve a model name (-> its latest job) or an exact job name."""
    model = load_model(session, name)
    if model and model.get("latest_job"):
        record = load_job(session, model["latest_job"])
        if record:
            return record
    record = load_job(session, name)
    if record:
        return record
    raise SkyhammerError(
        f"No skyhammer job or model named {name!r} in {session.region}. "
        "Run `skyhammer jobs list` to see what exists."
    )


def list_models(session: Session) -> list[dict[str, Any]]:
    s3 = _s3(session)
    bucket = session.resources.bucket
    names = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=f"manifests/{_MODELS}/"
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                names.append(key.rsplit("/", 1)[-1].removesuffix(".json"))
    models = [m for n in names if (m := load_model(session, n))]
    return sorted(models, key=lambda m: m.get("updated_at", 0), reverse=True)


def list_jobs(session: Session, *, include_deleted: bool = False) -> list[JobRecord]:
    records = []
    for model in list_models(session):
        if model.get("deleted") and not include_deleted:
            continue
        record = load_job(session, model["latest_job"]) if model.get("latest_job") else None
        if record:
            records.append(record)
    return records
