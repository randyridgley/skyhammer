"""Manifests are updated with S3 conditional writes, so concurrent writers don't lose changes."""

import json
from typing import Any

import boto3
import pytest

from skyhammer import Environment, registry
from skyhammer import tags as tagging
from skyhammer.errors import SkyhammerError
from skyhammer.session import Session
from tests.unit.test_registry_cli import record, s3_session  # noqa: F401  (fixture)


class Racing:
    """An S3 client where another process writes ``key`` just before our next
    conditional write lands (``times`` times)."""

    def __init__(self, real: Any, key: str, other: dict[str, Any], times: int = 1) -> None:
        self._real = real
        self.key = key
        self.other = other
        self.times = times
        self.puts: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def put_object(self, **kw: Any) -> Any:
        self.puts.append(kw)
        if kw["Key"] == self.key and self.times and ("IfMatch" in kw or "IfNoneMatch" in kw):
            self.times -= 1
            body = {**self.other, "race": self.times}  # a new version (and ETag) each time
            self._real.put_object(Bucket=kw["Bucket"], Key=self.key, Body=json.dumps(body).encode())
        return self._real.put_object(**kw)


def racing(session: Session, monkeypatch: pytest.MonkeyPatch, key: str, **kw: Any) -> Racing:
    client = Racing(boto3.client("s3", region_name="us-east-1"), key, **kw)
    real = session.client
    monkeypatch.setattr(
        session, "client", lambda svc, region=None: client if svc == "s3" else real(svc)
    )
    return client


def test_concurrent_update_keeps_both_changes(
    s3_session: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _ = s3_session
    registry.save_job(session, record())
    theirs = {**registry.load_model(session, "support-nova-v1"), "deployment": {"arn": "theirs"}}
    s3 = racing(session, monkeypatch, "manifests/models/support-nova-v1.json", other=theirs)

    registry.update_model(session, "support-nova-v1", eval={"score": 0.9})

    model = registry.load_model(session, "support-nova-v1")
    assert model["deployment"] == {"arn": "theirs"}  # not overwritten
    assert model["eval"] == {"score": 0.9}
    assert [("IfMatch" in p) for p in s3.puts] == [True, True]  # lost once, then retried
    assert s3.puts[0]["SSEKMSKeyId"] == session.settings.kms_key_arn


def test_first_write_of_a_new_manifest_is_conditional_too(
    s3_session: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _ = s3_session
    theirs = {"name": "support-nova-v1", "base_model": "nova-micro", "jobs": ["other-job"]}
    s3 = racing(session, monkeypatch, "manifests/models/support-nova-v1.json", other=theirs)

    registry.save_job(session, record())  # both processes create the model at once

    model_puts = [p for p in s3.puts if p["Key"].startswith("manifests/models/")]
    assert model_puts[0]["IfNoneMatch"] == "*" and "IfMatch" in model_puts[1]
    assert registry.load_model(session, "support-nova-v1")["jobs"] == [
        "other-job",
        "support-nova-v1-abcd1234",
    ]


def test_environment_changes_retry_on_conflict(
    s3_session: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _ = s3_session
    for name in ("v1", "v2", "v3"):
        registry.save_job(session, record(name, f"{name}-1"))
    env = Environment("support@prod", session=session)
    env.promote("v1")
    theirs = {
        "name": "support",
        "envs": {"prod": {"targets": {"v1": 100}}, "dev": {"targets": {"v3": 100}}},
    }
    racing(session, monkeypatch, "manifests/envs/support.json", other=theirs)

    env.promote("v2")

    assert env.targets == {"v2": 100}
    assert Environment("support@dev", session=session).targets == {"v3": 100}  # kept


def test_gives_up_when_the_manifest_keeps_changing(
    s3_session: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _ = s3_session
    registry.save_job(session, record())
    monkeypatch.setattr(tagging.time, "sleep", lambda s: None)
    racing(session, monkeypatch, "manifests/models/support-nova-v1.json", other={}, times=99)
    with pytest.raises(SkyhammerError, match="kept changing"):
        registry.update_model(session, "support-nova-v1", eval={"score": 0.9})


def test_update_of_a_missing_model_writes_nothing(s3_session: Any) -> None:  # noqa: F811
    session, _ = s3_session
    with pytest.raises(SkyhammerError, match="No skyhammer model named 'ghost'"):
        registry.update_model(session, "ghost", eval={})
    assert registry.load_model(session, "ghost") is None
