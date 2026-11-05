"""Tag schema and lineage manifests. Tags are the SDK's only state on AWS resources;
anything too long for a tag goes in a JSON manifest in the bootstrap bucket."""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from typing import Any

from botocore.exceptions import ClientError

from skyhammer.errors import SkyhammerError

PREFIX = "skyhammer:"
MAX_KEY = 128
MAX_VALUE = 256
_ALLOWED = re.compile(r"[^\w\s.:/=+\-@]")


def _clean(value: str, limit: int) -> str:
    return _ALLOWED.sub("_", value)[:limit]


def build_tags(sdk: dict[str, str], user: dict[str, str] | None = None) -> dict[str, str]:
    """Merge SDK lineage tags (prefixed ``skyhammer:``) with user/budget tags.

    SDK tags win on conflict so lineage can't be overwritten accidentally.
    """
    tags = {_clean(k, MAX_KEY): _clean(v, MAX_VALUE) for k, v in (user or {}).items()}
    tags.update({_clean(PREFIX + k, MAX_KEY): _clean(v, MAX_VALUE) for k, v in sdk.items()})
    return tags


def as_bedrock(tags: dict[str, str]) -> list[dict[str, str]]:
    return [{"key": k, "value": v} for k, v in tags.items()]


def as_aws(tags: dict[str, str]) -> list[dict[str, str]]:
    """CloudFormation/SageMaker/S3 style."""
    return [{"Key": k, "Value": v} for k, v in tags.items()]


def manifest_key(kind: str, name: str) -> str:
    return f"manifests/{kind}/{name}.json"


def _put(
    s3: Any, bucket: str, key: str, body: dict[str, Any], kms_key_arn: str | None, **cond: str
) -> None:
    extra: dict[str, str] = (
        {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key_arn} if kms_key_arn else {}
    )
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(body, indent=2, default=str).encode(),
        ContentType="application/json",
        **extra,
        **cond,
    )


def write_manifest(
    s3: Any, bucket: str, kind: str, name: str, body: dict[str, Any], kms_key_arn: str | None = None
) -> str:
    """Write a whole manifest, replacing any other version (for single-writer records)."""
    key = manifest_key(kind, name)
    _put(s3, bucket, key, body, kms_key_arn)
    return f"s3://{bucket}/{key}"


# S3 answers a conditional write that lost a race with 412, or 409 while another
# conditional write to the same key is in flight.
_CONFLICTS = ("PreconditionFailed", "ConditionalRequestConflict")


def update_manifest(
    s3: Any,
    bucket: str,
    kind: str,
    name: str,
    change: Callable[[dict[str, Any] | None], dict[str, Any] | None],
    kms_key_arn: str | None = None,
    *,
    attempts: int = 8,
) -> dict[str, Any] | None:
    """Read, change and write a manifest without losing another process's update.

    ``change`` gets the current manifest (None if there is none) and returns the new one,
    or None to leave it as it is. The write is conditional (S3 ``If-Match`` on the ETag
    read, or ``If-None-Match: *`` for a new manifest); if another writer got there first,
    the manifest is read again and ``change`` reapplied to the newer version.
    """
    key = manifest_key(kind, name)
    for attempt in range(attempts):
        current: dict[str, Any] | None
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            current = json.loads(obj["Body"].read())
            cond = {"IfMatch": obj["ETag"]}
        except s3.exceptions.NoSuchKey:
            current, cond = None, {"IfNoneMatch": "*"}
        body = change(current)
        if body is None:
            return current
        try:
            _put(s3, bucket, key, body, kms_key_arn, **cond)
            return body
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") not in _CONFLICTS:
                raise
        time.sleep(random.uniform(0, min(0.05 * 2**attempt, 2)))
    raise SkyhammerError(
        f"s3://{bucket}/{key} kept changing while being updated ({attempts} attempts); "
        "another process is writing it. Try again."
    )


def read_manifest(s3: Any, bucket: str, kind: str, name: str) -> dict[str, Any] | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=manifest_key(kind, name))
    except s3.exceptions.NoSuchKey:
        return None
    result: dict[str, Any] = json.loads(obj["Body"].read())
    return result
