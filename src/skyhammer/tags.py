"""Tag schema and lineage manifests. Tags are the SDK's only state on AWS resources;
anything too long for a tag goes in a JSON manifest in the bootstrap bucket."""

from __future__ import annotations

import json
import re
from typing import Any

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


def write_manifest(
    s3: Any, bucket: str, kind: str, name: str, body: dict[str, Any], kms_key_arn: str | None = None
) -> str:
    key = manifest_key(kind, name)
    extra: dict[str, str] = (
        {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key_arn} if kms_key_arn else {}
    )
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(body, indent=2, default=str).encode(),
        ContentType="application/json",
        **extra,
    )
    return f"s3://{bucket}/{key}"


def read_manifest(s3: Any, bucket: str, kind: str, name: str) -> dict[str, Any] | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=manifest_key(kind, name))
    except s3.exceptions.NoSuchKey:
        return None
    result: dict[str, Any] = json.loads(obj["Body"].read())
    return result
