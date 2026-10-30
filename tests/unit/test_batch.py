import json
from typing import Any

import boto3
import pytest
from botocore.validate import validate_parameters
from moto import mock_aws

from skyhammer.batch import BatchJob, build_records
from skyhammer.errors import SkyhammerError
from skyhammer.pricing import InferencePrices
from skyhammer.session import Session


def reqs(n: int) -> list[dict[str, Any]]:
    return [
        {"custom_id": f"q{i:03d}", "messages": [{"role": "user", "content": f"q{i}"}]}
        for i in range(n)
    ]


def test_record_limits_and_converse_format() -> None:
    with pytest.raises(SkyhammerError, match="100-100,000 records"):
        build_records("m", reqs(99), {})
    recs = build_records("m", reqs(100), {"max_tokens": 50})
    assert recs[0]["recordId"] == "q000" and "modelId" not in recs[0]["modelInput"]
    assert recs[0]["modelInput"]["inferenceConfig"] == {"maxTokens": 50}
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    with_tools = build_records("m", [{**r, "tools": [tool]} for r in reqs(100)], {})
    assert with_tools[0]["modelInput"]["toolConfig"]["tools"][0]["toolSpec"]["name"] == "f"
    dup = reqs(100)
    dup[1]["custom_id"] = "q000"
    with pytest.raises(SkyhammerError, match="Duplicate custom_id"):
        build_records("m", dup, {})


class FakeBedrock:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}

    def create_model_invocation_job(self, **kw: Any) -> dict[str, Any]:
        client = boto3.client("bedrock", region_name="us-east-1")
        validate_parameters(
            kw, client.meta.service_model.operation_model("CreateModelInvocationJob").input_shape
        )
        arn = f"arn:aws:bedrock:us-east-1:1:model-invocation-job/{kw['jobName']}"
        self.jobs[kw["jobName"]] = {**kw, "jobArn": arn}
        return {"jobArn": arn}

    def list_model_invocation_jobs(self, nameContains: str) -> dict[str, Any]:
        return {
            "invocationJobSummaries": [
                {"jobName": n, "jobArn": j["jobArn"], "modelId": j["modelId"]}
                for n, j in self.jobs.items()
                if nameContains in n
            ]
        }


def test_submit_uploads_kms_encrypted_input_and_is_idempotent(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=session.settings.bucket or "")
        fake = FakeBedrock()
        real_client = session.client
        monkeypatch.setattr(
            session,
            "client",
            lambda svc, region=None: fake if svc == "bedrock" else real_client(svc, region),
        )
        job = BatchJob.submit(session, "us.amazon.nova-micro-v1:0", reqs(100), name="eval-1")
        created = fake.jobs["eval-1"]
        assert created["modelInvocationType"] == "Converse"
        assert created["roleArn"] == session.settings.bedrock_role_arn
        assert (
            created["outputDataConfig"]["s3OutputDataConfig"]["s3EncryptionKeyId"]
            == session.settings.kms_key_arn
        )
        head = s3.head_object(Bucket=session.settings.bucket, Key="batch/eval-1/input.jsonl")
        assert head["ServerSideEncryption"] == "aws:kms"
        again = BatchJob.submit(session, "us.amazon.nova-micro-v1:0", reqs(100), name="eval-1")
        assert again.job_arn == job.job_arn and len(fake.jobs) == 1


def test_results_parse_errors_and_batch_tier_cost(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = [
        {
            "recordId": "q001",
            "modelInput": {},
            "modelOutput": {
                "output": {"message": {"content": [{"text": "b"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
            },
        },
        {
            "recordId": "q000",
            "modelInput": {},
            "modelOutput": {
                "output": {"message": {"content": [{"text": "a"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
            },
        },
        {
            "recordId": "q002",
            "modelInput": {},
            "error": {"errorCode": 400, "errorMessage": "bad request"},
        },
    ]
    rows = [
        {
            "model": "nova micro",
            "kind": k,
            "tier": "batch",
            "routing": "regional",
            "usd_per_token": p,
        }
        for k, p in (("input", 1e-6), ("output", 2e-6))
    ]
    monkeypatch.setattr(
        InferencePrices, "for_session", classmethod(lambda cls, s: InferencePrices(rows))
    )
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=session.settings.bucket or "")
        s3.put_object(
            Bucket=session.settings.bucket,
            Key="batch/eval-1/output/abc/input.jsonl.out",
            Body="\n".join(json.dumps(r) for r in out).encode(),
        )
        job = BatchJob(session, "eval-1", "arn", "m", model_name="Nova Micro")
        results = job.results()
    assert [r.custom_id for r in results] == ["q000", "q001", "q002"]  # input order
    assert results[0].completion and results[0].completion.choices[0].message.content == "a"
    assert results[0].completion.usage.cost_usd == pytest.approx(100 * 1e-6 + 10 * 2e-6)
    assert results[2].completion is None and results[2].error == "bad request"


def test_real_batch_output_with_null_union_members_parses() -> None:
    """Regression (live 2026-10-29): batch output sets every absent content field to null."""
    from skyhammer.inference import from_converse_response
    from tests.conftest import FIXTURES

    for line in (FIXTURES / "batch_output_converse.jsonl").read_text().splitlines():
        rec = json.loads(line)
        r = from_converse_response("m", rec["modelOutput"])
        assert r.choices[0].message.content and r.choices[0].message.tool_calls is None
        assert r.usage.total_tokens > 0 and r.usage.cache_read_tokens == 0
