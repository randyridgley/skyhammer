"""Batch inference: discounted, asynchronous Converse calls over many requests.

    job = LLM("nova-micro").batch([{"custom_id": "q1", "messages": [...]}, ...], name="eval-1")
    job.wait()
    for r in job.results():
        print(r.custom_id, r.completion.choices[0].message.content if r.completion else r.error)

Records use Bedrock's Converse batch format (``modelInvocationType="Converse"``), built with
the same mapping as live chat, so tools/structured output/etc. work per record. Bedrock
limits (Service Quotas, 2026-10-29): 100-100,000 records per job, 1 GB per input file.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skyhammer import tags as tagging
from skyhammer.errors import JobFailedError, SkyhammerError
from skyhammer.inference import ChatCompletion, from_converse_response, to_converse_request

if TYPE_CHECKING:
    from skyhammer.session import Session

MIN_RECORDS = 100
MAX_RECORDS = 100_000
MAX_FILE_BYTES = 1 << 30
_TERMINAL = {"Completed", "PartiallyCompleted", "Failed", "Stopped", "Expired"}


@dataclass
class BatchResult:
    custom_id: str
    completion: ChatCompletion | None
    error: str | None = None


@dataclass(frozen=True)
class BatchStatus:
    status: str
    message: str | None
    total: int | None
    processed: int | None
    succeeded: int | None
    failed: int | None

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL


def build_records(
    model_id: str, requests: list[dict[str, Any]], defaults: dict[str, Any]
) -> list[dict[str, Any]]:
    """OpenAI-style requests -> Converse batch records. Validates Bedrock's record limits."""
    if not MIN_RECORDS <= len(requests) <= MAX_RECORDS:
        raise SkyhammerError(
            f"Bedrock batch jobs need {MIN_RECORDS:,}-{MAX_RECORDS:,} records; got "
            f"{len(requests):,}. Use live calls for small jobs or split large ones."
        )
    records, seen = [], set()
    for i, r in enumerate(requests):
        rid = str(r.get("custom_id") or f"{i:08d}")
        if rid in seen:
            raise SkyhammerError(f"Duplicate custom_id {rid!r}")
        seen.add(rid)
        params = {**defaults, **{k: v for k, v in r.items() if k not in ("custom_id", "messages")}}
        req = to_converse_request(model_id, r["messages"], **params)
        req.pop("modelId")
        records.append({"recordId": rid, "modelInput": req})
    return records


def batch_name(model_id: str, records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(json.dumps([model_id, records], sort_keys=True).encode()).hexdigest()
    return f"skyhammer-batch-{digest[:12]}"


class BatchJob:
    def __init__(
        self,
        session: Session,
        name: str,
        job_arn: str,
        model_id: str,
        model_name: str | None = None,
    ) -> None:
        self.session = session
        self.name = name
        self.job_arn = job_arn
        self.model_id = model_id
        self.model_name = model_name  # display name, for batch-tier pricing

    @property
    def _bedrock(self) -> Any:
        return self.session.client("bedrock")

    @classmethod
    def submit(
        cls,
        session: Session,
        model_id: str,
        requests: list[dict[str, Any]],
        *,
        name: str | None = None,
        model_name: str | None = None,
        timeout_hours: int = 24,
        defaults: dict[str, Any] | None = None,
    ) -> BatchJob:
        records = build_records(model_id, requests, defaults or {})
        name = name or batch_name(model_id, records)
        existing = cls._find(name, session, model_name)
        if existing:
            return existing  # idempotent: same name -> same job
        body = b"".join(json.dumps(r).encode() + b"\n" for r in records)
        if len(body) > MAX_FILE_BYTES:
            raise SkyhammerError(
                f"Batch input is {len(body) / 2**30:.2f} GB; Bedrock allows 1 GB "
                "per file. Split the requests across jobs."
            )
        res = session.resources
        prefix = f"batch/{name}"
        extra = (
            {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": res.kms_key_arn}
            if res.kms_key_arn
            else {}
        )
        session.client("s3").put_object(
            Bucket=res.bucket, Key=f"{prefix}/input.jsonl", Body=body, **extra
        )
        output: dict[str, Any] = {"s3Uri": f"s3://{res.bucket}/{prefix}/output/"}
        if res.kms_key_arn:
            output["s3EncryptionKeyId"] = res.kms_key_arn
        resp = session.client("bedrock").create_model_invocation_job(
            jobName=name,
            roleArn=res.bedrock_role_arn,
            modelId=model_id,
            modelInvocationType="Converse",
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3Uri": f"s3://{res.bucket}/{prefix}/input.jsonl",
                    "s3InputFormat": "JSONL",
                }
            },
            outputDataConfig={"s3OutputDataConfig": output},
            timeoutDurationInHours=timeout_hours,
            tags=tagging.as_bedrock(
                tagging.build_tags({"name": name, "kind": "batch"}, session.settings.tags)
            ),
        )
        return cls(session, name, resp["jobArn"], model_id, model_name)

    @classmethod
    def _find(cls, name: str, session: Session, model_name: str | None = None) -> BatchJob | None:
        resp = session.client("bedrock").list_model_invocation_jobs(nameContains=name)
        for j in resp.get("invocationJobSummaries", []):
            if j["jobName"] == name:
                return cls(session, name, j["jobArn"], j["modelId"], model_name)
        return None

    @classmethod
    def get(
        cls, name: str, *, session: Session | None = None, model_name: str | None = None
    ) -> BatchJob:
        """Reattach to a batch job by name."""
        from skyhammer.session import get_session

        job = cls._find(name, session or get_session(), model_name)
        if job is None:
            raise SkyhammerError(f"No batch job named {name!r}")
        return job

    def status(self) -> BatchStatus:
        j = self._bedrock.get_model_invocation_job(jobIdentifier=self.job_arn)
        return BatchStatus(
            j["status"],
            j.get("message"),
            j.get("totalRecordCount"),
            j.get("processedRecordCount"),
            j.get("successRecordCount"),
            j.get("errorRecordCount"),
        )

    def wait(
        self,
        *,
        poll_seconds: float = 60,
        timeout_seconds: float | None = None,
        on_status: Any = None,
    ) -> BatchJob:
        deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
        while True:
            st = self.status()
            if on_status:
                on_status(st)
            if st.status in ("Completed", "PartiallyCompleted"):
                return self
            if st.terminal:
                raise JobFailedError(self.name, st.message or st.status)
            if deadline and time.monotonic() > deadline:
                raise TimeoutError(f"Batch {self.name!r} still {st.status} after timeout")
            time.sleep(poll_seconds)

    def cancel(self) -> None:
        self._bedrock.stop_model_invocation_job(jobIdentifier=self.job_arn)

    def results(self) -> list[BatchResult]:
        """Per-record completions (cost at batch-tier prices when known), in input order."""
        from skyhammer.pricing import InferencePrices

        res = self.session.resources
        s3 = self.session.client("s3")
        prefix = f"batch/{self.name}/output/"
        lines: list[str] = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=res.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".jsonl.out"):
                    body = s3.get_object(Bucket=res.bucket, Key=obj["Key"])["Body"].read()
                    lines += body.decode().splitlines()
        prices = None
        if self.model_name and self.session.settings.track_costs:
            try:
                prices = InferencePrices.for_session(self.session)
            except Exception:
                prices = None
        out = []
        for line in filter(None, lines):
            rec = json.loads(line)
            if rec.get("error") or not rec.get("modelOutput"):
                err = rec.get("error") or {}
                out.append(
                    BatchResult(
                        rec["recordId"],
                        None,
                        err.get("errorMessage") if isinstance(err, dict) else str(err),
                    )
                )
                continue
            completion = from_converse_response(self.model_id, rec["modelOutput"])
            if prices and self.model_name:
                u = completion.usage
                u.cost_usd = prices.cost(
                    self.model_name, u.prompt_tokens, u.completion_tokens, tier="batch"
                )
            out.append(BatchResult(rec["recordId"], completion))
        return sorted(out, key=lambda r: r.custom_id)
