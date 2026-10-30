import json
from typing import Any

import boto3
import pytest
from botocore.validate import validate_parameters
from moto import mock_aws

from skyhammer import LLM
from skyhammer.errors import SkyhammerError
from skyhammer.evaluation import EvalJob, eval_records
from skyhammer.session import Session

RECORDS = [
    {"messages": [{"role": "user", "content": "Capital of France?"}], "answer": "Paris"},
    {
        "messages": [
            {"role": "system", "content": "One word."},
            {"role": "user", "content": "Red planet?"},
            {"role": "assistant", "content": "Mars"},
        ],
        "category": "geo",
    },
]


def test_records_map_to_bedrock_prompt_dataset() -> None:
    rows = eval_records(RECORDS)
    assert rows[0] == {"prompt": "Capital of France?", "referenceResponse": "Paris"}
    assert rows[1]["prompt"] == "System: One word.\n\nUser: Red planet?"
    assert rows[1]["referenceResponse"] == "Mars" and rows[1]["category"] == "geo"
    byo = eval_records(RECORDS, ["Paris.", "Mars"], "my-model")
    assert byo[0]["modelResponses"] == [{"response": "Paris.", "modelIdentifier": "my-model"}]
    with pytest.raises(SkyhammerError, match="1-1,000 prompts"):
        eval_records(RECORDS * 501)


class FakeBedrock:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}

    def create_evaluation_job(self, **kw: Any) -> dict[str, Any]:
        shape = boto3.client("bedrock", region_name="us-east-1").meta.service_model
        validate_parameters(kw, shape.operation_model("CreateEvaluationJob").input_shape)
        arn = f"arn:aws:bedrock:us-east-1:1:evaluation-job/{kw['jobName']}"
        self.jobs[kw["jobName"]] = {**kw, "jobArn": arn}
        return {"jobArn": arn}

    def get_evaluation_job(self, jobIdentifier: str) -> dict[str, Any]:
        return {
            "status": next(
                j.get("status", "InProgress")
                for j in self.jobs.values()
                if j["jobArn"] == jobIdentifier
            )
        }

    def list_evaluation_jobs(self, nameContains: str) -> dict[str, Any]:
        return {
            "jobSummaries": [
                {"jobName": n, "jobArn": j["jobArn"]}
                for n, j in self.jobs.items()
                if nameContains in n
            ]
        }


def test_submit_builds_judge_job_and_is_idempotent(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=session.settings.bucket or "")
        fake = FakeBedrock()
        real = session.client
        monkeypatch.setattr(
            session,
            "client",
            lambda svc, region=None: fake if svc == "bedrock" else real(svc, region),
        )
        llm = LLM("nova-micro", session=session)
        job = llm.create_eval_job(
            RECORDS, metrics=["correctness", "Builtin.Refusal"], judge="nova-2-lite"
        )
        req = fake.jobs[job.name]
        auto = req["evaluationConfig"]["automated"]
        assert auto["datasetMetricConfigs"][0]["metricNames"] == [
            "Builtin.Correctness",
            "Builtin.Refusal",
        ]
        assert auto["evaluatorModelConfig"]["bedrockEvaluatorModels"][0]["modelIdentifier"] == (
            LLM("nova-2-lite", session=session).model_id
        )
        model = req["inferenceConfig"]["models"][0]["bedrockModel"]
        assert model["modelIdentifier"] == llm.model_id
        assert json.loads(model["inferenceParams"]) == {"inferenceConfig": {"maxTokens": 1024}}
        assert req["customerEncryptionKeyId"] == session.settings.kms_key_arn
        assert {"key": "skyhammer:kind", "value": "eval"} in req["jobTags"]
        body = s3.get_object(Bucket=session.settings.bucket, Key=f"evals/{job.name}/prompts.jsonl")
        assert len(body["Body"].read().splitlines()) == 2
        again = llm.create_eval_job(
            RECORDS, metrics=["correctness", "Builtin.Refusal"], judge="nova-2-lite"
        )
        assert again.name == job.name and len(fake.jobs) == 1
        fake.jobs[job.name]["status"] = "Failed"
        retry = llm.create_eval_job(
            RECORDS, metrics=["correctness", "Builtin.Refusal"], judge="nova-2-lite"
        )
        assert retry.name.startswith(job.name + "-") and len(fake.jobs) == 2

        with pytest.raises(SkyhammerError, match="Unknown judge metric"):
            llm.create_eval_job(RECORDS, metrics=["vibes"])


def test_precomputed_responses_for_models_bedrock_cannot_invoke(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skyhammer import evaluation

    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(
            Bucket=session.settings.bucket or ""
        )
        fake = FakeBedrock()
        real = session.client
        monkeypatch.setattr(
            session,
            "client",
            lambda svc, region=None: fake if svc == "bedrock" else real(svc, region),
        )
        monkeypatch.setattr(
            evaluation,
            "evaluate",
            lambda llm, records, graders, max_tokens: evaluation.EvalResult(
                "m", outputs=["Paris", "Mars"]
            ),
        )
        job = EvalJob.submit(
            LLM("nova-micro", session=session),
            RECORDS,
            metrics=["correctness"],
            judge="nova-2-lite",
            precomputed=True,
        )
        model = fake.jobs[job.name]["inferenceConfig"]["models"][0]
        assert model == {"precomputedInferenceSource": {"inferenceSourceIdentifier": "nova-micro"}}


def test_parse_real_judge_output() -> None:
    """Output of a live Bedrock LLM-as-judge job (Nova Micro judged by Nova Pro, 2026-10-29)."""
    from pathlib import Path

    from skyhammer.evaluation import EvalReport, parse_eval_output

    lines = (Path(__file__).parents[1] / "fixtures" / "eval_judge_output.jsonl").read_text()
    records = parse_eval_output(lines.splitlines())
    assert len(records) == 5
    first = records[0]
    assert first.prompt == "Capital of France?" and first.category == "geo"
    assert "Paris" in (first.response or "")
    assert set(first.scores) == {"correctness", "completeness", "helpfulness"}
    assert all(0 <= v <= 1 for r in records for v in r.scores.values() if v is not None)
    assert first.explanations.get("correctness")
    means = EvalReport(records).means
    assert means["correctness"] == 1.0 and 0.9 < means["helpfulness"] < 0.95
