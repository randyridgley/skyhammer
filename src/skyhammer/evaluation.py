"""Evaluation: a local harness for fast iteration, Bedrock eval jobs (LLM as a judge) at scale."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import fmean
from typing import TYPE_CHECKING, Any

from skyhammer import tags as tagging
from skyhammer.errors import JobFailedError, SkyhammerError
from skyhammer.formats import message_text

if TYPE_CHECKING:
    from skyhammer.graders import Grader
    from skyhammer.llm import LLM
    from skyhammer.session import Session


@dataclass
class EvalResult:
    model: str
    scores: dict[str, list[float]] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)

    @property
    def means(self) -> dict[str, float]:
        return {k: fmean(v) for k, v in self.scores.items() if v}


def evaluate(
    llm: LLM,
    records: list[dict[str, Any]],
    graders: list[Grader],
    *,
    max_tokens: int = 1024,
    on_progress: Callable[[int], None] | None = None,
) -> EvalResult:
    """Run each record's prompt (all messages except a trailing assistant turn) and grade.

    Graders receive the full record, so reference answers can live in any field.
    """
    result = EvalResult(model=llm.model_id, scores={g.name: [] for g in graders})
    for i, record in enumerate(records):
        messages = record["messages"]
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        resp = llm.chat.completions.create(messages=messages, max_tokens=max_tokens)
        output = message_text(resp.choices[0].message.content)
        result.outputs.append(output)
        for g in graders:
            result.scores[g.name].append(float(g(record, output)))
        if on_progress:
            on_progress(i + 1)
    return result


# ---- managed evaluation jobs (Bedrock Model Evaluation, LLM as a judge) ----

JUDGE_METRICS = {
    "correctness": "Builtin.Correctness",
    "completeness": "Builtin.Completeness",
    "faithfulness": "Builtin.Faithfulness",
    "helpfulness": "Builtin.Helpfulness",
    "coherence": "Builtin.Coherence",
    "relevance": "Builtin.Relevance",
    "following_instructions": "Builtin.FollowingInstructions",
    "professional_style": "Builtin.ProfessionalStyleAndTone",
    "harmfulness": "Builtin.Harmfulness",
    "stereotyping": "Builtin.Stereotyping",
    "refusal": "Builtin.Refusal",
}
MAX_EVAL_PROMPTS = 1000
_EVAL_TERMINAL = {"Completed", "Failed", "Stopped"}


def _metric_name(metric: str) -> str:
    if metric.startswith("Builtin."):
        return metric
    try:
        return JUDGE_METRICS[metric.lower()]
    except KeyError:
        raise SkyhammerError(
            f"Unknown judge metric {metric!r}; choose from {', '.join(JUDGE_METRICS)}"
        ) from None


def _prompt_text(messages: list[dict[str, Any]]) -> str:
    """Bedrock eval prompts are a single string; multi-turn chats are rendered as a transcript."""
    if len(messages) == 1 and messages[0]["role"] == "user":
        return message_text(messages[0]["content"])
    return "\n\n".join(f"{m['role'].title()}: {message_text(m['content'])}" for m in messages)


def eval_records(
    records: list[dict[str, Any]], responses: list[str] | None = None, source: str = ""
) -> list[dict[str, Any]]:
    """OpenAI-style records -> Bedrock eval prompt dataset rows.

    The reference answer is the record's trailing assistant turn, or its ``reference`` /
    ``answer`` field. ``category`` passes through so scores are reported per category.
    """
    if not 1 <= len(records) <= MAX_EVAL_PROMPTS:
        raise SkyhammerError(
            f"Bedrock evaluation jobs take 1-{MAX_EVAL_PROMPTS:,} prompts; got {len(records):,}"
        )
    rows = []
    for i, record in enumerate(records):
        messages = record["messages"]
        reference = record.get("reference") or record.get("answer")
        if messages and messages[-1]["role"] == "assistant":
            reference = reference or message_text(messages[-1]["content"])
            messages = messages[:-1]
        row: dict[str, Any] = {"prompt": _prompt_text(messages)}
        if reference:
            row["referenceResponse"] = str(reference)
        if record.get("category"):
            row["category"] = str(record["category"])
        if responses is not None:
            row["modelResponses"] = [{"response": responses[i], "modelIdentifier": source}]
        rows.append(row)
    return rows


def eval_job_name(model_id: str, rows: list[dict[str, Any]], metrics: list[str], judge: str) -> str:
    blob = json.dumps([model_id, rows, metrics, judge], sort_keys=True)
    return f"skyhammer-eval-{hashlib.sha256(blob.encode()).hexdigest()[:12]}"


@dataclass(frozen=True)
class EvalJobStatus:
    status: str
    message: str | None

    @property
    def terminal(self) -> bool:
        return self.status in _EVAL_TERMINAL


@dataclass
class EvalRecordScore:
    prompt: str
    response: str | None
    scores: dict[str, float | None]
    explanations: dict[str, str]
    category: str | None = None


@dataclass
class EvalReport:
    records: list[EvalRecordScore]

    @property
    def means(self) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for r in self.records:
            for k, v in r.scores.items():
                if v is not None:
                    values.setdefault(k, []).append(v)
        return {k: fmean(v) for k, v in values.items()}


def _short_metric(name: str) -> str:
    full = name if name.startswith("Builtin.") else f"Builtin.{name}"
    return next((k for k, v in JUDGE_METRICS.items() if v == full), name)


def parse_eval_output(lines: list[str]) -> list[EvalRecordScore]:
    out = []
    for line in filter(None, (ln.strip() for ln in lines)):
        rec = json.loads(line)
        inp = rec.get("inputRecord") or {}
        responses = inp.get("modelResponses") or rec.get("modelResponses") or [{}]
        scores: dict[str, float | None] = {}
        explanations: dict[str, str] = {}
        for s in (rec.get("automatedEvaluationResult") or {}).get("scores", []):
            name = _short_metric(s["metricName"])
            scores[name] = s.get("result")
            details = s.get("evaluatorDetails") or []
            if details and details[0].get("explanation"):
                explanations[name] = details[0]["explanation"]
        out.append(
            EvalRecordScore(
                inp.get("prompt", ""),
                responses[0].get("response"),
                scores,
                explanations,
                inp.get("category"),
            )
        )
    return out


class EvalJob:
    """A Bedrock Model Evaluation job (LLM as a judge). Create with ``LLM.create_eval_job``."""

    def __init__(self, session: Session, name: str, arn: str) -> None:
        self.session = session
        self.name = name
        self.arn = arn

    @property
    def _bedrock(self) -> Any:
        return self.session.client("bedrock")

    @classmethod
    def submit(
        cls,
        llm: LLM,
        records: list[dict[str, Any]],
        *,
        metrics: list[str],
        judge: str | LLM = "nova-pro",
        name: str | None = None,
        max_tokens: int = 1024,
        precomputed: bool | None = None,
    ) -> EvalJob:
        from skyhammer.llm import LLM

        session = llm.session
        res = session.resources
        metric_names = [_metric_name(m) for m in metrics]
        judge_llm = judge if isinstance(judge, LLM) else LLM(judge, session=session)
        judge_id = judge_llm.model_id
        model_id = llm.model_id
        if precomputed is None:  # Bedrock can't invoke SageMaker endpoints itself
            precomputed = model_id.startswith("arn:aws:sagemaker:")
        source = re.sub(r"[^A-Za-z0-9_.-]", "-", llm.name)[:60] if precomputed else ""
        responses = None
        if precomputed:
            responses = evaluate(llm, records, [], max_tokens=max_tokens).outputs
        rows = eval_records(records, responses, source)
        name_given = name is not None
        name = name or eval_job_name(model_id, rows, metric_names, judge_id)
        existing = cls._find(session, name)
        if existing:
            if name_given or existing.status().status not in ("Failed", "Stopped"):
                return existing  # same inputs -> same job; reattach
            name = f"{name}-{int(time.time()) % 100000}"  # retry a failed/stopped run
        prefix = f"evals/{name}"
        extra = (
            {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": res.kms_key_arn}
            if res.kms_key_arn
            else {}
        )
        session.client("s3").put_object(
            Bucket=res.bucket,
            Key=f"{prefix}/prompts.jsonl",
            Body=b"".join(json.dumps(r).encode() + b"\n" for r in rows),
            **extra,
        )
        if precomputed:
            model: dict[str, Any] = {
                "precomputedInferenceSource": {"inferenceSourceIdentifier": source}
            }
        else:
            params = {"inferenceConfig": {"maxTokens": max_tokens}}
            model = {
                "bedrockModel": {"modelIdentifier": model_id, "inferenceParams": json.dumps(params)}
            }
        kms = {"customerEncryptionKeyId": res.kms_key_arn} if res.kms_key_arn else {}
        resp = session.client("bedrock").create_evaluation_job(
            jobName=name,
            roleArn=res.bedrock_role_arn,
            applicationType="ModelEvaluation",
            evaluationConfig={
                "automated": {
                    "datasetMetricConfigs": [
                        {
                            "taskType": "General",
                            "dataset": {
                                "name": "prompts",
                                "datasetLocation": {
                                    "s3Uri": f"s3://{res.bucket}/{prefix}/prompts.jsonl"
                                },
                            },
                            "metricNames": metric_names,
                        }
                    ],
                    "evaluatorModelConfig": {
                        "bedrockEvaluatorModels": [{"modelIdentifier": judge_id}]
                    },
                }
            },
            inferenceConfig={"models": [model]},
            outputDataConfig={"s3Uri": f"s3://{res.bucket}/{prefix}/output/"},
            jobTags=tagging.as_bedrock(
                tagging.build_tags({"name": name, "kind": "eval"}, session.settings.tags)
            ),
            **kms,
        )
        return cls(session, name, resp["jobArn"])

    @classmethod
    def _find(cls, session: Session, name: str) -> EvalJob | None:
        resp = session.client("bedrock").list_evaluation_jobs(nameContains=name)
        for j in resp.get("jobSummaries", []):
            if j["jobName"] == name:
                return cls(session, name, j["jobArn"])
        return None

    @classmethod
    def get(cls, name: str, *, session: Session | None = None) -> EvalJob:
        """Reattach to an evaluation job by name."""
        from skyhammer.session import get_session

        job = cls._find(session or get_session(), name)
        if job is None:
            raise SkyhammerError(f"No evaluation job named {name!r}")
        return job

    def status(self) -> EvalJobStatus:
        j = self._bedrock.get_evaluation_job(jobIdentifier=self.arn)
        return EvalJobStatus(j["status"], "; ".join(j.get("failureMessages") or []) or None)

    def wait(
        self,
        *,
        poll_seconds: float = 60,
        timeout_seconds: float | None = None,
        on_status: Any = None,
    ) -> EvalJob:
        deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
        while True:
            st = self.status()
            if on_status:
                on_status(st)
            if st.status == "Completed":
                return self
            if st.terminal:
                raise JobFailedError(self.name, st.message or st.status)
            if st.message and "KMS key" in st.message:  # reported while still InProgress
                raise JobFailedError(
                    self.name,
                    f"{st.message} Run `skyhammer bootstrap` to update the key policy; with "
                    "your own key, allow bedrock.amazonaws.com kms:GenerateDataKey/Decrypt/"
                    "DescribeKey for evaluation-job ARNs.",
                )
            if deadline and time.monotonic() > deadline:
                raise TimeoutError(f"Evaluation {self.name!r} still {st.status} after timeout")
            time.sleep(poll_seconds)

    def cancel(self) -> None:
        self._bedrock.stop_evaluation_job(jobIdentifier=self.arn)

    def results(self) -> EvalReport:
        """Per-prompt judge scores (0-1) and explanations; ``.means`` per metric."""
        res = self.session.resources
        s3 = self.session.client("s3")
        lines: list[str] = []
        prefix = f"evals/{self.name}/output/"
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=res.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".jsonl"):
                    lines += (
                        s3.get_object(Bucket=res.bucket, Key=obj["Key"])["Body"]
                        .read()
                        .decode()
                        .splitlines()
                    )
        return EvalReport(parse_eval_output(lines))
