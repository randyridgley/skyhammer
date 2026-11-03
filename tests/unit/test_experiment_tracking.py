"""W&B and Langfuse integrations (EV-10), against the real SDKs: W&B in offline mode, Langfuse
with an in-memory OpenTelemetry exporter (scores captured at the client)."""

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import wandb
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from skyhammer.errors import JobFailedError, SkyhammerError
from skyhammer.evaluation import EvalRecordScore, EvalReport, EvalResult
from skyhammer.integrations import langfuse as sh_langfuse
from skyhammer.integrations import mlflow as sh_mlflow
from skyhammer.integrations import wandb as sh_wandb
from skyhammer.jobs import JobState, JobStatus

from .test_framework_adapters import ScriptedLLM

RECORDS = [
    {
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Capital of France?"},
            {"role": "assistant", "content": "Paris"},
        ]
    },
]
RESULT = EvalResult("my-model", {"exact": [1.0, 0.0], "judge": [0.75, 0.25]}, ["4", "Lyon"])
REPORT = EvalReport(
    [
        EvalRecordScore(
            "2+2?", "4", {"correctness": 1.0, "helpfulness": None}, {"correctness": "ok"}
        ),
        EvalRecordScore("Hi", "Hello", {"correctness": 0.5}, {}, category="greeting"),
    ]
)


# ---- W&B ----


class FakeRecord:
    job_name = "support-qwen-v1-ab12"
    output_model = "support-qwen-v1"
    base_model = "qwen3-0.6b"
    backend_model_id = "huggingface-reasoning-qwen3-06b"
    technique = "sft"
    training_type = "lora"
    backend = "sagemaker"
    region = "us-east-1"
    dataset: ClassVar = {"name": "support", "sha256": "abc", "rows": 200}
    hyperparameters: ClassVar = {"epochs": 2, "learning_rate": 1e-4, "beta": None}
    estimate = "$3.10"


class FakeJob:
    """Three new steps per poll; the third poll completes (or fails)."""

    def __init__(self, fail: bool = False) -> None:
        self.record = FakeRecord()
        self.polls = 0
        self.fail = fail

    def metrics(self) -> list[dict[str, float]]:
        rows = [{"step": float(s), "loss": 2.0 / s} for s in range(1, 3 * self.polls + 1)]
        if self.polls >= 3:  # evaluation for step 6 lands a poll after its training loss
            rows[5]["eval_loss"] = 0.5
        return rows

    def status(self) -> JobStatus:
        self.polls += 1
        if self.polls < 3:
            return JobStatus(JobState.TRAINING)
        if self.fail:
            return JobStatus(JobState.FAILED, "OOM")
        return JobStatus(JobState.COMPLETED, output_model_arn="arn:aws:bedrock:model/x")

    def wait(self, *, poll_seconds: float, on_status: Any) -> "FakeJob":
        while True:
            st = self.status()
            on_status(st)
            if st.state is JobState.COMPLETED:
                return self
            if st.state.terminal:
                raise JobFailedError(self.record.job_name, st.message or "")


@pytest.fixture
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    return tmp_path


def _spy(run: Any) -> list[dict[str, Any]]:
    """Record what is logged to ``run`` (still logging it for real)."""
    logged: list[dict[str, Any]] = []
    log = run.log

    def spy(data: dict[str, Any], step: int | None = None, **kw: Any) -> None:
        logged.append({**data, "_step": step})
        log(data, step=step, **kw)

    run.log = spy
    return logged


def test_log_job_follows_and_streams_metrics(offline: Path) -> None:
    run = wandb.init(project="skyhammer")
    history = _spy(run)
    assert sh_wandb.log_job(FakeJob(), run=run, follow=True, poll_seconds=0) is run  # type: ignore[arg-type]
    assert run.config["technique"] == "sft"
    assert run.config["hyperparameters"] == {"epochs": 2, "learning_rate": 1e-4}
    assert run.summary["skyhammer/state"] == "completed"
    assert run.summary["skyhammer/output_model_arn"] == "arn:aws:bedrock:model/x"
    run.finish()
    steps = [r["_step"] for r in history]
    assert steps == sorted(set(steps)) == [1, 2, 3, 4, 5, 6, 7, 8, 9]  # each step once
    assert next(r for r in history if r["_step"] == 6)["eval_loss"] == 0.5


def test_log_job_failure_marks_run_and_reraises(offline: Path) -> None:
    run = wandb.init(project="skyhammer")
    with pytest.raises(JobFailedError):
        sh_wandb.log_job(FakeJob(fail=True), run=run, follow=True, poll_seconds=0)  # type: ignore[arg-type]
    assert run.summary["skyhammer/state"] == "failed"
    run.finish()


def test_log_eval_local_and_managed(offline: Path) -> None:
    run = wandb.init(project="skyhammer")
    history = _spy(run)
    sh_wandb.log_eval(RESULT, RECORDS, run=run)
    assert run.summary["eval/exact"] == 0.5 and run.summary["eval/judge"] == 0.5
    table = history[0]["eval/samples"]
    assert table.columns == [
        "prompt",
        "expected",
        "output",
        "category",
        "exact",
        "judge",
        "comments",
    ]
    assert table.data[1][:6] == [
        '[{"role": "user", "content": "Capital of France?"}]',
        "Paris",
        "Lyon",
        None,
        0.0,
        0.25,
    ]
    run.finish()

    owned = sh_wandb.log_eval(REPORT, key="judge")  # starts and finishes its own run
    assert owned is not None and wandb.run is None
    sh_wandb.log_eval(REPORT, key="judge", run=(run := wandb.init(project="skyhammer")))
    assert run.summary["judge/correctness"] == 0.75
    run.finish()


def test_eval_rows() -> None:
    from skyhammer.integrations._eval import eval_rows

    rows = eval_rows(RESULT, RECORDS)
    assert rows[1].prompt == [{"role": "user", "content": "Capital of France?"}]
    assert (rows[1].expected, rows[1].output, rows[1].scores) == (
        "Paris",
        "Lyon",
        {"exact": 0.0, "judge": 0.25},
    )
    assert eval_rows(RESULT)[0].prompt is None
    assert eval_rows(REPORT)[0].comments == {"correctness": "ok"}
    with pytest.raises(SkyhammerError, match="records has 1 rows"):
        eval_rows(RESULT, RECORDS[:1])


# ---- Langfuse ----


@pytest.fixture
def langfuse() -> tuple[Langfuse, InMemorySpanExporter, list[dict[str, Any]]]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = Langfuse(
        public_key=f"pk-lf-{uuid.uuid4().hex}",  # clients are cached per key
        secret_key="sk-lf-test",
        host="http://127.0.0.1:9",
        tracer_provider=provider,
        span_exporter=exporter,
        flush_at=1,
    )
    scores: list[dict[str, Any]] = []
    client.create_score = lambda **kw: scores.append(kw)  # type: ignore[method-assign]
    return client, exporter, scores


def _attrs(exporter: InMemorySpanExporter) -> list[dict[str, Any]]:
    return [dict(s.attributes or {}) for s in exporter.get_finished_spans()]


def test_traced_records_a_generation_per_call(langfuse: Any) -> None:
    client, exporter, _ = langfuse
    model = sh_langfuse.traced(ScriptedLLM(), client=client, metadata={"team": "support"})
    reply = model.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}], max_tokens=20, tools=None
    )
    assert reply.choices[0].message.content == "echo: hi"
    (span,) = _attrs(exporter)
    assert span["langfuse.observation.type"] == "generation"
    assert json.loads(span["langfuse.observation.input"]) == [{"role": "user", "content": "hi"}]
    assert json.loads(span["langfuse.observation.output"])["content"] == "echo: hi"
    assert json.loads(span["langfuse.observation.usage_details"]) == {
        "input": 7,
        "output": 3,
        "total": 10,
    }
    assert json.loads(span["langfuse.observation.cost_details"]) == {"total": 0.0002}
    assert json.loads(span["langfuse.observation.model.parameters"]) == {"max_tokens": 20}
    assert json.loads(span["langfuse.observation.metadata.team"]) == "support"


def test_traced_stream_and_error(langfuse: Any) -> None:
    client, exporter, _ = langfuse
    model = sh_langfuse.traced(ScriptedLLM(), client=client)
    stream = model.chat.completions.create(messages=[{"role": "user", "content": "x"}], stream=True)
    assert stream.final().choices[0].message.content == "Hello"
    (span,) = _attrs(exporter)
    assert json.loads(span["langfuse.observation.output"])["content"] == "Hello"
    assert "langfuse.observation.completion_start_time" in span
    assert json.loads(span["langfuse.observation.usage_details"])["total"] == 10

    class Broken:
        name = "broken"
        chat = type("C", (), {"completions": None})()

        def __init__(self) -> None:
            def boom(**_: Any) -> Any:
                raise SkyhammerError("throttled")

            self.chat.completions = type("X", (), {"create": staticmethod(boom)})()

    with pytest.raises(SkyhammerError):
        sh_langfuse.traced(Broken(), client=client).chat.completions.create(messages=[])
    failed = _attrs(exporter)[-1]
    assert failed["langfuse.observation.level"] == "ERROR"
    assert "throttled" in failed["langfuse.observation.status_message"]


def test_traced_passes_attributes_through(langfuse: Any) -> None:
    llm = ScriptedLLM()
    llm.model_id = "us.amazon.nova-micro-v1:0"  # type: ignore[attr-defined]
    assert sh_langfuse.traced(llm, client=langfuse[0]).model_id == "us.amazon.nova-micro-v1:0"


def test_langfuse_log_eval_scores_each_sample(langfuse: Any) -> None:
    client, exporter, scores = langfuse
    ids = sh_langfuse.log_eval(RESULT, RECORDS, client=client, run="v2 vs base")
    assert len(ids) == 2 and len(set(ids)) == 2
    spans = _attrs(exporter)
    assert all(s["langfuse.observation.type"] == "evaluator" for s in spans)
    # Spans can reach the exporter in either order; find the Paris sample's.
    (paris,) = [
        s for s in spans if json.loads(s["langfuse.observation.metadata.expected"]) == "Paris"
    ]
    assert json.loads(paris["langfuse.observation.metadata.run"]) == "v2 vs base"
    assert [(s["name"], s["value"], s["trace_id"]) for s in scores[2:]] == [
        ("exact", 0.0, ids[1]),
        ("judge", 0.25, ids[1]),
    ]
    scores.clear()
    sh_langfuse.log_eval(REPORT, client=client)
    assert [(s["name"], s["value"], s.get("comment")) for s in scores] == [
        ("correctness", 1.0, "ok"),  # None (not scored) is skipped
        ("correctness", 0.5, None),
    ]


# ---- MLflow ----


@pytest.fixture
def mlruns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # a server or SQL store needs more deps
    return f"file://{tmp_path}/mlruns"


def test_mlflow_log_job_resumes_and_finishes(mlruns: str) -> None:
    from mlflow import MlflowClient

    job = FakeJob()
    first = sh_mlflow.log_job(job, tracking_uri=mlruns)  # type: ignore[arg-type]
    # Called again (a later poll, another process): same run, only new steps.
    again = sh_mlflow.log_job(job, tracking_uri=mlruns, follow=True, poll_seconds=0)  # type: ignore[arg-type]
    assert first == again
    client = MlflowClient(tracking_uri=mlruns)
    run = client.get_run(first)
    assert run.info.run_name == "support-qwen-v1-ab12"
    assert run.info.status == "FINISHED"
    assert client.get_experiment(run.info.experiment_id).name == "support-qwen-v1"
    assert run.data.params["hp.epochs"] == "2" and "hp.beta" not in run.data.params
    assert run.data.tags["skyhammer.output_model_arn"] == "arn:aws:bedrock:model/x"
    steps = [m.step for m in client.get_metric_history(first, "loss")]
    assert steps == list(range(1, 10))  # each step once across both calls
    assert [m.step for m in client.get_metric_history(first, "eval_loss")] == [6]


def test_mlflow_failed_job_marks_the_run(mlruns: str) -> None:
    from mlflow import MlflowClient

    with pytest.raises(JobFailedError):
        sh_mlflow.log_job(FakeJob(fail=True), tracking_uri=mlruns, follow=True, poll_seconds=0)  # type: ignore[arg-type]
    (run,) = MlflowClient(tracking_uri=mlruns).search_runs(
        [MlflowClient(tracking_uri=mlruns).get_experiment_by_name("support-qwen-v1").experiment_id]
    )
    assert run.info.status == "FAILED"


def test_mlflow_skips_sagemaker_jobs_logged_natively(settings: Any) -> None:
    job = FakeJob()
    job.record.backend = "sagemaker"  # type: ignore[misc]
    job._backend = SimpleNamespace(  # type: ignore[attr-defined]
        session=SimpleNamespace(settings=settings.model_copy(update={"mlflow_arn": "arn:x"}))
    )
    assert sh_mlflow.log_job(job) is None  # type: ignore[arg-type]


def test_mlflow_log_eval(mlruns: str) -> None:
    from mlflow import MlflowClient

    run_id = sh_mlflow.log_eval(RESULT, RECORDS, tracking_uri=mlruns)
    client = MlflowClient(tracking_uri=mlruns)
    run = client.get_run(run_id)
    assert run.data.metrics == {"eval/exact": 0.5, "eval/judge": 0.5}
    assert run.info.run_name == "my-model"
    path = client.download_artifacts(run_id, "eval_samples.json")
    table = json.loads(Path(path).read_text())
    assert table["columns"][4:6] == ["exact", "judge"]
    assert table["data"][1][1:3] == ["Paris", "Lyon"]
    # Into an existing run (e.g. the training job's), under another key.
    sh_mlflow.log_eval(REPORT, run_id=run_id, tracking_uri=mlruns, key="judge")
    assert client.get_run(run_id).data.metrics["judge/correctness"] == 0.75
