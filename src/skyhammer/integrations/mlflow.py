"""MLflow: training curves and eval results for every backend, not only SageMaker.

SageMaker training jobs log to MLflow natively when settings ``mlflow_arn`` is set
(experiment = output model name, run = job name). This module gives Bedrock jobs and
evaluations the same layout, so all runs sit side by side:

- ``log_job(job)`` logs config as params and per-step metrics into the run named after
  the job (created or resumed, so calling it again only adds new steps); ``follow=True``
  waits for the job and logs as it trains.
- ``log_eval(result, records)`` logs mean scores as metrics and every sample as an MLflow
  table (``eval_samples.json``). Takes a local ``evaluate`` result or a managed eval report.

The tracking server is ``tracking_uri``, else settings ``mlflow_arn`` (a SageMaker MLflow
App, through the ``sagemaker-mlflow`` plugin), else MLflow's own default
(``MLFLOW_TRACKING_URI``). Needs ``pip install "skyhammer[mlflow]"``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunStatus

from skyhammer.integrations._eval import eval_rows

if TYPE_CHECKING:
    from skyhammer.evaluation import EvalReport, EvalResult
    from skyhammer.jobs import FineTuningJob, JobStatus
    from skyhammer.session import Session

log = logging.getLogger("skyhammer")

EVAL_EXPERIMENT = "skyhammer-evals"
_LAST_STEP = "skyhammer.last_step"
_MAX_PARAM = 6000  # MLflow's param value limit


def client(tracking_uri: str | None = None, session: Session | None = None) -> MlflowClient:
    if tracking_uri is None and session is not None:
        tracking_uri = session.settings.mlflow_arn
    return MlflowClient(tracking_uri=tracking_uri)


def _experiment(mlflow: MlflowClient, name: str) -> str:
    found = mlflow.get_experiment_by_name(name)
    return str(found.experiment_id) if found else str(mlflow.create_experiment(name))


def _run(mlflow: MlflowClient, experiment_id: str, name: str, tags: dict[str, str]) -> Any:
    existing = mlflow.search_runs(
        [experiment_id], filter_string=f"attributes.run_name = '{name}'", max_results=1
    )
    if existing:
        return existing[0]
    return mlflow.create_run(experiment_id, run_name=name, tags=tags)


def job_params(job: FineTuningJob) -> dict[str, str]:
    r = job.record
    params: dict[str, Any] = {
        "base_model": r.base_model,
        "backend_model_id": r.backend_model_id,
        "technique": r.technique,
        "training_type": r.training_type,
        "backend": r.backend,
        "region": r.region,
        "dataset": r.dataset.get("name"),
        "dataset_sha256": r.dataset.get("sha256"),
        "dataset_rows": r.dataset.get("rows"),
        **{f"hp.{k}": v for k, v in r.hyperparameters.items() if v is not None},
    }
    return {
        k: (v if isinstance(v, str) else json.dumps(v))[:_MAX_PARAM]
        for k, v in params.items()
        if v is not None
    }


def log_job(
    job: FineTuningJob,
    *,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    follow: bool = False,
    poll_seconds: float = 60,
) -> str | None:
    """Log a job's params and per-step metrics; returns the MLflow run ID. Experiment:
    the output model name; run: the job name (as SageMaker's native logging does).
    Returns None for SageMaker jobs already logging natively to settings ``mlflow_arn``."""
    r = job.record
    session = getattr(getattr(job, "_backend", None), "session", None)
    if (
        r.backend == "sagemaker"
        and session is not None
        and session.settings.mlflow_arn
        and (tracking_uri in (None, session.settings.mlflow_arn))
    ):
        log.info("%s logs to MLflow natively (settings mlflow_arn); nothing to add", r.job_name)
        return None
    mlflow = client(tracking_uri, session)
    exp = _experiment(mlflow, experiment or r.output_model)
    run = _run(
        mlflow,
        exp,
        r.job_name,
        {"skyhammer.technique": r.technique, "skyhammer.base_model": r.base_model},
    )
    run_id = str(run.info.run_id)
    logged = {p for p in (run.data.params or {})}
    params = [Param(k, v) for k, v in job_params(job).items() if k not in logged]
    if params:
        mlflow.log_batch(run_id, params=params)
    last = int((run.data.tags or {}).get(_LAST_STEP, -1))

    def flush(final: bool) -> None:
        # The newest step can still gain keys (evaluation lands after the training
        # loss), so it waits for the next poll or the end.
        nonlocal last
        rows = job.metrics()
        newest = max((int(row.get("step", 0)) for row in rows), default=-1)
        now = int(time.time() * 1000)
        metrics = []
        for row in rows:
            step = int(row.get("step", 0))
            if step <= last or (step == newest and not final):
                continue
            metrics += [Metric(k, float(v), now, step) for k, v in row.items() if k != "step"]
            last = step
        for i in range(0, len(metrics), 1000):  # log_batch takes up to 1,000 metrics
            mlflow.log_batch(run_id, metrics=metrics[i : i + 1000])
        mlflow.set_tag(run_id, _LAST_STEP, str(last))

    def on_status(st: JobStatus) -> None:
        mlflow.set_tag(run_id, "skyhammer.state", st.state.value)
        if st.output_model_arn:
            mlflow.set_tag(run_id, "skyhammer.output_model_arn", st.output_model_arn)
        flush(final=st.state.terminal or not follow)
        if st.state.terminal:
            ok = st.state.value == "completed"
            mlflow.set_terminated(
                run_id, RunStatus.to_string(RunStatus.FINISHED if ok else RunStatus.FAILED)
            )

    if follow:
        job.wait(on_status=on_status, poll_seconds=poll_seconds)
    else:
        on_status(job.status())
    return run_id


def log_eval(
    result: EvalResult | EvalReport,
    records: list[dict[str, Any]] | None = None,
    *,
    run_id: str | None = None,
    tracking_uri: str | None = None,
    experiment: str = EVAL_EXPERIMENT,
    run_name: str | None = None,
    key: str = "eval",
    session: Session | None = None,
) -> str:
    """Log mean scores (``<key>/<grader>``) and a per-sample table; returns the run ID.
    Pass ``run_id`` to add them to an existing run (for example a training job's)."""
    mlflow = client(tracking_uri, session)
    rows = eval_rows(result, records)
    model = getattr(result, "model", None)
    if run_id is None:
        exp = _experiment(mlflow, experiment)
        tags = {"skyhammer.model": model} if model else {}
        run_id = str(mlflow.create_run(exp, run_name=run_name or model, tags=tags).info.run_id)
        owned = True
    else:
        owned = False
    now = int(time.time() * 1000)
    mlflow.log_batch(
        run_id, metrics=[Metric(f"{key}/{k}", v, now, 0) for k, v in result.means.items()]
    )
    metrics = sorted({k for row in rows for k in row.scores})
    columns = ["prompt", "expected", "output", "category", *metrics, "comments"]
    data = [
        [
            row.prompt
            if isinstance(row.prompt, str) or row.prompt is None
            else json.dumps(row.prompt, ensure_ascii=False),
            row.expected,
            row.output,
            row.category,
            *(row.scores.get(m) for m in metrics),
            json.dumps(row.comments, ensure_ascii=False) if row.comments else None,
        ]
        for row in rows
    ]
    # MLflow's table format (what mlflow.log_table writes), without needing pandas.
    mlflow.log_dict(run_id, {"columns": columns, "data": data}, f"{key}_samples.json")
    if owned:
        mlflow.set_terminated(run_id)
    return run_id
