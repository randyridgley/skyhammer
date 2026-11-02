"""Weights & Biases: training curves and eval results as W&B runs.

- ``log_job(job)`` logs a fine-tuning job's configuration (base model, technique,
  hyperparameters, dataset) and its per-step metrics. With ``follow=True`` it waits for
  the job and streams new metrics as they appear, so the W&B chart is live.
- ``log_eval(result, records)`` logs mean scores to the run summary and every sample
  (prompt, reference, output, per-grader scores) as a ``wandb.Table``. Takes a local
  ``evaluate`` result or a managed eval job's report (judge explanations included).

Both take ``run=`` to log into a run you already have; otherwise they start one (in
``project``, default "skyhammer") and finish it. Needs ``pip install "skyhammer[wandb]"``
and ``wandb login`` (or ``WANDB_API_KEY``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import wandb

from skyhammer.integrations._eval import eval_rows

if TYPE_CHECKING:
    from skyhammer.evaluation import EvalReport, EvalResult
    from skyhammer.jobs import FineTuningJob, JobStatus

PROJECT = "skyhammer"


def job_config(job: FineTuningJob) -> dict[str, Any]:
    r = job.record
    return {
        "base_model": r.base_model,
        "backend_model_id": r.backend_model_id,
        "technique": r.technique,
        "training_type": r.training_type,
        "backend": r.backend,
        "region": r.region,
        "output_model": r.output_model,
        "dataset": r.dataset,
        "hyperparameters": {k: v for k, v in r.hyperparameters.items() if v is not None},
    }


def _start(run: Any, project: str, **init: Any) -> tuple[Any, bool]:
    if run is not None:
        return run, False
    return wandb.init(project=project, **init), True


def log_job(
    job: FineTuningJob,
    *,
    run: Any = None,
    project: str = PROJECT,
    follow: bool = False,
    poll_seconds: float = 60,
    **init: Any,
) -> Any:
    """Log a job's config and metrics to W&B; returns the run. ``follow=True`` waits for
    the job (``job.wait``) and logs new steps on every poll. ``init`` goes to ``wandb.init``."""
    r = job.record
    run, owned = _start(
        run,
        project,
        **{
            "name": r.output_model,
            "job_type": r.technique,
            "tags": [r.technique, r.base_model],
            **init,
        },
    )
    run.config.update(job_config(job), allow_val_change=True)
    run.summary["skyhammer/job"] = r.job_name
    if r.estimate:
        run.summary["skyhammer/estimate"] = r.estimate
    last = -1

    def flush(final: bool) -> None:
        # The newest step can still gain keys (evaluation lands after the training loss),
        # and W&B steps only go forward, so it waits for the next poll or the end.
        nonlocal last
        rows = job.metrics()
        newest = max((int(r.get("step", 0)) for r in rows), default=-1)
        for row in rows:
            step = int(row.get("step", 0))
            if step <= last or (step == newest and not final):
                continue
            run.log({k: v for k, v in row.items() if k != "step"}, step=step)
            last = step

    def on_status(st: JobStatus) -> None:
        run.summary["skyhammer/state"] = st.state.value
        if st.output_model_arn:
            run.summary["skyhammer/output_model_arn"] = st.output_model_arn
        flush(final=st.state.terminal or not follow)

    failed = False
    try:
        if follow:
            job.wait(on_status=on_status, poll_seconds=poll_seconds)
        else:
            on_status(job.status())
    except BaseException:
        failed = True
        raise
    finally:
        if owned:
            run.finish(exit_code=1 if failed else 0)
    return run


def log_eval(
    result: EvalResult | EvalReport,
    records: list[dict[str, Any]] | None = None,
    *,
    run: Any = None,
    project: str = PROJECT,
    key: str = "eval",
    **init: Any,
) -> Any:
    """Log mean scores (``<key>/<grader>`` in the summary) and a per-sample table
    (``<key>/samples``); returns the run. Pass ``records`` with a local result to include
    each prompt and reference answer."""
    model = getattr(result, "model", None)
    rows = eval_rows(result, records)
    metrics = sorted({k for row in rows for k in row.scores})
    run, owned = _start(
        run, project, **{"job_type": "eval", **({"name": model} if model else {}), **init}
    )
    try:
        for name, mean in result.means.items():
            run.summary[f"{key}/{name}"] = mean
        columns: list[str | int] = ["prompt", "expected", "output", "category"]
        columns += [*metrics, "comments"]
        table = wandb.Table(columns=columns)
        for row in rows:
            prompt = (
                row.prompt
                if isinstance(row.prompt, str) or row.prompt is None
                else (json.dumps(row.prompt, ensure_ascii=False))
            )
            table.add_data(
                prompt,
                row.expected,
                row.output,
                row.category,
                *(row.scores.get(m) for m in metrics),
                json.dumps(row.comments, ensure_ascii=False) if row.comments else None,
            )
        run.log({f"{key}/samples": table})
    finally:
        if owned:
            run.finish()
    return run
