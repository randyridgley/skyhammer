"""Experiment tracking: traces, eval scores and training curves in Langfuse, W&B and MLflow.

Langfuse: `traced(model)` records every `chat.completions.create` as a generation with
messages, tokens, Skyhammer's cost and latency (streams too); `log_eval` records each eval
sample with a score per grader. W&B: `log_eval` logs mean scores and a table of samples;
`log_job` logs a fine-tuning job's config and loss curve (`follow=True` streams it live).
MLflow: the same `log_job` / `log_eval` for Bedrock jobs and evals, in the layout SageMaker
jobs already use natively with settings `mlflow_arn` (experiment = model, run = job).

Needs `pip install "skyhammer[langfuse,wandb,mlflow]"`, Langfuse keys (`LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) and `wandb login`; set `WANDB_MODE=offline` to try
W&B without an account. MLflow uses settings `mlflow_arn` or `MLFLOW_TRACKING_URI`.
Cost: three Nova Micro calls, a fraction of a cent.
Run: uv run python examples/38_experiment_tracking.py
"""

from langfuse import get_client

from skyhammer import LLM, FineTuningJob, Grader
from skyhammer.evaluation import evaluate
from skyhammer.integrations import langfuse, mlflow, wandb

records = [
    {
        "messages": [
            {"role": "user", "content": "Capital of France? One word."},
            {"role": "assistant", "content": "Paris"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "12 * 12? Just the number."},
            {"role": "assistant", "content": "144"},
        ]
    },
]

# Every call through the traced model becomes a Langfuse generation (cost included).
model = langfuse.traced(LLM("nova-micro"), metadata={"team": "support"})
print(model.chat.completions.create(messages=[{"role": "user", "content": "Hi!"}]).choices[0])

# Evaluate through the traced model, then send the scores to both tools.
result = evaluate(model, records, [Grader.builtin("contains")], max_tokens=20)
trace_ids = langfuse.log_eval(result, records, run="nova-micro baseline")
run = wandb.log_eval(result, records, project="skyhammer-examples")
mlflow_run = mlflow.log_eval(result, records, run_name="nova-micro baseline")
print("scores:", result.means, "| langfuse traces:", len(trace_ids), "| wandb run:", run.id)

# A fine-tuning job's config and loss curve, if you have one (follow=True waits and streams).
jobs = FineTuningJob.list()
if jobs:
    wandb.log_job(jobs[0], project="skyhammer-examples")
    mlflow.log_job(jobs[0])  # Bedrock jobs; SageMaker ones already log natively

get_client().flush()  # Langfuse sends in the background; flush before a script exits
