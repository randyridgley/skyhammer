"""Resume a multi-turn RL run from a checkpoint, or train its output further.

Multi-turn RL registers the full training state (adapter weights, optimizer, step) as a
model package every step. ``job.checkpoints()`` lists them; ``resumable`` ones let a new
job pick up where a failed or stopped run left off. AWS resumes only with the same base
model, tasks and hyperparameters, so pass the same arguments as the first run; Skyhammer
compares them with that run's record and refuses before starting if anything differs.

A finished output can also be trained further with new tasks, rewards or settings:
``create_agent_rl_job`` on ``LLM.get(output_model)`` starts a new run from its weights
with a fresh optimizer.

This follows examples/43_agent_rl.py (same tasks, agent and settings). Cost: billed per
token for the remaining steps only, at most what the rest of that run would have cost
(roughly $5 for all 5 steps).
Run: AGENT_RUNTIME_ARN=... uv run python examples/46_resume_agent_rl.py
"""

import os
import random

from skyhammer import LLM, Dataset, FineTuningJob

rng = random.Random(0)
tasks = Dataset.from_records(
    [{"secret": rng.randint(1, 100), "id": i} for i in range(64)], name="guess-number"
)
settings = {
    "agent": os.environ["AGENT_RUNTIME_ARN"],
    "output_model": "guesser-gptoss",
    "max_steps": 5,
    "batch_size": 32,
    "group_size": 4,
}

job = FineTuningJob.get("guesser-gptoss")
for c in job.checkpoints():
    print(f"step {c.step}: {'resumable' if c.resumable else 'weights'} {c.uri}")

if job.status().state.value in ("failed", "stopped"):
    latest = [c for c in job.checkpoints() if c.resumable][-1]
    job = LLM("gpt-oss-20b").create_agent_rl_job(tasks, resume_from=latest, **settings)
    print("resuming from step", latest.step, "as", job.name)
else:
    # Finished: a second round on the same tasks with more rollouts per task.
    job = LLM.get("guesser-gptoss").create_agent_rl_job(
        tasks, **{**settings, "output_model": "guesser-gptoss-r2", "group_size": 8}
    )
    print("training further as", job.name)
job.wait(on_status=lambda st: print(" ", st.state.value, st.message or ""))
