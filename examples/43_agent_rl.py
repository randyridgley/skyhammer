"""Multi-turn (agentic) RL: train GPT-OSS 20B on whole episodes of your agent's work.

The job sends each task to your agent, which plays it out with the model being trained
(several turns of tool calls) and reports a reward; the model learns from how rewards
compare across rollouts of the same task. The agent here is a number-guessing game in
agent_rl/main.py. Deploy it to Amazon Bedrock AgentCore first as a runtime named
skyhammer_agent_guess (its role needs the AWS managed policy
AmazonSageMakerJobRuntimeAccess) and set AGENT_RUNTIME_ARN.

Also needed: a SageMaker MLflow App (settings ``mlflow_arn``; the service logs rewards
and trajectories there), `skyhammer bootstrap` (adds the job role) and the extra IAM
from `skyhammer auth policy --agent-rl`.

Cost: billed per token (prefill, sample, train), so it depends on how long episodes run.
Here 5 steps x 32 tasks x 4 rollouts = 640 short games, roughly $5, plus AgentCore
runtime time and the MLflow App. max_steps bounds every run; the finished job reports
the tokens it billed. Run: AGENT_RUNTIME_ARN=... uv run python examples/43_agent_rl.py
"""

import os
import random

from skyhammer import LLM, Dataset

rng = random.Random(0)
tasks = Dataset.from_records(
    [{"secret": rng.randint(1, 100), "id": i} for i in range(64)], name="guess-number"
)

job = LLM("gpt-oss-20b").create_agent_rl_job(
    tasks,
    agent=os.environ["AGENT_RUNTIME_ARN"],
    output_model="guesser-gptoss",
    max_steps=5,  # bounds the token bill
    batch_size=32,  # tasks per step
    group_size=4,  # rollouts per task, compared against each other
)
print("training", job.name, "- rewards and trajectories are in MLflow")
job.wait(on_status=lambda st: print(" ", st.state.value, st.message or ""))
print(job.status().message)  # the step count and billed tokens
