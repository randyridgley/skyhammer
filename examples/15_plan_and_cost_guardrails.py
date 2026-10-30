"""Plan before you spend: route, live price estimate, cost cap, RL time cap, tags, KMS, residency.

Cost: free (planning only). Run: uv run python examples/15_plan_and_cost_guardrails.py
Your data never leaves your account: datasets, weights and logs live in your S3 bucket,
encrypted with your KMS key (created by `skyhammer bootstrap`). Every API call Skyhammer
makes is in your CloudTrail audit log (see the end of this script).
"""

from pathlib import Path

import skyhammer
from skyhammer import LLM, CostLimitExceededError, Dataset

skyhammer.configure(
    tags={"cost-center": "1234", "project": "support-bot"},  # on every AWS resource
    # kms_key_arn="arn:aws:kms:...",       # bring your own key instead of the bootstrap key
    # allowed_regions=["us-east-1"],       # data residency
    # vpc={"us-east-1": {"subnet_ids": ["subnet-a"], "security_group_ids": ["sg-1"]}},
    #                                      # train, import and serve inside your VPC
)
ds = Dataset.from_file(Path(__file__).parent / "data" / "train.jsonl")

for model in ["nova-micro", "llama-3.1-8b", "gemma-4-31b"]:
    request, estimate = LLM(model).plan(
        "sft", ds, output_model=f"support-{model.replace('.', '-')}", epochs=3, accept_eula=True
    )
    print(f"{model:14} {request.route.describe():60} {estimate}")

try:  # a hard cap: the job is refused before anything is submitted
    LLM("nova-pro").create_sft_job(ds, output_model="support-pro", epochs=50, max_cost_usd=0.01)
except CostLimitExceededError as e:
    print("refused:", e)

# RL is billed by the hour, so RL jobs always have a runtime cap (4h unless you set
# max_hours); the estimate is the most it can cost, and max_cost_usd applies to that.
reward = skyhammer.Grader.from_lambda("arn:aws:lambda:us-east-1:111122223333:function:my-reward")
rl_prompts = Dataset.from_records(
    [
        {"messages": [{"role": "user", "content": f"What is {i} + {i}?"}], "answer": str(2 * i)}
        for i in range(40)
    ],
    name="math",
)
_, rl_estimate = LLM("qwen3-8b").plan(
    "rft", rl_prompts, output_model="math-rl", grader=reward, max_hours=2
)
print("RL:", rl_estimate)

# Audit: every Skyhammer action is an ordinary AWS API call in CloudTrail.
from skyhammer.session import get_session  # noqa: E402

trail = get_session().client("cloudtrail")
events = trail.lookup_events(
    LookupAttributes=[{"AttributeKey": "EventSource", "AttributeValue": "bedrock.amazonaws.com"}],
    MaxResults=5,
)
for e in events["Events"]:
    print("audit:", e["EventTime"], e["EventName"], e.get("Username"))
