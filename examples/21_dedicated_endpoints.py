"""Dedicated SageMaker endpoints that feel serverless: scale to zero, autoscale, warm up,
and move to Bedrock later without retraining.

Use this for models Bedrock can't serve on demand (e.g. Gemma 4), or base models Bedrock
doesn't offer serverless. Cost: GPU instance-hours while running (e.g. ml.g6e.2xlarge
$2.80/h); scales to zero when idle. Measured live: deploy ~8 min; scale to zero after
~20 min idle; the first request after that waits ~10 min for an instance (the SDK retries),
so use min_copies=1 or warm() before interactive traffic.
Run: uv run python examples/21_dedicated_endpoints.py
"""

from skyhammer import LLM
from skyhammer.serving import Scaling

# A base model on a dedicated endpoint, straight from AWS's prepacked artifacts.
llm = LLM("qwen3-8b", deployment_type="dedicated").deploy(
    name="qwen3-8b-dedicated",
    scaling=Scaling(min_copies=0, max_copies=2, target_invocations_per_copy=5),  # defaults: 0..1
)
print(
    llm.chat.completions.create(messages=[{"role": "user", "content": "Hello!"}], max_tokens=40)
    .choices[0]
    .message.content
)

llm.warm()  # pre-scale before a burst so the first request doesn't wait for an instance

# A fine-tune served on SageMaker can move to Bedrock serverless once supported:
#   ft = LLM("qwen3-0.6b").create_sft_job(ds, output_model="m", serving="sagemaker")...
#   ft.deploy(serving="bedrock")   # imports the stored weights; the endpoint is removed

llm.delete()  # deletes the endpoint, inference component, scaling policies and alarm
