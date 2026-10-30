"""Dedicated SageMaker endpoints that feel serverless: scale to zero, autoscale, warm up,
and move to Bedrock later without retraining.

Use this for models Bedrock can't serve on demand (e.g. Gemma 4), or base models Bedrock
doesn't offer serverless. Cost: GPU instance-hours while running (e.g. ml.g6e.2xlarge
$2.80/h); scales to zero when idle. Measured live: deploy ~8 min; scale to zero after
~20 min idle; the first request after that waits ~10 min for an instance (the SDK retries),
so use min_copies=1 or warm() before interactive traffic.

GPUs run out: when the preferred instance has no capacity, SageMaker falls back to other
GPU types that fit the model (up to 5, in order), at creation and on every scale-out.
Leave instance_type unset for automatic fallbacks, pass a list for your own order, or one
type to pin it. Measured live 2026-10-30: ml.g6e.2xlarge had no capacity; the endpoint
came up on ml.g5.xlarge in 2 minutes instead of failing.
Run: uv run python examples/21_dedicated_endpoints.py
"""

from skyhammer import LLM
from skyhammer.serving import Scaling

# A base model on a dedicated endpoint, straight from AWS's prepacked artifacts.
llm = LLM("qwen3-8b", deployment_type="dedicated").deploy(
    name="qwen3-8b-dedicated",
    scaling=Scaling(min_copies=0, max_copies=2, target_invocations_per_copy=5),  # defaults: 0..1
    # instance_type=["ml.g6e.2xlarge", "ml.g5.2xlarge"],  # optional: your own fallback order
)
print(
    llm.chat.completions.create(messages=[{"role": "user", "content": "Hello!"}], max_tokens=40)
    .choices[0]
    .message.content
)

llm.warm()  # pre-scale before a burst so the first request doesn't wait for an instance

# Many LoRA fine-tunes on one GPU (multi-LoRA): start a shared base endpoint, then add each
# adapter to it. Adapters come from LoRA jobs that keep the adapter separate:
#   shared = LLM("qwen3-8b", deployment_type="dedicated").deploy(name="qwen-shared", adapters=True)
#   LLM("qwen3-8b").create_sft_job(ds, output_model="support-v1", serving="sagemaker",
#                                  hyperparameters_extra={"merge_weights": False}).wait()
#   v1 = LLM.get("support-v1").deploy(on="qwen-shared")  # no new GPUs; chat as usual
#   v1.deployment.delete()  # removes just the adapter; deleting the endpoint removes all

# A fine-tune served on SageMaker can move to Bedrock serverless once supported:
#   ft = LLM("qwen3-0.6b").create_sft_job(ds, output_model="m", serving="sagemaker")...
#   ft.deploy(serving="bedrock")   # imports the stored weights; the endpoint is removed

llm.delete()  # deletes the endpoint, inference component, scaling policies and alarm
