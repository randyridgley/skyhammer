"""AWQ quantization: a managed SageMaker job writes 4-bit weights, then serve them small.

``create_optimization_job`` runs a SageMaker AI inference optimization job that
AWQ-quantizes the weights once (activation-aware, 4-bit). The output is registered as a new
model, about a quarter of the bf16 size, so it fits an instance with less GPU memory. vLLM
reads AWQ weights on its own, so it deploys like any other SageMaker model. Works on a base
model (AWS's prepacked weights, as here) or a SageMaker-trained fine-tune
(``LLM.get("my-model").create_optimization_job(...)``). Runtime fp8 without a job:
``deploy(quantize="fp8")`` (examples/44_quantized_endpoint.py).

Needs the extra IAM from `skyhammer auth policy --optimization`.

Cost: the optimization job bills for the instance SageMaker picks while it runs (capped
here at 1 hour); then one ml.g6.2xlarge endpoint (1x L4, about $1/h) for ~20 min, deleted
at the end. Run: uv run python examples/45_awq_quantization.py
"""

from skyhammer import LLM

job = LLM("qwen3-8b").create_optimization_job(
    "qwen3-8b-awq",
    instance_type="ml.g6.2xlarge",  # the instance to optimize (and later serve) for
    max_hours=1,
)
job.wait(on_status=lambda st: print(" ", st.state, st.message or ""))

small = LLM.get("qwen3-8b-awq").deploy(instance_type="ml.g6.2xlarge")
r = small.chat.completions.create(
    messages=[{"role": "user", "content": "In one sentence: what does AWQ quantization do?"}],
    max_tokens=80,
)
print(r.choices[0].message.content)
small.delete()  # removes the endpoint (stops billing)
