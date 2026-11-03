"""Quantization and speculative decoding on a SageMaker endpoint.

``quantize="fp8"`` quantizes the weights as the LMI (vLLM) container loads them: half the
GPU memory of bf16, with little quality loss on most tasks (``"bitsandbytes"`` is 4-bit).
``speculative=`` drafts several tokens per step and verifies them in one pass, which
speeds up decoding when drafts are often right. The ``ngram`` method drafts from text
already in the prompt (good for summaries, edits and RAG answers) and needs no extra
model; ``{"method": "eagle3", "model": "<draft>", ...}`` uses a trained draft head.

Both are SageMaker endpoint options; Bedrock serving doesn't expose them. Without
``instance_type`` the endpoint uses AWS's recommended instance, which for Qwen3 8B is an
8-GPU ml.p4d.24xlarge (~$25/h). Here it's pinned to the cheapest type AWS validated the
model on, with a capacity fallback of the same GPU count; Skyhammer applies AWS's settings
for that type (tensor parallelism, memory). Works the same on fine-tunes served from
SageMaker: ``LLM.get("my-model").deploy(quantize="fp8")``.

Cost: ml.g6.12xlarge (4x L4, $5.75/h) for ~25 min, about $2.40, deleted at the end.
Run: uv run python examples/44_quantized_endpoint.py
"""

from skyhammer import LLM

llm = LLM("qwen3-8b").deploy(
    "qwen3-8b-fp8",
    instance_type=["ml.g6.12xlarge", "ml.g5.12xlarge"],  # 4 GPUs each; fallback on capacity
    quantize="fp8",
    speculative={"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4},
)
policy = (
    "Refunds: items can be returned within 30 days with a receipt. Opened software and "
    "gift cards can't be returned. Refunds go back to the original payment method."
)
r = llm.chat.completions.create(
    messages=[
        {"role": "system", "content": f"Answer from this policy only:\n{policy}"},
        {"role": "user", "content": "Can I return an opened video game after two weeks?"},
    ],
    max_tokens=120,
)
print(r.choices[0].message.content)
llm.delete()  # removes the endpoint (stops billing)
