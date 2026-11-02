"""Quantization and speculative decoding on a SageMaker endpoint.

``quantize="fp8"`` quantizes the weights as the LMI (vLLM) container loads them: half the
GPU memory of bf16, with little quality loss on most tasks (``"bitsandbytes"`` is 4-bit).
``speculative=`` drafts several tokens per step and verifies them in one pass, which
speeds up decoding when drafts are often right. The ``ngram`` method drafts from text
already in the prompt (good for summaries, edits and RAG answers) and needs no extra
model; ``{"method": "eagle3", "model": "<draft>", ...}`` uses a trained draft head.

Both are SageMaker endpoint options; Bedrock serving doesn't expose them. AWS's
recommended instance stays the default, so pass a smaller ``instance_type`` to turn the
memory fp8 frees into a cheaper endpoint. Works the same on fine-tunes served from
SageMaker: ``LLM.get("my-model").deploy(quantize="fp8")``.

Cost: one GPU endpoint for ~20 min (Qwen3 8B on AWS's recommended instance, about $1/h),
deleted at the end. Run: uv run python examples/44_quantized_endpoint.py
"""

from skyhammer import LLM

llm = LLM("qwen3-8b").deploy(
    "qwen3-8b-fp8",
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
