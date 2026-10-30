"""Fine-tune an open-weight model (Qwen, Llama, GPT-OSS, DeepSeek...) and serve it on Bedrock.

Trains on SageMaker AI serverless, then imports the merged weights into Bedrock
automatically. Cost: ~$0.001 of training + Bedrock import storage; ~30 min.
Run: uv run python examples/17_fine_tune_open_weight.py
"""

from pathlib import Path

from skyhammer import LLM, Dataset, UnsupportedParameterError

ds = Dataset.from_file(Path(__file__).parent / "data" / "train.jsonl", name="support")
base = LLM("qwen3-0.6b")  # or "llama-3.1-8b" (gated: pass accept_eula=True), "gpt-oss-20b"...

# Hyperparameters are checked against AWS's own recipe limits before anything runs.
try:
    base.plan("sft", ds, output_model="x", learning_rate=0.01)
except UnsupportedParameterError as e:
    print("caught before spending:", e)

job = base.create_sft_job(
    ds,
    output_model="support-qwen",
    epochs=1,
    lora_rank=16,  # LoRA by default; training_type="full" for full fine-tuning, which
    # defaults to learning_rate=1e-5 (1e-4 suits LoRA but makes full FT memorize and ramble)
    learning_rate=1e-4,
    # hyperparameters_extra={"lora_dropout": 0.1},  # any recipe key, validated too
)
job.wait(on_status=lambda st: print(" ", st.state.value, st.message or ""))  # train -> import

ft = job.output_llm  # imported models are served on demand; no deploy step needed
r = ft.chat.completions.create(
    messages=[{"role": "user", "content": "How do I reset my password?"}], max_tokens=80
)  # first call may wait for a cold start
print(r.choices[0].message.content)

ft.delete()
