"""Fine-tune Amazon Nova on Bedrock, deploy it serverless, chat, clean up.

Cost: well under $0.01 of training + on-demand inference; ~45 min (mostly Bedrock job
startup). Run: uv run python examples/16_fine_tune_nova.py
"""

from pathlib import Path

from skyhammer import LLM, Dataset

ds = Dataset.from_file(Path(__file__).parent / "data" / "train.jsonl", name="support")

# Re-running this script reattaches to the same job instead of starting a new one.
job = LLM("nova-micro").create_sft_job(
    ds, output_model="support-nova", epochs=2, validation_split=0.1, max_cost_usd=1.00
)
print("training", job.name, "- follow it with: skyhammer jobs logs support-nova")
job.wait(on_status=lambda st: print(" ", st.state.value))

ft = job.output_llm.deploy()  # on-demand: pay per token, nothing to scale or babysit
r = ft.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful support agent."},
        {"role": "user", "content": "How do I reset my password?"},
    ]
)
print(r.choices[0].message.content)

ft.delete()  # removes the deployment and the custom model (stops storage charges)
