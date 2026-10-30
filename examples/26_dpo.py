"""Preference tuning (DPO): teach a model which of two answers you prefer.

Each record is a prompt plus a preferred and a non-preferred answer (OpenAI's preference
format). Trains on SageMaker AI serverless customization, then imports into Bedrock.
Cost: SageMaker training ~$1-2 for Qwen3 0.6B; ~1h.
Run: uv run python examples/26_dpo.py
"""

from skyhammer import LLM, Dataset

topics = ["password reset", "billing", "refunds", "2FA", "data export"]
prefs = Dataset.from_records(
    [
        {
            "input": {
                "messages": [
                    {"role": "system", "content": "You are a support agent."},
                    {"role": "user", "content": f"Customer #{i}: I need help with {t}."},
                ]
            },
            "preferred_output": [
                {"role": "assistant", "content": f"Happy to help with {t}! Here are the steps..."}
            ],
            "non_preferred_output": [{"role": "assistant", "content": "Read the docs."}],
        }
        for i in range(20)
        for t in topics
    ],
    name="support-prefs",
)

job = LLM("qwen3-0.6b").create_dpo_job(
    prefs,
    output_model="support-qwen-dpo",
    epochs=1,
    beta=0.05,  # beta: preference strength
)
job.wait(on_status=lambda st: print(" ", st.state.value))
tuned = job.output_llm.deploy()
reply = tuned.chat.completions.create(
    messages=[{"role": "user", "content": "I need help with refunds."}], max_tokens=200
)
print(reply.choices[0].message.content)
tuned.delete()
