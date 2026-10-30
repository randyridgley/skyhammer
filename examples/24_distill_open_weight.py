"""Distill any Bedrock model (here Claude Haiku) into a small open-weight student (Qwen3 0.6B).

You provide prompts only. Skyhammer gets the teacher's answers (live calls; a 50%-off batch
job from 1,000 prompts, which can queue for an hour+), caches them in S3, then fine-tunes.
Re-running reattaches instead of paying for the answers again.
Cost: teacher answers under $0.10 + SageMaker training ~$1-2; ~1-2h.
Run: uv run python examples/24_distill_open_weight.py
"""

from skyhammer import LLM, Dataset

topics = ["password reset", "billing", "refunds", "2FA", "data export"]
prompts = Dataset.from_records(
    [
        {
            "messages": [
                {"role": "system", "content": "You are a concise support agent."},
                {"role": "user", "content": f"Customer #{i}: how do I handle {t}?"},
            ]
        }
        for i in range(20)
        for t in topics
    ],
    name="support-prompts",
)

job = LLM("qwen3-0.6b").create_distillation_job(
    prompts,
    teacher="claude-haiku-4-5",  # any serverless model, or a fine-tuned LLM
    output_model="support-qwen-distilled",
    teacher_max_tokens=300,
    max_cost_usd=5,  # covers the teacher's worst case and the training estimate
)
job.wait(on_status=lambda st: print(" ", st.state.value))
student = job.output_llm.deploy()  # imported into Bedrock, pay per token
reply = student.chat.completions.create(
    messages=[{"role": "user", "content": "How do I enable 2FA?"}], max_tokens=200
)
print(reply.choices[0].message.content)
student.delete()
