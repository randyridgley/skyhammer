"""Distill a large teacher (Nova Pro) into a small, cheap student (Nova Micro).

You provide prompts only; Bedrock generates teacher answers and trains the student.
Cost: teacher inference + training, roughly $0.10-0.50. Measured live: ~1h40m
(Bedrock first generates the teacher's answers, then trains).
Run: uv run python examples/20_distillation.py
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
        for i in range(30)
        for t in topics
    ],
    name="support-prompts",
)

job = LLM("nova-micro").create_distillation_job(
    prompts, teacher="amazon.nova-pro-v1:0", output_model="support-distilled"
)
job.wait(on_status=lambda st: print(" ", st.state.value))
student = job.output_llm.deploy()
print(
    student.chat.completions.create(messages=[{"role": "user", "content": "How do I enable 2FA?"}])
    .choices[0]
    .message.content
)
student.delete()
