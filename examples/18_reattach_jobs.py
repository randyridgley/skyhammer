"""Jobs outlive your process: reattach by name from anywhere, follow logs, chat later.

Cost: free (reads job state). Run after 16/17: uv run python examples/18_reattach_jobs.py
CLI: skyhammer jobs list | jobs get NAME | jobs wait NAME | jobs logs NAME
"""

from skyhammer import LLM, FineTuningJob

for job in FineTuningJob.list():  # latest job for each of your fine-tuned models
    print(f"{job.record.output_model:24} {job.status().state.value:10} {job.record.backend}")

job = FineTuningJob.get("support-nova")  # by model name (latest job) or exact job name
print(job.status())
for line in job.logs(limit=10):  # Bedrock step metrics or SageMaker training progress
    print(" ", line)

ft = LLM.get("support-nova")  # reuses the recorded deployment; no redeploy
print(
    ft.chat.completions.create(messages=[{"role": "user", "content": "Hi!"}], max_tokens=30)
    .choices[0]
    .message.content
)
