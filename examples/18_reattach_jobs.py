"""Jobs outlive your process: reattach by name, follow logs and checkpoints, chat later,
and keep training a fine-tuned model on new data.

Continued fine-tuning starts from the model you already trained, on the backend that made
it: Bedrock trains the earlier custom model further, SageMaker starts from the model
package the earlier job registered. It's a new job and a new model; the old one stays.

Cost: free (reads job state) until the commented continued-training lines, which bill like
any fine-tuning job. Run after 16/17: uv run python examples/18_reattach_jobs.py
CLI: skyhammer jobs list | jobs get NAME | jobs wait NAME | jobs logs NAME
"""

from skyhammer import LLM, FineTuningJob

for job in FineTuningJob.list():  # latest job for each of your fine-tuned models
    print(f"{job.record.output_model:24} {job.status().state.value:10} {job.record.backend}")

job = FineTuningJob.get("support-nova")  # by model name (latest job) or exact job name
print(job.status())
for line in job.logs(limit=10):  # Bedrock step metrics or SageMaker training progress
    print(" ", line)

# Intermediate checkpoints with their metrics. bedrock-mantle checkpoints are models you can
# call (checkpoint.llm()); SageMaker's are training state in S3; Bedrock keeps none.
for ckpt in job.checkpoints():
    print(f"  step {ckpt.step}: {ckpt.metrics} {ckpt.model_id or ckpt.uri}")

ft = LLM.get("support-nova")  # reuses the recorded deployment; no redeploy
print(
    ft.chat.completions.create(messages=[{"role": "user", "content": "Hi!"}], max_tokens=30)
    .choices[0]
    .message.content
)

# Continue training it on new data (a new job and model; the original is untouched):
#   more = Dataset.from_file("data/train_v2.jsonl")
#   v2 = LLM.get("support-nova").create_sft_job(more, output_model="support-nova-v2")
#   v2.wait()
