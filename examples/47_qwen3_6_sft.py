"""Fine-tune Qwen 3.6 27B on SageMaker and chat with it.

Qwen 3.6 trains through SageMaker AI serverless customization with AWS's verl recipes
(27B: SFT, DPO, RLVR and RLAIF; 35B-A3B: SFT, RLVR and RLAIF; LoRA or full). Bedrock
can't import the architecture, so the fine-tune serves from a SageMaker endpoint. They are
vision-language models, but Skyhammer trains them on text only for now: AWS hasn't
documented the image row format for these recipes.

The recipe's own defaults apply (LoRA rank 32, alpha 64, batch 16 or more), and small
datasets get a validation split of at least one batch.

Cost: training is billed per token, about $0.02 for these 50 rows. The endpoint is the
LoRA recipe's serving profile, ml.g6e.48xlarge (8x L40S, $37.66/h); it takes 20-30 min
to start, so expect roughly $19-25 in all. It's deleted at the end.
Run: uv run python examples/47_qwen3_6_sft.py
"""

from skyhammer import LLM, Dataset

job = LLM("qwen3.6-27b").create_sft_job(
    Dataset.from_file("examples/data/train.jsonl"),
    output_model="support-qwen36",
    batch_size=16,
)
job.wait(progress=True)

llm = job.output_llm.deploy("support-qwen36")
r = llm.chat.completions.create(
    messages=[{"role": "user", "content": "How do I reset my password?"}], max_tokens=200
)
print(r.choices[0].message.content)
llm.delete()  # removes the endpoint (stops billing)
