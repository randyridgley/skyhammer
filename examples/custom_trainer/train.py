"""A custom training loop for examples/40_custom_training.py: plain PyTorch, LoRA, loss on
the answer tokens only. Runs inside the SageMaker job; Skyhammer passes the base weights
(SM_CHANNEL_MODEL), the rows (SM_CHANNEL_TRAIN) and hyperparameters (--epochs ...), and
imports whatever lands in SM_MODEL_DIR into Bedrock.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--learning_rate", type=float, default=1e-4)
p.add_argument("--lora_rank", type=int, default=16)
p.add_argument("--max_length", type=int, default=512)
args, _ = p.parse_known_args()  # Skyhammer also passes training_type

base, out = os.environ["SM_CHANNEL_MODEL"], os.environ["SM_MODEL_DIR"]
files = glob.glob(os.path.join(os.environ["SM_CHANNEL_TRAIN"], "**/*.jsonl"), recursive=True)
rows = [json.loads(line) for path in files for line in Path(path).read_text().splitlines() if line]

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).cuda()
model = get_peft_model(
    model,
    LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    ),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)


def example(row):
    """Chat-formatted prompt + answer; labels mask the prompt so only the answer trains."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True
    )
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(row["answer"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[: args.max_length]
    labels = ([-100] * len(p_ids) + a_ids)[: args.max_length]
    return torch.tensor([ids]).cuda(), torch.tensor([labels]).cuda()


step = 0
with open(os.path.join(out, "train_metrics.jsonl"), "w") as metrics:
    model.train()
    for epoch in range(args.epochs):
        for row in rows:
            ids, labels = example(row)
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            metrics.write(
                json.dumps(
                    {"train/global_step": step, "train/loss": loss.item(), "train/epoch": epoch}
                )
                + "\n"
            )

model = model.merge_and_unload()  # Bedrock imports plain weights, not adapters
model.save_pretrained(out, safe_serialization=True)
tokenizer.save_pretrained(out)
