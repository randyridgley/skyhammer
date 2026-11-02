"""Continued pre-training entry point, run by Skyhammer's CPT SageMaker training jobs.

Runs inside the Hugging Face PyTorch training container (transformers, peft, datasets).
Reads base weights from the ``model`` channel and ``{"text": ...}`` JSONL from ``train``
(and optionally ``validation``), packs the text into ``max_seq_length`` blocks, trains with
the causal-LM objective, and writes merged Hugging Face weights to /opt/ml/model, which
Bedrock Custom Model Import or a SageMaker endpoint can serve directly.

Per-step metrics go to ``train_metrics.jsonl`` / ``metrics.jsonl`` in the model folder in
the same shape as the SageMaker recipes, so ``job.metrics()`` reads them the same way.
"""

import argparse
import glob
import json
import math
import os
import shutil

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

MODEL_DIR = "/opt/ml/model"
CHECKPOINT_DIR = "/opt/ml/checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--training_type", default="lora", choices=["lora", "full"])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)  # global, per optimizer step
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--gradient_checkpointing", type=lambda s: s.lower() == "true", default=True)
    return p.parse_args()


def channel(name):
    return os.environ.get(f"SM_CHANNEL_{name.upper()}")


def jsonl_files(path):
    return sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))


def packed(path, tokenizer, block):
    """Tokenize every document, join them with EOS, and cut fixed-length blocks."""
    ds = load_dataset("json", data_files=jsonl_files(path), split="train")
    eos = tokenizer.eos_token_id

    def tokenize(batch):
        return {"input_ids": [tokenizer(t)["input_ids"] + [eos] for t in batch["text"]]}

    def group(batch):
        ids = [i for row in batch["input_ids"] for i in row]
        n = (len(ids) // block) * block or len(ids)  # keep a short corpus as one block
        blocks = [ids[i : i + block] for i in range(0, n, block)]
        return {"input_ids": blocks, "attention_mask": [[1] * len(b) for b in blocks]}

    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    return ds.map(group, batched=True, remove_columns=ds.column_names)


class MetricsFile(TrainerCallback):
    """Write losses as JSONL rows keyed ``train/global_step`` / ``eval/global_step``."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        kind = "eval" if any(k.startswith("eval_") for k in logs) else "train"
        name = "metrics.jsonl" if kind == "eval" else "train_metrics.jsonl"
        row = {f"{kind}/global_step": state.global_step}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                row[f"{kind}/{k.removeprefix('eval_')}"] = v
        with open(os.path.join(MODEL_DIR, name), "a") as f:
            f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    base = channel("model")
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if args.training_type == "lora":
        from peft import LoraConfig, get_peft_model

        if args.gradient_checkpointing:
            model.enable_input_require_grads()
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()

    train = packed(channel("train"), tokenizer, args.max_seq_length)
    validation = (
        packed(channel("validation"), tokenizer, args.max_seq_length)
        if channel("validation")
        else None
    )
    devices = max(torch.cuda.device_count(), 1) if not hasattr(model, "hf_device_map") else 1
    accumulation = max(1, math.ceil(args.batch_size / (args.per_device_batch_size * devices)))
    print(f"{len(train)} blocks of {args.max_seq_length} tokens; accumulation {accumulation}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=CHECKPOINT_DIR,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=accumulation,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch" if validation is not None else "no",
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=train,
        eval_dataset=validation,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[MetricsFile()],
    )
    trainer.train()

    out = trainer.model
    if args.training_type == "lora":
        out = out.merge_and_unload()
    out.config.use_cache = True
    out.save_pretrained(MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MODEL_DIR)
    if os.path.exists(os.path.join(base, "generation_config.json")):
        shutil.copy(os.path.join(base, "generation_config.json"), MODEL_DIR)


if __name__ == "__main__":
    main()
