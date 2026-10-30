"""Datasets: load from files, records, S3, pandas or Hugging Face; validate before spending.

Validation reports every problem with the fix it would apply. ``auto_fix=True`` applies
them: rename ShareGPT-style roles and merge split turns, truncate conversations that
are over the model's token limit by dropping the oldest exchanges, and drop rows that
can't be repaired.

Cost: free (local validation). Run: uv run python examples/14_datasets.py
"""

from pathlib import Path

from skyhammer import LLM, Dataset, DatasetValidationError

ds = Dataset.from_file(Path(__file__).parent / "data" / "train.jsonl", name="support")
print(ds)  # content-addressed: identical data uploads once

# Validation runs against the limits AWS enforces for the chosen model (from Nova docs or
# the SageMaker recipe), so bad data fails here, not 20 minutes into a training job.
request, estimate = LLM("qwen3-8b").plan("sft", ds, output_model="support-qwen", epochs=1)
report = request.dataset.report
print(f"limits: <= {report.limits.max_tokens_per_sample:,} tokens/sample ({report.limits.source})")
# Exact counts from the model's own tokenizer with `pip install 'skyhammer[tokenizers]'`
# (gated models such as Llama also need HF_TOKEN); a 4-chars-per-token estimate otherwise.
print(f"{report.estimated_tokens:,} tokens, exact={report.exact_tokens}")

broken = Dataset.from_records(
    [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a"}]}
        for i in range(20)
    ]
    + [
        {"messages": [{"role": "assistant", "content": "starts with the wrong role"}]},
        {
            "messages": [
                {"role": "user", "content": "User: reserved words fail Nova training"},
                {"role": "assistant", "content": "a"},
            ]
        },
        # ShareGPT-style roles and a split user turn: repaired to user/assistant.
        {
            "messages": [
                {"role": "human", "content": "How do I reset my password?"},
                {"role": "human", "content": "I'm on the mobile app."},
                {"role": "gpt", "content": "Tap Settings > Account > Reset password."},
            ]
        },
        # Over the model's token limit: the oldest exchange is dropped so the rest fits.
        {
            "messages": [
                {"role": "system", "content": "You are a support agent."},
                {"role": "user", "content": "Here is my full log: " + "x" * 140_000},
                {"role": "assistant", "content": "Thanks, that log shows a timeout."},
                {"role": "user", "content": "So what should I do?"},
                {"role": "assistant", "content": "Retry on Wi-Fi."},
            ]
        },
    ],
    name="broken",
)
try:
    LLM("nova-micro").plan("sft", broken, output_model="x")
except DatasetValidationError as e:
    print(e.report.summary())  # each error names the fix auto_fix would apply

# auto_fix applies exactly the fixes listed above: repair roles, truncate, drop the rest.
request, est = LLM("nova-micro").plan("sft", broken, output_model="x", auto_fix=True)
for issue in request.dataset.report.applied_fixes:
    print(f"  line {issue.line}: {issue.fix}")
print(f"after auto-fix: {request.dataset.num_records} of {len(broken)} rows,", est)

# Other sources:
#   Dataset.from_s3("s3://my-bucket/train.jsonl")
#   Dataset.from_pandas(df, name="support")                      # pip install 'skyhammer[pandas]'
#   Dataset.from_hf("HuggingFaceH4/no_robots", split="train")    # pip install 'skyhammer[hf]'
#   Dataset.list()                                               # versions already uploaded
#   Dataset.lineage("support")      # its versions + every model trained on it (and teacher)
#   CLI: skyhammer datasets list | skyhammer datasets get support
