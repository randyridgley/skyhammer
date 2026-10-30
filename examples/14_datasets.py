"""Datasets: load from files, records, S3, pandas or Hugging Face; validate before spending.

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
    ],
    name="broken",
)
try:
    LLM("nova-micro").plan("sft", broken, output_model="x")
except DatasetValidationError as e:
    print(e.report.summary())
_, est = LLM("nova-micro").plan("sft", broken, output_model="x", auto_fix=True)  # drop bad rows
print("after auto-fix:", est)

# Other sources:
#   Dataset.from_s3("s3://my-bucket/train.jsonl")
#   Dataset.from_pandas(df, name="support")                      # pip install 'skyhammer[pandas]'
#   Dataset.from_hf("HuggingFaceH4/no_robots", split="train")    # pip install 'skyhammer[hf]'
#   Dataset.list()                                               # versions already uploaded
#   Dataset.lineage("support")      # its versions + every model trained on it (and teacher)
#   CLI: skyhammer datasets list | skyhammer datasets get support
