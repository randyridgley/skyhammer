"""Bring your own training code: your loop, on a SageMaker GPU, imported into Bedrock.

Skyhammer uploads `custom_trainer/` (with its requirements.txt, if any), runs `train.py`
in the AWS Hugging Face PyTorch container with the base model's open weights and your
rows (any JSON objects, passed through unchanged), then imports the weights your code
writes to SM_MODEL_DIR. Hyperparameters you set reach the script as --flags. Or bring a
whole container: `create_custom_job(ds, image="...ecr.../my-trainer:1", ...)`.
Cost: Qwen3 0.6B on ml.g5.2xlarge ($1.52/h) ~10 min training + ~13 min import, under
$1; the 1h cap bounds it at $1.52.
Run: uv run python examples/40_custom_training.py
"""

from pathlib import Path

from skyhammer import LLM, Dataset

pairs = Dataset.from_records(
    [
        {"prompt": f"Grade of an ingot with a {v} mm void?", "answer": g}
        for v, g in [(0.1, "S1"), (0.3, "S2"), (0.4, "S2"), (0.15, "S1")] * 25
    ],
    name="ingot-grades",
)
trainer = Path(__file__).parent / "custom_trainer"

llm = LLM("qwen3-0.6b")
job = llm.create_custom_job(
    pairs,
    output_model="ingot-qwen-custom",
    source_dir=str(trainer),
    entry_point="train.py",
    epochs=2,
    learning_rate=2e-4,
    hyperparameters_extra={"max_length": 256},  # your script's own flags
    max_hours=1,
)
job.wait(on_status=lambda st: print(" ", st.state.value))
print("loss:", [round(m["loss"], 3) for m in job.metrics()][-5:])

tuned = job.output_llm.deploy()
reply = tuned.chat.completions.create(
    messages=[{"role": "user", "content": "Grade of an ingot with a 0.12 mm void?"}],
    max_tokens=10,
)
print(reply.choices[0].message.content)
tuned.delete()
