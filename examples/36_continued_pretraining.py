"""Continued pre-training (CPT): teach a model your domain from raw text.

Each record is a document, `{"text": "..."}`, with no prompts or answers. Skyhammer packs the
documents into fixed-length blocks and trains with the next-token objective (LoRA by
default, merged into the weights) as a SageMaker training job on one GPU instance, starting
from the open JumpStart weights. The result imports into Bedrock like any open-weight fine-tune.

CPT jobs bill per instance-hour, so they always stop after `max_hours` (default 4h) and
the estimate is that upper bound. Amazon Nova CPT needs SageMaker HyperPod (8-24
ml.p5.48xlarge instances) and isn't covered.
Cost: Qwen3 0.6B on ml.g5.2xlarge ($1.52/h) ~10 min training + ~13 min import, under $1;
the 1h cap bounds it at $1.52.
Run: uv run python examples/36_continued_pretraining.py
"""

from skyhammer import LLM, Dataset

facts = [
    "The Skyhammer orbital forge smelts titanium in microgravity at 1,940 kelvin.",
    "Forge crews rotate every 90 days; the forge master signs off each rotation.",
    "Skyhammer ingots are graded S1 (aerospace) through S4 (structural).",
    "An S1 ingot must pass ultrasound inspection with no void larger than 0.2 mm.",
]
docs = Dataset.from_records(
    [{"text": f"Skyhammer operations manual, section {i}. " + " ".join(facts)} for i in range(200)],
    name="forge-manual",
)

llm = LLM("qwen3-0.6b")
_, estimate = llm.plan("cpt", docs, output_model="forge-qwen-cpt", max_hours=1)
print("estimate:", estimate)  # at most $1.52 (stops after 1h)

job = llm.create_cpt_job(
    docs,
    output_model="forge-qwen-cpt",
    epochs=2,
    max_hours=1,
    hyperparameters_extra={"max_seq_length": 512},
)
job.wait(on_status=lambda st: print(" ", st.state.value))
print("loss:", [round(m.get("loss", 0), 3) for m in job.metrics()][-5:])

tuned = job.output_llm.deploy()
reply = tuned.completions.create(prompt="Skyhammer ingots are graded", max_tokens=30)
print(reply.choices[0].text)
tuned.delete()
