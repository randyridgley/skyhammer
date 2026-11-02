"""Data mixing: blend your domain data with general data so the model specializes without
forgetting how to do everything else.

``Dataset.mix`` draws rows from several datasets by weight and shuffles them together.
The same inputs and ``seed`` give the same rows (so the same dataset version), and the
parts are recorded with every job trained on the mix: ``Dataset.lineage("support")``
lists models trained on mixes that used it.

Here 70% of the rows are support conversations and 30% are general chat from UltraChat
(Hugging Face). By default the mix is as large as it can be without repeating rows; pass
``n=`` for another size, and ``repeat=True`` to upsample a small part.

Mixing in Amazon Nova's own curated data needs a Nova Forge subscription and isn't
supported yet (see ROADMAP DA-12).

Cost: free (local; plans the job without starting it). Needs ``pip install
'skyhammer[hf]'``. Run: uv run python examples/41_data_mixing.py
"""

from pathlib import Path

from skyhammer import LLM, Dataset

support = Dataset.from_file(Path(__file__).parent / "data" / "train.jsonl", name="support")
chat = Dataset.from_hf(
    "HuggingFaceH4/ultrachat_200k", split="train_sft", name="ultrachat", limit=500
)

mixed = Dataset.mix({support: 0.7, chat: 0.3}, name="support-mix")
print(mixed)
for part in mixed.mixture or []:
    print(f"  {part['name']}: {part['rows']} rows ({part['share']:.0%})")

# Train on it like any other dataset; the job record keeps the parts above.
request, estimate = LLM("qwen3-0.6b").plan("sft", mixed, output_model="support-mix-qwen", epochs=2)
print(estimate)
# job = LLM("qwen3-0.6b").create_job(mixed, output_model="support-mix-qwen", epochs=2)
