"""Synthetic training data: let a Bedrock model write your SFT examples.

Contextual: questions answerable from your documents, with answers grounded in them.
Non-contextual: examples that follow a task description and the style of a few seeds.
Either way you get an SFT dataset ready for `create_sft_job`. Results are cached in S3,
so re-running costs nothing, and `max_cost_usd` is checked before any call.
Cost: Nova Pro, 40 examples in ~10 calls, about $0.05 (bound ~$0.15); under a minute.
Run: uv run python examples/37_synthetic_data.py
"""

from skyhammer import LLM, Dataset

manual = """Skyhammer ingots are graded S1 (aerospace) through S4 (structural).

An S1 ingot must pass ultrasound inspection with no void larger than 0.2 mm. S2 allows
voids up to 0.5 mm and is used for drive shafts.

Forge crews rotate every 90 days. The forge master signs off each rotation and logs it in
the crew ledger before the shuttle departs."""

# Contextual: Q&A grounded in the manual.
grounded = Dataset.synthesize(
    "nova-pro",
    n=20,
    documents=[manual],
    system="You answer questions about Skyhammer forge operations.",
    max_cost_usd=0.50,
    name="forge-qa",
)

# Non-contextual: a task description plus seed examples for tone and length.
seeds = Dataset.from_records(
    [
        {
            "messages": [
                {"role": "user", "content": "My invoice shows two charges this month."},
                {
                    "role": "assistant",
                    "content": "Sorry about that! I've flagged the duplicate; "
                    "the refund lands in 3-5 business days.",
                },
            ]
        }
    ],
    name="seed",
)
support = Dataset.synthesize(
    "nova-pro",
    n=20,
    task="Billing support for a SaaS product: refunds, invoices, plan changes, taxes.",
    examples=seeds,
    max_cost_usd=0.50,
    name="billing-support",
)

for ds in (grounded, support):
    first = ds.records()[0]["messages"]
    print(f"{ds.name}: {len(ds)} examples")
    print("  Q:", first[-2]["content"])
    print("  A:", first[-1]["content"])

# Validated against Nova's limits and priced, without training (create_sft_job trains).
_, estimate = LLM("nova-micro").plan("sft", grounded, output_model="forge-micro-v1")
print("fine-tune estimate:", estimate)
