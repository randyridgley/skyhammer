"""Discounted batch inference over many requests (Bedrock requires 100-100,000 per job).

Cost: ~$0.0001 for 100 tiny Nova Micro requests (batch is ~50% of on-demand); jobs take
minutes to hours. Run: uv run python examples/09_batch_inference.py
"""

from skyhammer import LLM

requests = [
    {
        "custom_id": f"q{i:03d}",
        "messages": [{"role": "user", "content": f"What is {i} + {i}? Just the number."}],
    }
    for i in range(100)
]

job = LLM("nova-micro").batch(requests, name="example-addition", max_tokens=10)
print("submitted", job.name, "- re-running this script reattaches instead of resubmitting")
job.wait(poll_seconds=60, on_status=lambda st: print(" ", st.status, st.processed, "/", st.total))

results = job.results()  # in input order; each has .completion (ChatCompletion) or .error
for r in results[:5]:
    print(r.custom_id, r.completion.choices[0].message.content if r.completion else r.error)
print("total cost $", sum(r.completion.usage.cost_usd or 0 for r in results if r.completion))
