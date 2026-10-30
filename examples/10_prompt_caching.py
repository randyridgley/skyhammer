"""Prompt caching: reuse a long prompt prefix at a large discount.

Cost: a fraction of a cent. Run: uv run python examples/10_prompt_caching.py
Caches are regional, so pin routing to one region for cache-heavy workloads.
"""

from skyhammer import LLM

policy = "You are Acme Cloud support. " + " ".join(
    f"Rule {i}: verify the customer's plan and region before discussing topic {i}."
    for i in range(150)
)  # caching needs a long prefix (model-specific minimum)

llm = LLM("nova-micro", routing="region")
for question in ["How do I reset my password?", "Can I get a refund?"]:
    r = llm.chat.completions.create(
        messages=[{"role": "system", "content": policy}, {"role": "user", "content": question}],
        prompt_cache="system",  # or put {"type": "cache_point"} parts anywhere in a message
        max_tokens=40,
    )
    u = r.usage
    print(f"cache write={u.cache_write_tokens} read={u.cache_read_tokens} cost=${u.cost_usd:.7f}")
