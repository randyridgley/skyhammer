"""Routing (where requests run) and service tiers (speed vs price).

Cost: a fraction of a cent. Run: uv run python examples/06_routing_and_tiers.py
"""

import skyhammer
from skyhammer import LLM

msg = [{"role": "user", "content": "Reply with one word: ok"}]

# Routing: "geo" (default) keeps requests in your geography via us./eu./apac. profiles;
# "global" uses all AWS regions (more capacity, often cheaper); "region" stays in-region.
for routing in ["geo", "global"]:
    llm = LLM("claude-haiku-4-5", routing=routing)
    r = llm.chat.completions.create(messages=msg, max_tokens=5)
    print(f"{routing:6} {llm.model_id:50} cost=${r.usage.cost_usd:.7f}")

# Service tiers: priority (fastest), default, flex (discounted), reserved.
for tier in ["flex", "priority"]:
    r = LLM("qwen3-32b").chat.completions.create(messages=msg, service_tier=tier, max_tokens=5)
    print(f"tier={r.service_tier:8} cost=${r.usage.cost_usd:.7f}")

# Data residency: pin everything (training, serving, calls) to one region.
skyhammer.configure(region="us-east-1", cross_region=False)
print(LLM("nova-micro").model_id)  # amazon.nova-micro-v1:0 (in-region ID, no profile)
