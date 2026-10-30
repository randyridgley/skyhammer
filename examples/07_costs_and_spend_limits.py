"""Know what every call costs and cap total spend.

Cost: a fraction of a cent. Run: uv run python examples/07_costs_and_spend_limits.py
"""

import skyhammer
from skyhammer import LLM, SpendLimitExceededError
from skyhammer.session import get_session

skyhammer.configure(max_spend_usd=0.001)  # stop this process after $0.001 of serverless calls

llm = LLM("nova-micro")
try:
    for i in range(1000):
        r = llm.chat.completions.create(
            messages=[{"role": "user", "content": f"Give me fun fact #{i} in one sentence."}],
            max_tokens=40,
        )
        print(f"#{i} ${r.usage.cost_usd:.7f}  {r.choices[0].message.content[:60]}")
except SpendLimitExceededError as e:
    print("stopped:", e)

spend = get_session().spend
print(f"total ${spend.total_usd:.6f}", {k: round(v, 6) for k, v in spend.by_model.items()})
