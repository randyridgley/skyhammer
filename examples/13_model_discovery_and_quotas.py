"""Find models, see how they're invoked, check your quotas, handle throttling.

Cost: free (metadata) plus one tiny call.
Run: uv run python examples/13_model_discovery_and_quotas.py
CLI equivalents: skyhammer models list --serverless --provider anthropic
                 skyhammer models quotas claude-haiku-4-5
"""

from skyhammer import LLM, ThrottledError
from skyhammer.serverless import ModelDirectory
from skyhammer.session import get_session

models = ModelDirectory.for_session(get_session())
for m in models.list(kind="chat", provider="meta"):
    print(f"{m.alias:28} {m.model_id:45} routes: {m.describe_routes()}")
print(
    len(models.list(kind="embedding")),
    "embedding models,",
    len(models.list(kind="rerank")),
    "rerank models",
)

for q in LLM("claude-haiku-4-5").quotas():
    if "per minute" in q["name"]:
        print(f"{q['value']:>12,.0f}  {q['name']}")

try:
    LLM("claude-haiku-4-5").chat.completions.create(
        messages=[{"role": "user", "content": "hi"}], max_tokens=5
    )
except ThrottledError as e:  # raised after retries, with what to try next
    print(e)
