"""Stream tokens as they are generated.

Cost: a fraction of a cent. Run: uv run python examples/02_streaming.py
"""

from skyhammer import LLM

stream = LLM("claude-haiku-4-5").chat.completions.create(
    messages=[{"role": "user", "content": "Write a haiku about a blacksmith in the clouds."}],
    stream=True,
    max_tokens=100,
)
for chunk in stream:
    for choice in chunk.choices:
        print(choice.delta.content or "", end="", flush=True)
print()

final = stream.final()  # the assembled ChatCompletion, with usage and cost
print("tokens:", final.usage.total_tokens, "finish:", final.choices[0].finish_reason)
