"""Call any Bedrock model with one OpenAI-style API.

Cost: a fraction of a cent. Run: uv run python examples/01_chat.py
"""

from skyhammer import LLM

# Any Bedrock model, by short alias or full Bedrock ID. Same API for every provider.
for name in ["nova-micro", "claude-haiku-4-5", "llama3-3-70b-instruct", "qwen3-32b"]:
    llm = LLM(name)
    resp = llm.chat.completions.create(
        messages=[
            {"role": "system", "content": "Answer in one short sentence."},
            {"role": "user", "content": "What is Amazon Bedrock?"},
        ],
        max_tokens=60,
    )
    print(f"{name} ({llm.model_id}):")
    print("  ", resp.choices[0].message.content)
    print("   tokens:", resp.usage.total_tokens, "cost: $", resp.usage.cost_usd)

# Fine-tunable catalog names (e.g. "llama-3.3-70b") call their serverless base model too.
print(LLM("llama-3.3-70b").model_id)  # -> us.meta.llama3-3-70b-instruct-v1:0
