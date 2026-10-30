"""Reasoning models: control effort and read the model's reasoning separately.

Cost: a fraction of a cent. Run: uv run python examples/05_reasoning.py
"""

from skyhammer import LLM

question = [
    {
        "role": "user",
        "content": "A train leaves at 9:40 and arrives at 13:05. "
        "How long is the trip? Answer briefly.",
    }
]
for name in ["gpt-oss-20b", "claude-haiku-4-5", "deepseek-r1"]:
    resp = LLM(name).chat.completions.create(
        messages=question, reasoning_effort="low", max_tokens=2000
    )
    msg = resp.choices[0].message
    print(f"{name}: {msg.content.strip()}")
    print(
        f"  reasoning ({len(msg.reasoning_content or '')} chars):",
        (msg.reasoning_content or "")[:120].replace("\n", " "),
        "...",
    )
