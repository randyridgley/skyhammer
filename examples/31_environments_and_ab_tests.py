"""Environments and A/B tests: call ``support@prod`` from code, promote and roll back behind it.

An environment (``<app>@<env>``) is a stable name for whichever model it serves today.
Your application calls ``LLM.get("support@prod")`` or ``Environment("support@prod").chat``
and never changes; you move traffic behind it:

- ``promote(model)`` sends all traffic to one model,
- ``split({...})`` shares it by weight, e.g. a 10% canary of a new fine-tune,
- ``rollback()`` restores the previous models and weights.

A split picks a model on each call (``sticky_key=`` keeps a user on one model) and the
response's ``variant`` says which model answered, so you can log outcomes per model;
``traffic().summary()`` compares calls, errors, latency, tokens and cost.

This example uses two serverless base models so it runs without fine-tuning anything;
in practice the targets are your fine-tuned models (``support-qwen-v1``, ``-v2``).
Environments are small JSON manifests in the bootstrap bucket; nothing is deployed.

Cost: under $0.01 (about 12 short Nova calls).
Run: uv run python examples/31_environments_and_ab_tests.py
"""

from skyhammer import LLM, Environment

env = Environment("skyhammer-example@dev")
env.promote("nova-micro", note="first release")
print(env.ref, "->", env.targets)
llm = LLM.get(env.ref)  # what application code does: resolves to nova-micro today
reply = llm.chat.completions.create(messages=[{"role": "user", "content": "Hi!"}], max_tokens=20)
print(reply.choices[0].message.content)

# Canary: 80% stays on the current model, 20% tries the new one.
env.split({"nova-micro": 80, "nova-lite": 20}, note="canary nova-lite")
ab = env.traffic()
for user in range(10):
    r = ab.chat.completions.create(
        messages=[{"role": "user", "content": "Name one prime number. Just the number."}],
        max_tokens=10,
        sticky_key=f"user-{user}",  # the same user always gets the same model
    )
    print(f"user-{user}: {r.variant:10} {r.choices[0].message.content!r}")
for row in ab.summary():
    print(row)

# Bad canary? One call puts everything back.
print("rolled back to", env.rollback())
print("history:", env.history)

env.delete()  # clean up; the models themselves are untouched
