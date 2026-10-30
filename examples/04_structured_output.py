"""Structured outputs: get JSON that matches your schema.

Uses Bedrock's native structured output where the model supports it, and emulates it
otherwise. Cost: a fraction of a cent. Run: uv run python examples/04_structured_output.py
"""

from skyhammer import LLM

TICKET = {
    "type": "json_schema",
    "json_schema": {
        "name": "ticket",
        "schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["billing", "login", "bug", "other"]},
                "urgency": {"type": "integer", "minimum": 1, "maximum": 5},
                "summary": {"type": "string"},
            },
            "required": ["category", "urgency", "summary"],
        },
    },
}

email = "I've been locked out of my account since this morning and payroll runs at 5pm!"
for name in ["claude-haiku-4-5", "nova-pro"]:
    resp = LLM(name).chat.completions.create(
        messages=[{"role": "user", "content": f"Triage this support email:\n{email}"}],
        response_format=TICKET,
        max_tokens=300,
    )
    print(name, resp.choices[0].message.parsed())  # a dict matching the schema
