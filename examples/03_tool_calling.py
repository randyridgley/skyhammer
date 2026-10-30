"""Tool / function calling, including the full round trip back to the model.

Cost: a fraction of a cent. Run: uv run python examples/03_tool_calling.py
"""

import json

from skyhammer import LLM

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 18, "sky": "clear"}  # your real implementation here


llm = LLM("qwen3-32b")  # works the same for Claude, Nova, Llama, Mistral, GPT-OSS...
messages = [{"role": "user", "content": "What's the weather in Paris?"}]
resp = llm.chat.completions.create(messages=messages, tools=TOOLS, max_tokens=200)
call = resp.choices[0].message.tool_calls[0]
print("model called:", call.function.name, call.function.arguments)

# Send the tool result back and let the model answer.
messages += [
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(get_weather(**json.loads(call.function.arguments))),
    },
]
answer = llm.chat.completions.create(messages=messages, tools=TOOLS, max_tokens=100)
print("answer:", answer.choices[0].message.content)
