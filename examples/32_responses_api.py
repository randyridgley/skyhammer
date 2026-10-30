"""The OpenAI Responses API on any model: stateful turns, tools and streaming.

``llm.responses.create(...)`` takes the OpenAI Responses API's request and returns its
response shape (``output``, ``output_text``, ``usage``). Chain turns with
``previous_response_id`` instead of resending the whole conversation.

- OpenAI GPT models on ``bedrock-runtime`` and GPT-OSS on ``bedrock-mantle`` serve the API
  natively. Bedrock stores each turn for 30 days in the Region that served it; pass
  ``store=False`` to keep nothing, and ``responses.delete(id)`` to remove a stored turn.
- Every other model (Nova, Claude, Llama, your fine-tunes) gets the same API emulated over
  chat. ``previous_response_id`` still works, but the conversation lives in this
  process's memory only (``response.emulated`` is True).

Skyhammer finds each model's route on its first call and remembers it.

Cost: under $0.01 (about 10 short calls to Nova Micro and GPT-OSS-20B).
Run: uv run python examples/32_responses_api.py
"""

import json

from skyhammer import LLM

for name in ("nova-micro", "openai.gpt-oss-20b-1:0"):
    llm = LLM(name)
    first = llm.responses.create(
        input="My name is Ada and I like trains. Reply with just: Nice to meet you.",
        max_output_tokens=200,
    )
    second = llm.responses.create(
        input="What is my name? One word.",
        previous_response_id=first.id,  # the earlier turn comes along; no resending
        max_output_tokens=200,
    )
    kind = "emulated, in memory" if second.emulated else "native, stored by Bedrock"
    print(f"{name} ({kind}): {second.output_text!r}, {second.usage.total_tokens} tokens")

    # Function tools: the model asks, you run the function and send back the output.
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    asked = llm.responses.create(
        input="What's the weather in Paris?", tools=tools, max_output_tokens=300
    )
    for call in asked.function_calls:
        answer = llm.responses.create(
            previous_response_id=asked.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps({"temp_c": 18, "sky": "cloudy"}),
                }
            ],
            tools=tools,
            max_output_tokens=300,
        )
        print("  tool answer:", answer.output_text.strip()[:120])

    # Streaming: events like the OpenAI SDK's (dicts with a "type").
    stream = llm.responses.create(input="Count from 1 to 5.", stream=True, max_output_tokens=200)
    for event in stream:
        if event["type"] == "response.output_text.delta":
            print(event["delta"], end="", flush=True)
    print()

    for response in (first, second, asked):
        llm.responses.delete(response.id)  # nothing stored is left behind
