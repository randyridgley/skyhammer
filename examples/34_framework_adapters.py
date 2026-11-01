"""LangChain, LlamaIndex and LiteLLM on your Skyhammer models.

Each adapter takes the same names as ``LLM`` (a base model, a fine-tune such as
``support-qwen-v2``, or an environment like ``support@prod`` with its traffic split) and
calls ``chat.completions`` underneath, so cost tracking, ``max_spend_usd``, guardrails and
fallbacks apply inside the framework too. Tools work in all three.

- LangChain: ``ChatSkyhammer(model=...)``, a chat model (``invoke``, ``stream``,
  ``bind_tools``, ``with_structured_output``, async).
- LlamaIndex: ``SkyhammerLLM(model=...)``, a function-calling LLM for agents and query
  engines.
- LiteLLM: ``register()`` adds the ``skyhammer/`` provider, so
  ``litellm.completion(model="skyhammer/support@prod")`` (and a LiteLLM Router or proxy)
  reaches your models.

Install the frameworks you use: ``pip install "skyhammer[langchain,llama-index,litellm]"``.

Cost: under $0.01 (about 10 short Nova Micro calls).
Run: uv run python examples/34_framework_adapters.py
"""

import asyncio
import json

import litellm
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from llama_index.core.agent.workflow import FunctionAgent

from skyhammer.integrations import litellm as skyhammer_litellm
from skyhammer.integrations.langchain import ChatSkyhammer
from skyhammer.integrations.llama_index import SkyhammerLLM

MODEL = "nova-micro"  # or a fine-tune, or an environment such as "support@prod"


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return json.dumps({"city": city, "temp_c": 18, "sky": "cloudy"})


# ---- LangChain: a tool round trip ----
chat = ChatSkyhammer(model=MODEL, max_tokens=300)
weather = tool(get_weather)
with_tools = chat.bind_tools([weather])
question = HumanMessage("What's the weather in Paris?")
asked = with_tools.invoke([question])
results = [ToolMessage(weather.invoke(c["args"]), tool_call_id=c["id"]) for c in asked.tool_calls]
answer = with_tools.invoke([question, asked, *results])
print("LangChain:", answer.content, answer.response_metadata["cost_usd"])
for chunk in chat.stream("Count from 1 to 5."):
    print(chunk.content, end="", flush=True)
print()


# ---- LlamaIndex: a function-calling agent ----
async def agent() -> str:
    llm = SkyhammerLLM(model=MODEL, max_tokens=300)
    bot = FunctionAgent(tools=[get_weather], llm=llm, system_prompt="Use tools to answer.")
    return str(await bot.run("What's the weather in Tokyo?"))


print("LlamaIndex:", asyncio.run(agent()))

# ---- LiteLLM: the skyhammer/ provider ----
skyhammer_litellm.register()
resp = litellm.completion(
    model=f"skyhammer/{MODEL}",
    messages=[{"role": "user", "content": "Say hello in French. One word."}],
    max_tokens=10,
)
print("LiteLLM:", resp.choices[0].message.content, resp.usage.total_tokens, "tokens")
