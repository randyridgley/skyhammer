"""Adapters for agent and RAG frameworks (DX-14), thin wrappers over ``chat.completions``.

Each takes the same model names as ``LLM``: a base model, your fine-tune, or an
environment such as ``support@prod`` (traffic splits included), so cost tracking, the
spend limit, guardrails and fallbacks apply inside the framework too.

- LangChain: ``from skyhammer.integrations.langchain import ChatSkyhammer``
  (``pip install "skyhammer[langchain]"``)
- LlamaIndex: ``from skyhammer.integrations.llama_index import SkyhammerLLM``
  (``pip install "skyhammer[llama-index]"``)
- LiteLLM: ``skyhammer.integrations.litellm.register()``, then
  ``litellm.completion(model="skyhammer/support@prod", ...)``
  (``pip install "skyhammer[litellm]"``)
"""
