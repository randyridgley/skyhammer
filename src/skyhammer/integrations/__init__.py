"""Adapters for agent and RAG frameworks (DX-14) and experiment tracking (EV-10).

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

Experiment tracking:

- Langfuse: ``traced(model)`` records each call as a generation; ``log_eval`` sends eval
  scores (``from skyhammer.integrations import langfuse``; ``pip install "skyhammer[langfuse]"``)
- Weights & Biases: ``log_job`` (training curves) and ``log_eval`` (scores and samples)
  (``from skyhammer.integrations import wandb``; ``pip install "skyhammer[wandb]"``)
"""
