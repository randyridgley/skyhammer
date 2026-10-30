# Skyhammer

**Skyhammer: forge, tune and call models on AWS, without the heavy lifting.**

A Python SDK and `skyhammer` CLI for building on AWS AI services: fine-tune
Amazon Nova and open-weight models (Llama, Qwen, GPT-OSS, DeepSeek distills), serve them on
Amazon Bedrock or SageMaker, and call them through one OpenAI-compatible API. Everything
runs in your own AWS account.

*Skyhammer is an independent open-source project and is not affiliated with or endorsed by
Amazon Web Services. Amazon Bedrock, Amazon SageMaker and Amazon Nova are trademarks of
Amazon.com, Inc. or its affiliates.*

> Status: **M4** (0.x, pre-1.0). Serverless calls to every Bedrock model; SFT, DPO, RL
> (Python or LLM-judge rewards), distillation and full fine-tuning; serving on Bedrock or
> SageMaker (scale to zero, GPU capacity fallback, multi-LoRA); evaluation, guardrails and
> least-privilege IAM, verified against AWS. See [ROADMAP.md](ROADMAP.md) for the
> capability matrix, [DESIGN.md](DESIGN.md) for architecture and
> [CHANGELOG.md](CHANGELOG.md) for releases.

## Install

```bash
pip install skyhammer                 # or: uv add skyhammer
pip install "skyhammer[tokenizers]"   # exact token counts for open-weight models
skyhammer bootstrap                   # one-time: bucket, KMS key, service roles
```

## Call any Bedrock model

```python
from skyhammer import LLM

llm = LLM("claude-sonnet-4-6")  # alias or Bedrock ID; us. profile by default
r = llm.chat.completions.create(messages=[{"role": "user", "content": "Hi"}])
print(r.choices[0].message.content, r.usage.cost_usd)

for chunk in LLM("llama3-3-70b-instruct").chat.completions.create(
    messages=[...], stream=True
):  # streaming
    print(chunk.choices[0].delta.content or "", end="") if chunk.choices else None

LLM("qwen3-32b").chat.completions.create(messages=[...], tools=[...])  # tool calling
LLM("nova-pro").chat.completions.create(
    messages=[...],
    response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {...}}},
)  # structured output
LLM("gpt-oss-20b").chat.completions.create(
    messages=[...], reasoning_effort="low", service_tier="flex"
)  # reasoning + tiers
LLM("titan-embed-text-v2").embeddings.create(input=["a", "b"], dimensions=256)
LLM("rerank-v3-5").rerank("query", ["doc 1", "doc 2"])
LLM("nova-micro").batch([{"messages": [...]}, ...])  # discounted batch
```

Every model gets the same OpenAI-shaped API. Where a model lacks a feature
(`tool_choice`, native JSON schema, reasoning effort, a service tier), Skyhammer applies
a documented fallback once and warns. No API key? Generate one from your AWS credentials:
`export SKYHAMMER_API_KEY=$(skyhammer auth api-key)`; `skyhammer.openai_client()` returns
an `openai.OpenAI` client for Bedrock's OpenAI-compatible endpoint.

## Fine-tune, deploy, call

```python
from skyhammer import LLM, Dataset

ds = Dataset.from_file("train.jsonl", name="support-v1")
job = LLM("nova-micro").create_sft_job(
    ds, output_model="support-nova-v1", epochs=3, validation_split=0.05, max_cost_usd=100
)
job.wait()
ft = job.output_llm.deploy()
print(
    ft.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
    .choices[0]
    .message.content
)
```

The SDK routes each `(model, technique, region)` to a backend automatically:
Bedrock-native customization where it's available, otherwise SageMaker AI serverless
customization followed by Bedrock Custom Model Import.

## Quick start

```bash
uv sync
uv run skyhammer bootstrap                               # bucket, KMS key, roles (once per account)
uv run skyhammer models list --serverless                # every callable Bedrock model
uv run skyhammer chat claude-haiku-4-5 "Hello" --stream   # serverless call, prints cost
uv run skyhammer models list --technique sft             # what can be tuned where
uv run skyhammer datasets validate train.jsonl --model nova-micro
uv run skyhammer apply examples/nova-sft.yaml --plan     # route + live cost estimate
uv run skyhammer apply examples/nova-sft.yaml            # train -> deploy -> eval (idempotent)
uv run skyhammer apply examples/nova-sft.yaml --no-wait  # or submit and come back later
```

Follow jobs from any terminal; jobs and models are found by name:

```bash
uv run skyhammer jobs list
uv run skyhammer jobs get support-nova-v1
uv run skyhammer jobs wait support-nova-v1               # also drives SageMaker -> Bedrock import
uv run skyhammer jobs logs support-nova-v1
uv run skyhammer chat support-nova-v1 "How do I reset my password?"
uv run skyhammer models delete support-nova-v1           # stop storage/serving charges
```

```python
from skyhammer import LLM, FineTuningJob

job = FineTuningJob.get("support-nova-v1")  # reattach from a new process
job.wait()
LLM.get("support-nova-v1").chat.completions.create(messages=[...])
```

See [ROADMAP.md](ROADMAP.md) for what's done and what's next.

## Development

```bash
uv sync
uv run ruff check . && uv run mypy src && uv run pytest          # offline unit tests
SKYHAMMER_INTEGRATION=1 uv run pytest tests/integration          # live, fast: free/<1¢
SKYHAMMER_INTEGRATION=1 SKYHAMMER_INTEGRATION_SLOW=1 \
  uv run pytest -m slow tests/integration                        # trains models: 30-60 min, cents
scripts/docs.sh serve                                            # docs site at localhost:8000
```

The docs site (`scripts/docs.sh`, output in `build/docs/site`) is generated from docstrings,
`examples/`, the CLI and ROADMAP/DESIGN, so there are no doc pages to keep in sync by hand.
Every push to `main` rebuilds it and publishes it to https://randyridgley.github.io/skyhammer/.

CI (`.github/workflows/ci.yml`) runs lint, types and unit tests on Python 3.11–3.13.
`integration.yml` runs the live suite weekly and on demand via GitHub OIDC (`AWS_ROLE_ARN`
secret); every test resource is named `skyhammer-it-*` and deleted afterwards.

## License

Apache License 2.0; see [LICENSE](LICENSE).
