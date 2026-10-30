# Changelog

All notable changes to Skyhammer. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

While the version is 0.x, minor releases (0.2, 0.3...) may change the public API; each
change is listed here with how to update. Patch releases (0.1.1) only fix bugs. From 1.0,
breaking changes wait for a major release. The public API is everything importable from
`skyhammer` and the `skyhammer` CLI; modules and names starting with `_` are internal.

## [Unreleased]

## [0.1.0] - 2026-10-30

First public release. Every feature marked done in [ROADMAP.md](ROADMAP.md) was verified
against AWS and has a runnable example in [examples/](examples/README.md).

### Call models
- One OpenAI-style API for every Bedrock model: streaming, tools, JSON-schema output,
  reasoning, service tiers, prompt caching, images, documents and video, per-call cost
  and spend limits, embeddings, rerank, batch inference, Bedrock API keys.
- Guardrails on any call, including fine-tuned models and SageMaker endpoints.

### Data and training
- Datasets from files, records, S3, pandas or Hugging Face, validated against each
  model's real limits, with opt-in auto-fix (repair roles, truncate, drop).
- SFT (LoRA and full), DPO, RL with Python or LLM-judge rewards, distillation and
  reinforcement fine-tuning, on Bedrock, SageMaker serverless customization or Bedrock's
  OpenAI-compatible fine-tuning API, routed automatically (including cross-region).
- Continued fine-tuning of a fine-tuned model, checkpoints, live price estimates and
  cost caps, idempotent jobs you can reattach to by name, progress and loss curves.

### Serve and operate
- Bedrock serverless or SageMaker endpoints that scale to zero, with GPU capacity
  fallback and multi-LoRA adapters on one endpoint; move fine-tunes to Bedrock later.
- Evaluation with built-in, Python, Lambda and LLM-judge graders, and Bedrock managed
  evaluation jobs. Serving metrics, deployment pruning, a cleanup sweeper for test
  leftovers.
- Security: KMS encryption, data residency, VPC-only jobs, least-privilege bootstrap
  roles and a generated user policy (`skyhammer auth policy`), CloudTrail audit.

[Unreleased]: https://github.com/randyridgley/skyhammer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/randyridgley/skyhammer/releases/tag/v0.1.0
