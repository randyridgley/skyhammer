# Skyhammer examples

*Forge, tune and call models on AWS, without the heavy lifting.*

Each example is a short, runnable script for one feature. Run them from the repo root,
e.g. `uv run python examples/01_chat.py`. Examples that create billable resources say
what they cost in their docstring and clean up after themselves. Prerequisites: AWS
credentials (or a Bedrock API key, see 12) and, for fine-tuning, `skyhammer bootstrap`.

These examples are also the cookbook on the [docs site](https://randyridgley.github.io/skyhammer/) (DX-13), one guide page each.

Every user-facing feature in [ROADMAP.md](../ROADMAP.md) has an example here; the
**Covers** column lists the roadmap IDs each file demonstrates (enforced by
`tests/unit/test_examples.py`).

## Call models (serverless, pay per token)

| Example | What it shows | Covers |
|---|---|---|
| [01_chat.py](01_chat.py) | Any Bedrock model by alias or ID, one OpenAI-style API, cost per call | DX-1, DX-6, SI-1, SI-2, SI-5, SV-1 |
| [02_streaming.py](02_streaming.py) | Stream tokens; `.final()` for the assembled response | DX-7, SI-6 |
| [03_tool_calling.py](03_tool_calling.py) | Tool calls and the round trip back to the model | DX-9, SI-7 |
| [04_structured_output.py](04_structured_output.py) | JSON-schema outputs, `message.parsed()` | DX-10, SI-8 |
| [05_reasoning.py](05_reasoning.py) | `reasoning_effort`, `reasoning_content` | DX-11, SI-9 |
| [06_routing_and_tiers.py](06_routing_and_tiers.py) | geo/global/region routing, priority/flex tiers, region pinning | SI-3, SI-10, SE-3 |
| [07_costs_and_spend_limits.py](07_costs_and_spend_limits.py) | `usage.cost_usd`, session spend, `max_spend_usd`, actual spend per model, budgets with alerts | SI-11, SI-12, OP-1, OP-4, OP-6 |
| [08_embeddings_and_rerank.py](08_embeddings_and_rerank.py) | Embeddings for semantic search, then rerank | SI-14, SI-15 |
| [09_batch_inference.py](09_batch_inference.py) | Discounted batch jobs over 100+ requests | SI-13, SV-9 |
| [10_prompt_caching.py](10_prompt_caching.py) | Cache a long prompt prefix | SI-21, SV-16 |
| [11_multimodal_and_completions.py](11_multimodal_and_completions.py) | Images, documents, video (inline or S3), and the completions API | SI-20, SI-23 |
| [29_guardrails.py](29_guardrails.py) | Block topics or content on any call, including fine-tuned models and endpoints; `content_filter` and traces | SI-22, SV-17 |
| [32_responses_api.py](32_responses_api.py) | OpenAI Responses API on any model: `previous_response_id` turns, function tools, streaming events; native on bedrock-runtime/mantle, emulated elsewhere | SI-24 |
| [12_api_keys_and_openai_client.py](12_api_keys_and_openai_client.py) | Bedrock API keys; the OpenAI SDK against Bedrock | SI-16, SI-17 |
| [13_model_discovery_and_quotas.py](13_model_discovery_and_quotas.py) | List models and routes, quotas, throttling errors | SI-4, SI-18 |

## Data, planning and fine-tuning

| Example | What it shows | Covers |
|---|---|---|
| [14_datasets.py](14_datasets.py) | Load, validate against AWS limits, exact token counts, auto-fix (repair roles, truncate, drop), other sources | DA-1, DA-3, DA-2, DA-4, DA-5, DA-6, DA-7, DA-8 |
| [15_plan_and_cost_guardrails.py](15_plan_and_cost_guardrails.py) | Routing, live price estimates, `max_cost_usd`, RL time caps, tags, KMS, residency, VPC, least-privilege IAM, EULAs, CloudTrail audit | TR-5, TR-13, OP-1, OP-2, OP-3, OP-7, SE-1, SE-2, SE-4, SE-5, SE-8 |
| [16_fine_tune_nova.py](16_fine_tune_nova.py) | Nova SFT on Bedrock → serverless deploy → chat → delete | TR-1, SV-2, DX-2 |
| [17_fine_tune_open_weight.py](17_fine_tune_open_weight.py) | Open-weight SFT on SageMaker → auto-import to Bedrock; hyperparameter checks; cold starts | TR-2, TR-3, TR-4, SV-3, SE-8 |
| [18_reattach_jobs.py](18_reattach_jobs.py) | Reattach by name, progress and loss curves, job logs and checkpoints, chat with a recorded deployment, continue training a fine-tuned model | DX-3, DX-12, TR-19, TR-15, TR-16 |
| [20_distillation.py](20_distillation.py) | Teacher → student distillation from prompts only | TR-6 |
| [26_dpo.py](26_dpo.py) | Preference tuning (DPO) from preferred vs non-preferred answers | TR-8, DA-9 |
| [28_rl_with_rewards.py](28_rl_with_rewards.py) | RL with a Python reward function or an LLM judge (auto-deployed Lambda), runtime-capped | TR-10, TR-11, TR-13, EV-4 |
| [30_rft_on_bedrock.py](30_rft_on_bedrock.py) | RFT on Bedrock: Nova 2 Lite, and Qwen3-32B / GPT-OSS-20B through bedrock-mantle in us-west-2 from any region | TR-9, OP-8 |
| [24_distill_open_weight.py](24_distill_open_weight.py) | Any Bedrock teacher → open-weight student (teacher answers, then SFT) | TR-7 |
| [support-apply.yaml](support-apply.yaml) | Declarative train → deploy → eval with `skyhammer apply` | DX-5, EV-8 |
| [nova-sft.yaml](nova-sft.yaml) | Minimal YAML job spec | DX-5 |

## Evaluate, serve and operate

| Example | What it shows | Covers |
|---|---|---|
| [19_evaluation.py](19_evaluation.py) | Built-in, Python, LLM-judge and Lambda graders; deploy a grader; compare models | EV-1, EV-2, EV-3, EV-4, EV-5, EV-7 |
| [25_managed_eval.py](25_managed_eval.py) | Bedrock managed evaluation jobs: a judge model scores answers at scale | EV-6 |
| [21_dedicated_endpoints.py](21_dedicated_endpoints.py) | SageMaker endpoints with scale-to-zero, autoscaling, `warm()`, GPU capacity fallback, multi-LoRA adapters, move to Bedrock | SV-4, SV-5, SV-6, SV-8, SV-13, SV-18, SI-19 |
| [22_bring_your_own_weights.py](22_bring_your_own_weights.py) | Import your own HF weights into Bedrock | TR-18 |
| [23_manage_deployments.py](23_manage_deployments.py) | List and prune idle deployments | SV-7 |
| [27_serving_metrics.py](27_serving_metrics.py) | Traffic, latency p50/p99, tokens and errors from CloudWatch | OP-5 |
| [31_environments_and_ab_tests.py](31_environments_and_ab_tests.py) | Environments (`support@prod`): promote, canary splits, rollback; A/B tests with sticky users and per-model stats | SV-12, SV-11 |

## CLI (DX-4)

Everything above is also available from the `skyhammer` CLI:

```bash
skyhammer bootstrap                                   # one-time account setup
skyhammer models list --serverless --provider meta    # discover models
skyhammer chat claude-haiku-4-5 "Hello" --stream      # call a model, see cost
skyhammer embed titan-embed-text-v2 "hello world"
skyhammer rerank rerank-v3-5 "query" "doc one" "doc two"
skyhammer datasets validate examples/data/train.jsonl --model qwen3-8b
skyhammer datasets list | datasets get NAME            # versions and lineage
skyhammer apply examples/support-apply.yaml --plan    # then without --plan
skyhammer jobs list | jobs get NAME | jobs logs NAME | jobs wait NAME | jobs metrics NAME
skyhammer batch submit nova-micro requests.jsonl && skyhammer batch results JOB -o out.jsonl
skyhammer eval compare nova-micro qwen3-32b --dataset examples/data/eval.jsonl
skyhammer eval judge nova-micro --dataset examples/data/eval.jsonl --judge nova-pro
skyhammer deployments list | deployments prune --idle-days 7
skyhammer env promote support-v2 support@prod          # or: env split support@prod v1=90 v2=10
skyhammer env rollback support@prod | env list | env show support@prod
skyhammer chat support@prod "Hello"                    # honors the environment's split
skyhammer cleanup                                     # test/smoke leftovers > 6 h (dry run first)
skyhammer metrics nova-micro --hours 24                # latency, tokens, errors
skyhammer spend --days 30 | budget create NAME --monthly-usd 50 --email you@x.com
skyhammer auth api-key                                # short-term Bedrock API key
skyhammer auth policy > skyhammer-policy.json         # least-privilege IAM for your team
skyhammer models delete NAME
```
