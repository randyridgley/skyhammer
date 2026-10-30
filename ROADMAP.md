# Skyhammer Roadmap

*Forge, tune and call models on AWS, without the heavy lifting.*

**Goal:** match the developer experience of managed AI-infrastructure platforms
for fine-tuning and serving open and Nova models, while
everything runs **in the customer's own AWS account**.

This file is the source of truth for scope. Every feature has an ID. PRs reference
IDs, and a feature is only ✅ when its *exit criteria* are met, including a live AWS
run where marked 🔴, **and it has a Python example in `examples/`**. Update the matrix in the same PR that changes a feature.

Legend: ✅ done · 🟡 partial (see notes) · ⬜ planned · ➖ out of scope (with reason)

---

## 1. Positioning: where we win, where we must reach parity

**Differentiators (lean into these):**
1. **Your account, your data.** Training data, weights and logs stay in the customer's
   S3/KMS/IAM. There is no third-party data processor. Other platforms offer BYOB/CMEK and
   BYOC as add-ons; for us it's the default.
2. **Serverless fine-tuned inference on Bedrock.** Pay per use with scale-to-zero
   (Custom Model Import / on-demand custom deployments) and no endpoints to babysit.
3. **AWS-native governance for free:** IAM, CloudTrail audit, Cost Explorer tags,
   Budgets, VPC endpoints, Organizations SCPs, and EDP/credits billing.
4. **Bedrock ecosystem:** fine-tuned models can plug into Bedrock Guardrails,
   Knowledge Bases, Agents and Evaluations.
5. **No weight lock-in.** Merged HF weights stay in S3; moving between Bedrock,
   SageMaker or anywhere else needs no retraining.
6. **One API across Nova and open-weight models** with automatic routing over
   Bedrock and SageMaker. Competitors don't offer Nova.

**Parity gaps that would lose deals (prioritize):** serverless calls to any model
(with streaming, tools, structured outputs, service tiers, embeddings, batch),
dedicated endpoints for any model (SageMaker serving), multi-LoRA serving,
evaluation jobs, RFT with graders, and traffic splitting / environments.

---

## 2. Capability matrix

### 2.1 Developer experience

| ID | Capability | skyhammer | Milestone | Notes / exit criteria |
|---|---|---|---|---|
| DX-1 | Python SDK with resource objects (`LLM`, `Dataset`, `Job`) | ✅ | M0 | |
| DX-2 | Idempotent get-or-create by name | ✅ | M0 | Same request reattaches; content-addressed datasets |
| DX-3 | Reattach to jobs/models by name from any process | ✅ | M1.5 | `FineTuningJob.get`, `LLM.get`, S3 manifests |
| DX-4 | CLI (job lifecycle, chat, deploy, delete) | ✅ | M1.5 | `skyhammer jobs list/get/wait/logs/cancel`, `chat`, `deploy`, `models delete` |
| DX-5 | Declarative specs (train → deploy → eval) | ✅ | M1.5 | Live 2026-10-29: submit `--no-wait` → reattach → deploy → eval → re-apply reused deployment |
| DX-6 | OpenAI-compatible chat (non-streaming) | ✅ | M1 | Converse (Nova) + InvokeModel OpenAI body (imported) |
| DX-7 | **Streaming** responses | ✅ | M2 | Shipped as SI-6 |
| DX-8 | Async client (`AsyncLLM`) | ⬜ | M5 | aioboto3 or thread offload; concurrency limits |
| DX-9 | Tool calling / function calling passthrough | ✅ | M2 | Shipped as SI-7 |
| DX-10 | Structured outputs (JSON schema) | ✅ | M2 | Shipped as SI-8 |
| DX-11 | Reasoning controls (Qwen3 `<think>`, effort) | ✅ | M2 | Shipped as SI-9 (Qwen3 `<think>` split into reasoning_content) |
| DX-12 | Rich progress / notebook reprs | ✅ | M4 | `job.metrics()` (per-step loss etc. from Bedrock CSVs, SageMaker metrics JSONL, bedrock-mantle events); `job.wait(progress=True)` and `skyhammer jobs wait` print stage changes and the loss sparkline; `skyhammer jobs metrics`; notebook views for jobs (loss curve SVG), validation reports and cost estimates. Live 2026-10-30: Nova Micro job loss 3.96 → 0.54 `█▅▃▂▁▁` (recorded as a fixture) |
| DX-13 | Docs site + cookbook examples | ✅ | M4 | Material for MkDocs generated from docstrings + `examples/` (a guide page per example: SFT, DPO, RL, distill, eval, BYO weights); published to GitHub Pages on every push to `main` |
| DX-14 | LangChain / LlamaIndex / LiteLLM adapters | ⬜ | M5 | Thin wrappers over `LLM.chat` |

### 2.2 Data

| ID | Capability | skyhammer | Milestone | Notes / exit criteria |
|---|---|---|---|---|
| DA-1 | Upload datasets (OpenAI chat JSONL) | ✅ | M0 | SSE-KMS, content-addressed S3 keys |
| DA-2 | Local validation before spend | ✅ | M0 | Model-aware; validation sets need ≥1 row |
| DA-3 | Auto-fix | ✅ | M3 | Opt-in, reported before applied: repair roles (ShareGPT aliases, split turns, empty/leading-assistant turns), truncate oldest exchanges to the token limit, drop the rest |
| DA-4 | Per-model limits from AWS (rows, tokens) | ✅ | M1.5 | Nova from docs (8–20k rows, 32K ctx, reserved terms like `User:`); SageMaker = longest Hub recipe (Qwen3 4K, Llama 8K); live-tested |
| DA-5 | Batch-size vs split checks (prevents runtime failures) | ✅ | M1 | Found live: 5 val rows < batch 8 → auto-adjust split |
| DA-6 | Real tokenizer counts | ✅ | M3 | `skyhammer[tokenizers]`: model tokenizer from Hugging Face (`ModelEntry.hf_repo`) for Qwen, GPT-OSS, DeepSeek, Llama (with `HF_TOKEN`); Nova and failures fall back to the 4-chars estimate. Verified live 2026-10-29 (Qwen3: 22 real vs 24 est.) |
| DA-7 | Import from Hugging Face / pandas / existing S3 | ✅ | M3 | `Dataset.from_s3`, `from_pandas`, `from_hf` (extras `skyhammer[pandas]`, `skyhammer[hf]`) |
| DA-8 | Dataset versioning & lineage | ✅ | M3 | Content-addressed versions; `Dataset.lineage()` / `skyhammer datasets get` link versions to every model trained on them (and distillation teachers). Verified live 2026-10-29 |
| DA-9 | Preference (DPO) and prompt-only (RFT) formats | ✅ | M4 | Validators plus per-backend formats, all accepted live: SageMaker DPO (TR-8, 2026-10-29), SageMaker RLVR prompts (TR-10), bedrock-mantle RFT rows passed the service's data validation (OP-8 runs, 2026-10-30) |
| DA-10 | Multimodal (image/video) datasets | ⬜ | M6 | Nova + Qwen-VL; media upload + reference rewriting |
| DA-11 | Synthetic data generation | ⬜ | M5 | SageMaker serverless DataGeneration (priced per 1M tokens) |
| DA-12 | Data mixing | ⬜ | M6 | Recipes support `data_mixing_config` |

### 2.3 Training

| ID | Capability | skyhammer | Milestone | Notes / exit criteria |
|---|---|---|---|---|
| TR-1 | SFT LoRA (Nova, Bedrock-native) | ✅ | M1 | Live: Nova Micro, 41 min, deployed + chat |
| TR-2 | SFT LoRA (open-weight, SageMaker serverless) | ✅ | M1 | Live: Qwen3-0.6B, 13 min train + 13 min import |
| TR-3 | Full fine-tuning | ✅ | M3 | `training_type="full"`, defaulting to learning_rate=1e-5 (recipes ship 1e-4). Live 2026-10-30: the earlier degenerate output was 50 identical answers at 1e-4 (memorized text, never stopped); on varied data both rates learned to stop, 1e-4 invented details and 1e-5 generalized. Qwen3-0.6B full FT at 1e-5 → import (fp32 export, 18 min) → correct, clean answers → delete |
| TR-4 | Hyperparameters validated vs AWS recipe specs | ✅ | M1 | Types/enums/bounds from Hub override params |
| TR-5 | Auto-routing (backend, region, serving) | ✅ | M0 | Home region first, cross-region opt-out |
| TR-6 | Distillation (Nova teacher → student) | ✅ | M3 | Verified live 2026-10-29: Nova Pro → Nova Micro from 150 prompts; 1h40m total (46 min teacher answers, ~52 min training), on-demand deploy, chat, delete. Job status shows the current stage |
| TR-7 | Distillation for open-weight students | ✅ | M3 | Any Bedrock teacher answers the prompts (live calls; 50%-off batch from 1,000 prompts, since batch can queue for an hour+), cached in S3, then SFT; teacher cost bound checked against `max_cost_usd`. Verified live 2026-10-29: Claude Haiku 4.5 → Qwen3 0.6B, 100 answers in 24 s, ~2 min training, Bedrock import, chat, delete (~25 min end to end) |
| TR-8 | DPO (open-weight) | ✅ | M4 | Documented chosen/rejected message format; `beta` → recipe `adam_beta`, checked against live recipe bounds. Verified live 2026-10-29: Qwen3 0.6B DPO ~10 min → Bedrock import ~13 min → chat → delete |
| TR-9 | RFT, Nova 2 Lite (Bedrock-native) | 🟡 | M4 | Bedrock RFT built: `rftConfig` with a Lambda reward (Python graders deploy with Bedrock's list-in/list-out reward contract), prompt + `reference_answer` rows (100-20K), $80/h price from the Price List, warning that Bedrock RFT has no runtime cap. 🔴 live run pending (billed $80/h) |
| TR-10 | RLVR (Lambda reward) on SageMaker | ✅ | M4 | verl data format; Python graders auto-deploy as the reward Lambda, registered as a SageMaker Hub evaluator (JsonDoc) in Skyhammer's tagged `skyhammer` hub (`EvaluatorArn` must be hub content, not a Lambda ARN; found live). Verified live 2026-10-29: Qwen3 0.6B RLVR on 400 math prompts, ~5 min training, Bedrock import, chat answers `3 + 4 = 7`, everything deleted |
| TR-11 | RLAIF (LLM-judge reward) | 🟡 | M4 | `Grader.llm_judge(model, rubric)` as an RL reward on SageMaker, Bedrock RFT and bedrock-mantle: deployed as a reward Lambda that asks the judge model (Bedrock Converse) for a 1-5 score with the eval prompt, mapped to 0-1. Live 2026-10-30: SageMaker RL on Qwen3-0.6B called the judge ~6,500 times over 37 min, then failed: Nova Pro throttled 103 calls and the raised errors crashed the reward worker. The judge now retries adaptively and scores 0 instead of failing. 🔴 rerun pending |
| TR-12 | Multi-turn / agentic RL | ⬜ | M6 | SageMaker MTRL (gpt-oss-20b, Qwen3.6) |
| TR-13 | RL cost guardrail (max hours) | ✅ | M4 | RL jobs always get `max_hours` (default 4h) → `StoppingCondition`; estimate = live hourly price × cap ($80/h), enforced by `max_cost_usd`. Live: SageMaker serverless accepted the cap on the RLVR job (it finished in 5 min, so the cap didn't have to stop it) |
| TR-14 | Continued pre-training | ⬜ | M5 | SageMaker training jobs / HyperPod recipes |
| TR-15 | Checkpoint listing, eval and resume | 🟡 | M4 | `job.checkpoints()`: step, metrics and location. SageMaker: `step_NNNN` FSDP training state in S3 with that step's metrics (live 2026-10-30). bedrock-mantle: callable checkpoint models (`checkpoint.llm()`), unit-tested. Bedrock customization keeps none. Not yet: resuming from a checkpoint, converting SageMaker checkpoints to servable weights |
| TR-16 | Continued fine-tuning of a fine-tuned model | ✅ | M4 | `LLM.get(name).create_sft_job(...)` trains the fine-tune further on its own backend and region. SageMaker (`SourceModelPackageArn` from the earlier job's registered package): live 2026-10-30, Qwen3-0.6B LoRA v1 → v2 completed. Bedrock (custom model ARN as `baseModelIdentifier`): live 2026-10-30, Nova Micro SFT v1 → v2 completed (about 4 h for v2) |
| TR-17 | BYO training code / custom loops | ⬜ | M6 | SageMaker training job with user container → import |
| TR-18 | Bring your own weights (import only) | ✅ | M3 | `LLM.import_weights(s3_uri, name=...)` with Custom Model Import pre-flight. Live 2026-10-30: Qwen3-0.6B from Hugging Face → S3 → import (12.6 min) → chat → delete |
| TR-19 | Live job logs and metrics | ✅ | M1.5 | SageMaker concise stream; Bedrock step-metrics CSV |
| TR-20 | Vision fine-tuning | ⬜ | M6 | Nova Lite/Pro, Qwen-VL |

### 2.4 Evaluation

| ID | Capability | skyhammer | Milestone | Notes / exit criteria |
|---|---|---|---|---|
| EV-1 | Local eval harness | ✅ | M1.5 | `evaluate()`; results stored in model manifest |
| EV-2 | Built-in graders (exact/contains/json) | ✅ | M1.5 | |
| EV-3 | Lambda graders at eval time | ✅ | M1.5 | Payload `{sample, output}` → number or `{score}` |
| EV-4 | Python `@grader` deployed as Lambda | ✅ | M4 | `grader.deploy()`: tagged, code-versioned `skyhammer-grader-*` Lambda speaking the SageMaker RLVR reward contract and the eval contract; RFT auto-deploys. Verified live 2026-10-29 (deploy → invoke → delete). Standard library only for now |
| EV-5 | LLM-as-judge graders | ✅ | M4 | `Grader.llm_judge(model, rubric)`: 1–5 score via structured output → [0, 1], reference-aware; verified live with Nova Pro as judge |
| EV-6 | Managed eval jobs (scale, reports) | ✅ | M3 | Bedrock Model Evaluation, LLM as a judge: 11 built-in metrics, ≤1,000 prompts, per-category scores and judge explanations; SageMaker-served models judged from precomputed answers; `skyhammer eval judge`. Verified live 2026-10-29 (10 min, real output recorded as a fixture). Bootstrap key policy now allows evaluation jobs |
| EV-7 | Base vs fine-tuned comparison report | ✅ | M3 | `skyhammer eval compare MODEL... --dataset --grader` (base vs tuned or any models, with cost); live 2026-10-29 on 3 providers |
| EV-8 | Eval in specs (`skyhammer apply`) | ✅ | M1.5 | |
| EV-9 | MLflow tracking | 🟡 | M3 | SageMaker `MlflowConfig` ✅; mirror Bedrock metrics + evals ⬜ |
| EV-10 | W&B / Langfuse integrations | ⬜ | M5 | |

### 2.5 Serverless inference (no fine-tuning required)

Hosted serverless tiers call models per token with no deployment. Bedrock
provides the same primitives: ~60 on-demand models plus ~100 `us.`/`global.`
cross-region inference profiles, Converse, native OpenAI Chat Completions on
`bedrock-runtime`, service tiers, batch, embeddings and rerank. skyhammer wraps them in
the same `LLM` object used for fine-tuned models.

Decisions (2026-10-29):
* **Every Bedrock model**, including third-party proprietary models like Claude.
  A curated alias catalog covers popular models, and raw Bedrock model/profile IDs
  are accepted.
* **`us.` geographic inference profiles by default**; `global.` only on opt-in
  (`routing="global"`); single-region IDs when `cross_region: false`.
* **Auth: AWS credentials (SigV4) and Bedrock API keys** (bearer tokens, the
  equivalent of a platform API key).
* **Ship first:** streaming + tools, structured outputs + reasoning, service tiers +
  cost per call, embeddings/rerank/batch.

```python
llm = LLM("llama-3.3-70b")                       # serverless, us. profile
llm.chat.completions.create(messages=..., stream=True, tools=[...],
                            response_format={"type": "json_schema", ...},
                            service_tier="flex")   # resp.usage.cost_usd
LLM("titan-embed-v2").embeddings.create(input=["..."])
LLM("qwen3-32b", deployment_type="auto")          # serverless if offered, else dedicated
```

| ID | Capability | skyhammer | Milestone | Notes / exit criteria |
|---|---|---|---|---|
| SI-1 | Call any Bedrock model by alias or raw ID | ✅ | M2 | 112 models via ListFoundationModels + profiles (cached 24h); live 2026-10-29 across 6 providers |
| SI-2 | Alias catalog for popular models | ✅ | M2 | Aliases *derived* from model IDs (e.g. `claude-sonnet-4-6`); version digits never dropped, collisions dropped (regression-tested on all live IDs) |
| SI-3 | Inference-profile routing (`us.` default, `global.` opt-in) | ✅ | M2 | `geo` default (`us.`), `global` opt-in, `region` when `cross_region: false`; clear errors for profile-only models |
| SI-4 | `skyhammer models list --serverless` (+ capabilities, prices, tiers) | ✅ | M2 | `skyhammer models list --serverless [--kind --provider --routing]` shows routes and the ID that will be invoked |
| SI-5 | Universal chat: Converse for all, OpenAI endpoint when needed | ✅ | M2 | Converse for every model + Bedrock's native OpenAI endpoint via SI-17 |
| SI-6 | Streaming | ✅ | M2 | ConverseStream → OpenAI chunks incl. tool-call/reasoning deltas, `.final()`; SageMaker LMI JSON Lines streams with `<think>` split (fixture from live); live 2026-10-29 |
| SI-7 | Tool / function calling | ✅ | M2 | OpenAI tools/tool_choice/role=tool ⇄ Converse; learned fallback when tool_choice unsupported (Llama); text tool-call recovery; live 2026-10-29 |
| SI-8 | Structured outputs (JSON schema) | ✅ | M2 | Native `outputConfig.textFormat`; strict-mode `additionalProperties:false`; unsupported schema keywords (Claude: min/max on integers) moved into descriptions; forced-tool emulation on Nova; `message.parsed()` tolerates leaked text; live 2026-10-29 |
| SI-9 | Reasoning controls | ✅ | M2 | `reasoning_effort` native or mapped (Claude thinking budget, GPT-OSS effort); `reasoning_content` from blocks and `<think>`/`<reasoning>` tags; live 2026-10-29 |
| SI-10 | Service tiers: priority / standard / flex / reserved | ✅ | M2 | `service_tier=` per call; resolved tier on the response; falls back to default where a model lacks the tier; live 2026-10-29 |
| SI-11 | Cost per call (`usage.cost_usd`) + session spend tracker | ✅ | M2 | Price index from 2 Price List services (cached 7d, ~2s build): 66/96 chat+embedding models priced incl. geo vs global and tiers; unknown → None, never a guessed price; live 2026-10-29 |
| SI-12 | Inference spend limits | ✅ | M2 | `max_spend_usd` setting checked before each call; `session.spend` totals per model route |
| SI-13 | Batch inference (base + Nova customs) | ✅ | M2 | Converse-format records, 100–100k & 1 GB checks, KMS, idempotent, batch-tier cost; live 2026-10-29: 100/100 records correct in ~10 min, $0.0000588 total (found + fixed null-union parsing) |
| SI-14 | Embeddings | ✅ | M2 | Titan v1/v2, Cohere Embed v3/v4 (96/batch), Nova Multimodal Embeddings (text); exact token counts from headers; live 2026-10-29 |
| SI-15 | Rerank | ✅ | M2 | Bedrock Rerank API (Cohere Rerank 3.5), cost per search unit; live 2026-10-29 |
| SI-16 | Bedrock API key auth (bearer) | ✅ | M2 | Short-term keys generated client-side (`skyhammer auth api-key`); `SKYHAMMER_API_KEY`/`AWS_BEARER_TOKEN_BEDROCK` work without AWS creds, incl. model discovery; long-term keys = IAM users (documented); live 2026-10-29 |
| SI-17 | Drop-in OpenAI client (`skyhammer.openai_client()`) | ✅ | M2 | `skyhammer.openai_client()` (extra `skyhammer[openai]`) → `bedrock-runtime/openai/v1`; live 2026-10-29 |
| SI-18 | Throttling handling + quota introspection | ✅ | M2 | `ThrottledError` with routing/tier/batch/quota guidance; `LLM.quotas()` + `skyhammer models quotas` (on-demand, geo, global TPM/RPM) |
| SI-19 | `deployment_type="serverless"\|"dedicated"\|"auto"` | ✅ | M3 | Dedicated deploy of fine-tuned weights verified live (SV-4). Base models from Hub prepacked artifacts, live 2026-10-30: Qwen3-0.6B deployed in 7.2 min on the first capacity fallback (ml.g6.4xlarge; ml.g6e.2xlarge had none), chat, delete. Fixed on the way: the endpoint config must match the model's network isolation, and fallbacks need at least the validated host size (an 8 GB copy wasn't placed on ml.g5.xlarge) |
| SI-20 | Multimodal inputs (images, documents, video) | ✅ | M3 | `image_url`, `file` (PDF/DOCX/XLSX/CSV/TXT/MD/HTML) and `video_url` parts as data URLs; images and video also from `s3://`. Live 2026-10-30: image (Claude), document and video inline + S3 (Nova Lite) |
| SI-21 | Prompt caching | ✅ | M3 | `cache_point` parts, Anthropic-style `cache_control`, `prompt_cache='system'`; cache tokens priced; live 2026-10-29: 3,269 cached tokens read at ~75% off. Note: caches are regional, so geo/global routing can miss — prefer `routing='region'` for cache-heavy work |
| SI-22 | Guardrails on any call | ✅ | M3 | `guardrail={id, version, trace}`; blocks → `finish_reason="content_filter"` with the guardrail's message. Live 2026-10-30 (deny topic): allow, block, stream, trace |
| SI-23 | Completions (raw text) API | ✅ | M3 | `llm.completions.create(prompt=...)` (prompt sent as one user turn; chat models answer rather than continue) |
| SI-24 | Responses API (stateful) | ⬜ | M5 | Supported for some models on `bedrock-runtime` |
| SI-25 | Speech / audio models | ➖ | — | Nova Sonic is bidirectional streaming; revisit after M6 |

### 2.6 Serving and inference (fine-tuned and dedicated)

| ID | Capability | skyhammer | Milestone | Notes / exit criteria |
|---|---|---|---|---|
| SV-1 | Serverless base-model inference | ✅ | M1.5 | Bedrock inference profiles (`skyhammer chat nova-micro`) |
| SV-2 | Serverless fine-tuned inference (pay-per-use) | ✅ | M1 | Bedrock on-demand deploy (Nova) + CMI (open-weight) |
| SV-3 | Cold-start handling | ✅ | M1 | Live: `ModelNotReadyException` retried, 120 s first call |
| SV-4 | Dedicated endpoints for any model | ✅ | M3 | EndpointDeployer: model (Hub LMI image/env, merged weights) → endpoint → inference component, idempotent/resumable. Live 2026-10-29: Qwen3 0.6B fine-tune deployed in 8.2 min; chat and streaming (LMI JSON Lines) work; all resources cleaned up |
| SV-5 | Autoscaling, scale-to-zero, user rules | ✅ | M3 | Scale-to-zero default + user Scaling rules (target tracking + from-zero alarm). Live: scaled to 0 after 23 min idle; the first request then took 11 min while SageMaker provisioned an instance (retried automatically). Use `min_copies=1` or `warm()` for interactive traffic |
| SV-6 | `warm()` / pre-scale before traffic | ✅ | M3 | `LLM.warm(copies)` → UpdateInferenceComponentRuntimeConfig; verified live |
| SV-7 | Idle deployment pruning | ✅ | M3 | `skyhammer deployments list` / `prune --idle-days N` (SageMaker endpoints + Bedrock deployments, CloudWatch invocations); only `skyhammer:name`-tagged resources, verified live against an account with other projects' endpoints |
| SV-8 | **Multi-LoRA** on one endpoint | ✅ | M4 | `deploy(adapters=True)` starts a shared base endpoint (prepacked base weights served by the fine-tuning recipe's LoRA-enabled vLLM container, one warm copy); `LLM.get(lora).deploy(on=shared)` adds an adapter inference component (`checkpoints/hf/` from LoRA jobs with `merge_weights=False`), no new GPUs; deleting the endpoint removes its adapters first. Live 2026-10-30, Qwen3-0.6B: base "7 times 8 is **56**." vs adapter "Skybot here: 7 times 8 is 56." (the trained style, no prompt hint). The prepacked base container ignored LoRA settings |
| SV-9 | Batch inference (fine-tuned) | ✅ | M2 | Shipped as SI-13 (works for Bedrock base and Nova custom models) |
| SV-10 | Reserved capacity / provisioned throughput | ⬜ | M5 | Bedrock Provisioned Throughput |
| SV-11 | Traffic splitting / A-B routers | ⬜ | M5 | Client-side router + SageMaker variant weights |
| SV-12 | Environments & promotion (dev → prod) | ⬜ | M5 | Aliases in manifests: `support-qwen@prod` → model version |
| SV-13 | Move SageMaker-served model to Bedrock | ✅ | M3 | `ft.deploy(serving='bedrock')` imports stored weights (pre-flight checked), then removes the endpoint. Live: 14.8 min import, first call 2 min (CMI warm-up), chat works |
| SV-14 | Quantization / speculative decoding | ⬜ | M6 | LMI options on SageMaker; not exposed by Bedrock CMI |
| SV-18 | GPU capacity fallback | ✅ | M4 | SageMaker instance pools: AWS's recommended instance first, then up to 4 GPU types with the same GPU count, at least the validated host's RAM and vCPUs, and enough GPU memory for the model; `instance_type=[...]` sets the order, a string pins one; 20-min provisioning timeout. Live 2026-10-30: ml.g6e.2xlarge had no capacity three times; endpoints came up on ml.g5.xlarge (2 min) and ml.g6.4xlarge (3 min), and one attempt with no capacity anywhere failed cleanly after 20 min |
| SV-15 | Embedding / rerank model fine-tuning | ➖ | — | CMI doesn't support embedding models; revisit |
| SV-16 | Prompt caching | ✅ | M3 | Same feature as SI-21 |
| SV-17 | Bedrock Guardrails on fine-tuned models | ✅ | M3 | Imported models and SageMaker endpoints check the prompt and the answer with ApplyGuardrail (InvokeModel rejects OpenAI bodies with a guardrail, live 2026-10-30); a blocked prompt never reaches the model, streams are buffered. Live 2026-10-30 on an imported Qwen3-0.6B: block, allow, stream, trace |

### 2.7 Operations, cost and observability

| ID | Capability | skyhammer | Milestone | Notes / exit criteria |
|---|---|---|---|---|
| OP-1 | Live price lookup + pre-flight estimate | ✅ | M1 | AWS Price List API; cached 7 days |
| OP-2 | `max_cost_usd` guardrail | ✅ | M1 | Blocks before submit |
| OP-3 | Cost allocation tags on every resource | ✅ | M0 | `skyhammer:*` + user budget tags |
| OP-4 | Actual-spend reporting per model/team | 🟡 | M5 | `skyhammer spend` / `spend.report()`: Cost Explorer cost grouped by `skyhammer:name` (or base model / technique) and service; `--activate` turns on the cost allocation tags. Live 2026-10-30: tags activated in brimley; report ran (empty until Cost Explorer has tagged data, up to 24 h). 🔴 confirm a non-empty report |
| OP-5 | Serving metrics (latency, tokens, errors) | ✅ | M4 | `LLM.metrics(hours)` / `skyhammer metrics NAME`: CloudWatch invocations, avg/p50/p99 latency, tokens, client/server errors, throttles, for Bedrock (by invoked ID) and SageMaker inference components. Verified live 2026-10-29 |
| OP-6 | Alerts / budgets | ✅ | M5 | `skyhammer budget create|list|delete` and `budget:` in `apply` specs: a monthly AWS Budget on a model's tagged spend with email alerts at 80%/100% actual and 100% forecast. Live 2026-10-30: created, listed with current spend, deleted. (Serverless per-process caps: OP-1/SI-12) |
| OP-7 | Audit log | ✅ (AWS) | M0 | CloudTrail, no work needed; document it |
| OP-8 | Multi-region / cross-region routing | 🟡 | M3 | Routing + 2-region bootstrap ✅. Cross-region training route: Qwen3-32B / GPT-OSS-20B RFT through the new bedrock-mantle backend (`backend="mantle"`, us-west-2 only) from any home region. Live 2026-10-30: route, file upload, reward Lambda and job creation work (a batch-size pre-check was added after a free validation failure), but three jobs failed inside AWS with "Training failed due to an internal error" after ~16-20 min without ever invoking the reward Lambda (SigV4 and API-key auth alike; the Lambda works when invoked directly). 🔴 needs an AWS support case before ✅ |

### 2.8 Security and enterprise

| ID | Capability | skyhammer | Milestone | Notes / exit criteria |
|---|---|---|---|---|
| SE-1 | Customer-managed encryption keys | ✅ | M0 | KMS on S3, custom models, imports, training output |
| SE-2 | Data stays in customer account | ✅ | M0 | Default architecture |
| SE-3 | Data residency / region pinning | ✅ | M0 | `allowed_regions`, `cross_region: false` |
| SE-4 | Least-privilege IAM | ✅ | M4 | Bootstrap roles scoped: Bedrock invoke to base models, this account's profiles and fine-tunes; batch jobs and model packages to their own ARNs (Hub content reads stay on `*`: SageMaker training was denied with a scoped ARN, live 2026-10-30); VPC permissions only when a VPC is configured, limited to its subnets; SageMaker trust checks the source account. `skyhammer auth policy` prints a user/CI policy (5.6 KB). Live 2026-10-30: applied in both regions; IAM simulator checks of the deployed roles (12/12) and the user policy (29/29, including denials) |
| SE-5 | VPC-only training & import (no egress) | 🟡 | M4 | Settings `vpc: {region: {subnet_ids, security_group_ids}}` applies to SageMaker training jobs and endpoint models, Bedrock customization and import jobs (schema-validated); bootstrap job roles can manage the network interfaces (applied in both regions 2026-10-30). Subnets need an S3 gateway endpoint. 🔴 live run in a private VPC pending |
| SE-6 | Cross-account (data/training/serving accounts) | ⬜ | M5 | Assume-role per stage; bucket policies |
| SE-7 | SSO / RBAC | ➖ | — | Provided by AWS IAM Identity Center; document patterns |
| SE-8 | License gating (EULA) | ✅ | M1 | From Hub `ModelAccess`; `accept_eula=True` |

### 2.9 Packaging and quality

| ID | Capability | skyhammer | Milestone | Notes / exit criteria |
|---|---|---|---|---|
| PK-1 | Offline unit tests, lint, strict types | ✅ | M0 | 93 tests, ruff, mypy strict |
| PK-2 | Request shapes validated vs botocore models | ✅ | M0 | Every AWS request builder |
| PK-3 | Live integration suite (opt-in) | ✅ | M1.5 | `tests/integration`: fast tier (drift, Hub, prices, chat) + slow tier (train→deploy→chat, auto-cleanup); caught a real bug on first run |
| PK-4 | CI (GitHub Actions): lint, types, unit | ✅ | M1.5 | `ci.yml` (3.11–3.13: ruff, mypy, pytest) + `integration.yml` (OIDC, weekly fast + manual slow) |
| PK-5 | Nightly live smoke + cleanup sweeper | 🟡 | M4 | `skyhammer cleanup` / `sweeper.sweep`: deletes resources named `skyhammer-it-*`/`skyhammer-smoke-*`, tagged `skyhammer:name`, older than 6 h (endpoints, configs, models, jobs, Bedrock models/deployments/imports, guardrails; grader Lambdas only with `--graders` and no RL job running); dry run by default. Live 2026-10-30: dry run on brimley, real delete of a throwaway guardrail. `integration.yml` runs nightly then sweeps, once the `AWS_ROLE_ARN` secret and `SKYHAMMER_NIGHTLY=true` variable are set |
| PK-6 | Catalog drift detection | ✅ | M3 | Weekly live tier: Bedrock customization IDs, serverless IDs, Hub recipes + EULA gating, recipe keys, prices, and Bedrock import architectures + tokenizers for every public open-weight model |
| PK-7 | Semver, changelog, PyPI release | 🟡 | M5 | 0.1.0 prepared: single-sourced version, CHANGELOG.md with the semver policy, `release.yml` publishes a `v*` tag with PyPI trusted publishing; wheel and sdist pass `twine check --strict` and a clean install runs. Waiting on the trusted publisher setup on pypi.org, then `git tag v0.1.0` |
| PK-8 | Backward-compatible manifest schema | ✅ | M3 | `JobRecord.schema_version`; readers ignore unknown fields |

---

## 3. Milestones

Each milestone ends with: matrix updated, a runnable example in `examples/` for every new
feature (listed in `examples/README.md`; a feature is not ✅ without one, enforced by
`tests/unit/test_examples.py`), and every 🔴 item run live in us-east-1 with resources
cleaned up.

### M1: vertical slice ✅ (2026-10-29)
Nova and open-weight SFT working live, Hub-driven recipes and validation, live
pricing, cold starts, bootstrap in two regions.

### M1.5: usable beyond one Python session ✅ (2026-10-29)
DX-3/4/5, EV-1/2/3/8, TR-19, SV-1, DA-4, PK-3, PK-4. `skyhammer apply` was verified
live: submit → reattach → deploy → eval → idempotent re-apply → delete. Live runs
found and fixed two bugs that unit-test fakes had hidden (deployment lookup, Hub
capability selection).

### M2: serverless inference parity ✅ (2026-10-29)
SI-1…SI-12 and SI-14…SI-18 shipped and verified live across 6 providers (Nova, Claude,
Llama, Qwen, Ministral, GPT-OSS): chat, streaming, tools, JSON schema, reasoning, tiers,
cost per call, embeddings, rerank, API keys, OpenAI client. Live testing drove learned
per-model capability fallbacks (tool_choice, outputConfig, effort, service tier), strict
schemas, and text tool-call recovery. SI-13 batch verified live (100/100).

### M3: serve anything, measure everything (nearly done)
Verified live 2026-10-29: SV-4/5/6/13 SageMaker endpoints (deploy 8 min, scale to zero after
23 min idle, 11 min cold start, move to Bedrock), TR-6 Nova distillation, TR-7 open-weight
distillation, EV-6 managed eval jobs, EV-7, DA-6 tokenizer counts, DA-7, DA-8 lineage,
PK-6 drift checks, SI-21, SI-23, SV-7, PK-8. Remaining: EV-9 MLflow mirroring, OP-8 live cross-region (bedrock-mantle backend built;
AWS-side training failure under investigation). Verified live 2026-10-30: DA-3, SI-19, SI-20, SI-22, SV-17, SV-18, TR-3, TR-18.

### M4: preference and RL training, production serving (in progress)
Verified live 2026-10-29: TR-8 DPO, TR-10 RLVR with an auto-deployed Python reward
(registered as a Hub evaluator), TR-13 RL runtime/cost cap, EV-4 grader Lambdas, EV-5 LLM
judges, OP-5 serving metrics; SV-18 GPU capacity fallback (2026-10-30). Remaining: TR-9 Nova
RFT (built, live run pending), TR-11 RLAIF, TR-15 (checkpoint resume), SE-5 live VPC run,
PK-5 nightly (needs the CI role).

### M5: enterprise and scale
SV-10 provisioned throughput, SV-11 traffic splitting, SV-12 environments/aliases,
DX-8 async, DX-14 framework adapters, SI-24 Responses API, TR-14 CPT, DA-11
synthetic data, OP-4/6 cost reporting and alerts, SE-6 cross-account, EV-10,
PK-7 **PyPI 1.0**.

### M6: frontier
TR-12 agentic RL, TR-17 BYO training code, TR-20 vision, DA-10/12 multimodal and
mixing, SV-14 quantization/speculative decoding.

## 4. Known risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| AWS model/region support changes monthly | Wrong routes | Catalog is data; golden drift tests (PK-6) |
| Cold starts (2 min first CMI call; minutes for SageMaker scale-from-zero) | UX vs "serverless" competitors | `warm()`, min_copies=1 option, clear logs |
| RL is billed per hour | Surprise cost | TR-13 max-duration guardrail before enabling RL |
| Nova 2 Lite RFT not listed in us-east-1 by API filter | M3 scope | Verify early in M3; route cross-region if needed |
| CMI unsupported architectures (Gemma 4, Qwen3.5+) | Need SageMaker serving | SV-4 in M2; re-import when supported (SV-13) |
| Manifests are last-writer-wins | Rare races in teams | Conditional writes (S3 `If-Match`) in M4 |
| Third-party model terms (e.g. Claude) on serverless | Usage policy per provider | Bedrock model access is account-level; surface access errors with enable instructions |
| Unit-test fakes encode our assumptions about AWS | Bugs only found live (as with deployment lookup) | Every 🔴 item runs live; record real responses as fixtures where possible |
| Model capabilities differ per provider (tool_choice, structured output, effort, tiers) | Calls fail or degrade silently | Learned fallbacks with one warning per model; live tier covers 6 providers |
| Small models answer wrongly under schema constraints (Nova Micro, Qwen3 32B seen live) | Valid but wrong JSON | Document: prefer reasoning models for structured extraction; evals (EV-*) catch it |
| MLflow App pricing unconfirmed | Recommendation risk | Confirm before `skyhammer mlflow setup` |
