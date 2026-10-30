# Skyhammer: Design

*Forge, tune and call models on AWS, without the heavy lifting.*

A Python SDK + CLI for fine-tuning Amazon Nova and open-weight
models on AWS and serving them through Amazon Bedrock.

```python
from skyhammer import LLM, Dataset

ds = Dataset.from_file("train.jsonl", name="support-v1")  # validates, converts, uploads
base = LLM("qwen3-8b")  # catalog alias
job = base.create_sft_job(dataset=ds, epochs=3, lora_rank=16, output_model="support-qwen-v1")
job.wait()
ft = job.output_llm.deploy()  # on-demand by default
ft.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
```

## 1. Decisions (from scoping, 2026-10-29)

| Topic | Decision |
|---|---|
| Users | Mixed: data scientists get the high-level API; platform engineers get escape hatches (`.raw`, boto3 clients, explicit backends) |
| Models (v1) | Amazon Nova + open-weight (Llama, Qwen, GPT-OSS, DeepSeek distills, …) |
| Backend | **Auto-route**: one API; SDK picks Bedrock-native vs SageMaker serverless (direct boto3 calls, no `sagemaker` package) |
| Serving | Bedrock first (on-demand deployment / Custom Model Import); **SageMaker endpoints** when Bedrock can't serve the architecture. Weights are kept in S3 so a model can move to Bedrock later without retraining |
| Regions | **Cross-region on by default**: home region first, then other regions when the home region has no route. `allowed_regions` restricts it for data residency |
| Techniques | SFT (LoRA + full), distillation, DPO, RFT, continued pre-training (phased) |
| API style | Resource objects (`LLM`, `Dataset`, `FineTuningJob`, `Deployment`), idempotent by name |
| Surfaces | Python library, `skyhammer` CLI, YAML job specs |
| Infra | `skyhammer bootstrap` creates a CloudFormation stack (bucket, KMS key, service roles) in the home region **and** the other customization regions, so cross-region routes need no extra setup |
| Inference | OpenAI-compatible shim **and** access to the raw Bedrock response |
| Data in | OpenAI chat JSONL (`{"messages": [...]}`) and preference/RFT variants of it |
| Validation | Strict, with opt-in auto-fix (truncate, split validation set, drop bad rows) before any spend |
| Deployment | On-demand by default; provisioned throughput is opt-in |
| State | Stateless: AWS resource tags + JSON manifests in the bootstrap bucket |
| Graders | Python function → Lambda, existing Lambda ARN, or built-in LLM-as-judge |
| Eval | Local harness for iteration + Bedrock Model Evaluation jobs for formal reports |
| Enterprise | Customer-managed KMS everywhere; cost estimates/guardrails/budget tags |
| Phasing | Vertical slice first (see §9) |
| Tooling | uv, Python ≥3.11, pydantic v2, typer, boto3, ruff, mypy, pytest + moto |
| Region | us-east-1 for development and integration tests |
| Name & distribution | **Skyhammer**: *forge, tune and call models on AWS, without the heavy lifting.* Open source at github.com/randyridgley/skyhammer; PyPI `skyhammer` (and npm `skyhammer`) free as of 2026-10-29. No AWS marks in the name; README carries a non-affiliation notice |
| Serverless capability fallbacks | Bedrock rejects unsupported fields per model; the SDK learns them per process and degrades once with a warning: tool_choice→auto, json_schema→forced hidden tool (+ text-call recovery), reasoning_effort→Claude thinking budget / GPT-OSS effort / dropped, service tier→default. Prices: AWS Price List joined on normalized model names; only neutral tokens may differ, otherwise cost is unknown (never guessed) |
| Serverless inference | `LLM("<alias or Bedrock ID>")` calls **any Bedrock model** per token (no fine-tuning), serverless style; `us.` inference profiles by default, `global.` opt-in; SigV4 **and** Bedrock API keys; first: streaming, tools, structured outputs, reasoning, service tiers, cost per call, embeddings, rerank, batch (ROADMAP §2.5, M2) |

## 2. AWS landscape (verified against AWS docs, 2026-10-29)

The SDK's main job is hiding this matrix. It changes frequently, so it lives
in data (`catalog.py`), not in logic.

### 2.1 Bedrock-native customization (`CreateModelCustomizationJob`)

| Model | Bedrock model ID | SFT | Distill | RFT | Region | On-demand deploy |
|---|---|---|---|---|---|---|
| Nova Micro | `amazon.nova-micro-v1:0:128k` | ✓ | student | | us-east-1 | ✓ us-east-1 |
| Nova Lite | `amazon.nova-lite-v1:0:300k` | ✓ | student | | us-east-1 | ✓ us-east-1 |
| Nova Pro | `amazon.nova-pro-v1:0:300k` | ✓ | student | | us-east-1 | ✓ us-east-1 |
| Nova 2 Lite | `amazon.nova-2-lite-v1:0:256k` | ✓ | | ✓ | us-east-1 | ✓ us-east-1 |
| Llama 3.1/3.2/3.3 Instruct | `meta.llama3-*` | ✓ | | | **us-west-2** | Llama 3.3 70B only, us-west-2 |
| gpt-oss-20b, Qwen3 32B | `openai.gpt-oss-20b`, `qwen.qwen3-32b` | | | ✓ (OpenAI-compatible "mantle" API) | **us-west-2** | immediate |

Custom models are deployed with `CreateCustomModelDeployment`; the deployment
ARN is then used as `modelId` for Converse/InvokeModel.

### 2.2 SageMaker AI serverless model customization (open-weight)

Serverless jobs are plain `CreateTrainingJob` calls with a `ServerlessJobConfig`
(`BaseModelArn` = SageMaker Public Hub model, `JobType=FineTuning`,
`CustomizationTechnique` ∈ SFT/DPO/RLVR/RLAIF, `Peft=LORA` or omitted for full,
`EvaluatorArn` = reward Lambda, `AcceptEula`). This is the same API the SageMaker
Python SDK's `SFTTrainer`/`DPOTrainer` call (verified by reading SDK 3.27 source).
We call it directly: the `sagemaker` package installs **534 MB / 132 packages**
(pyarrow, scipy, mlflow, pandas, sklearn) versus 43 MB / 20 for our core deps,
and only needs boto3 ≥ 1.43. Covers SFT, DPO, RLVR
(verifiable rewards ≈ RFT with code graders) and RLAIF (LLM-judge rewards) for
Qwen2.5/3/3.5/3.6, Llama 3.x, DeepSeek R1 distills, GPT-OSS 20B/120B, Gemma 4,
Ministral 3 and Nemotron 3. Outputs register in a SageMaker Model Package Group
and can be deployed to SageMaker endpoints **or Bedrock Custom Model Import**.

### 2.3 Bedrock Custom Model Import (serving open-weight outputs)

* Regions: us-east-1, us-east-2, us-west-2, eu-central-1.
* Architectures: Llama 2/3/3.1/3.2/3.3/Mllama, Mistral, Mixtral, Flan-T5,
  GPTBigCode, Qwen2/2.5/2-VL/2.5-VL/Qwen3 (`Qwen3ForCausalLM`,
  `Qwen3MoeForCausalLM`), GPT-OSS (us-east-1 only).
* HF safetensors + config + tokenizer files; ≤200 GB text, ≤100 GB multimodal;
  context <128K; **transformers 4.51.3**.
* Invoked via `InvokeModel` (no batch inference). Models imported after
  2025-11-11 accept **OpenAI ChatCompletion bodies** (chat template taken from
  `tokenizer_config.json`). Converse is unsupported for Qwen2.5/Qwen3/GPT-OSS
  imports, so the SDK always uses OpenAI chat over InvokeModel for imported models.
* Idle imported models are unloaded; the next call raises `ModelNotReadyException`
  while the model is restored. The chat shim retries with backoff (default 10 min).

### 2.4 Consequences for the design

1. **The usable open-weight catalog in us-east-1 is the intersection** of
   "SageMaker serverless can train it" and "Custom Model Import can serve it":
   Llama 3.x, Qwen2.5, Qwen3 (dense + MoE), DeepSeek R1 distills (Qwen/Llama
   archs), GPT-OSS. Gemma 4, Nemotron 3, Qwen3.5+/3.6 and Ministral 3 are
   trainable but **not importable** (as of today). The catalog gives them no
   `bedrock_serving_regions`, and the router serves them from a **SageMaker
   endpoint**. When Bedrock adds the architecture, it's a catalog change, and
   `ft.deploy(serving="bedrock")` imports the stored weights (no retraining).
2. **In us-east-1, every open-weight model trains on SageMaker**, because
   Bedrock-native open-weight customization is us-west-2-only. Importable
   architectures are still **served by Bedrock** in us-east-1 (Custom Model
   Import), so SageMaker hosting is only needed for non-importable ones. The
   router is region-aware, so the same code takes the native path in us-west-2
   when that path supports on-demand serving.
3. LoRA outputs import directly: recipes default to `merge_weights=true`, which
   writes merged weights to `checkpoints/hf_merged/`. The SDK sets output
   `CompressionType=NONE` (Bedrock can't read tarballs) and rejects
   `merge_weights=false` when the model is served on Bedrock.
6. **SageMaker Public Hub metadata is the source of truth** for open-weight
   customization (`hub.py`). Per model it has recipes (technique × LoRA/full ×
   sequence length), hyperparameter specs with types/enums/bounds (validated
   locally before spend), gating (`ModelAccess`), and hosting configs (LMI/vLLM
   container image, instance type, GPU/memory). Recipe keys differ from the
   SDK's: `max_epochs`, `global_batch_size`, `lr_warmup_steps_ratio`, … and the
   backend maps them. The recipe's sequence length is chosen from the dataset's
   longest sample.
4. **Continued pre-training** is no longer listed for Bedrock-native
   customization and isn't in the serverless recipe table. It needs SageMaker
   training jobs / HyperPod recipes (Nova recipes support CPT). It is the most
   expensive technique to support, so it is the last milestone.
5. RFT on Nova 2 Lite (native) and RLVR/RLAIF on open-weight (SageMaker) have
   different grader contracts. The SDK's `Grader` abstraction compiles to
   whichever contract the backend needs.

## 3. Architecture

```
┌──────────────────────── user-facing ─────────────────────────┐
│  LLM   Dataset   FineTuningJob   Deployment   Grader   Eval  │   skyhammer/*.py
│  skyhammer CLI (typer)            YAML JobSpec (pydantic)        │   cli/, spec.py
├──────────────────────── core ────────────────────────────────┤
│  Catalog + Router  →  picks Backend for (model, technique,   │   catalog.py, router.py
│                       region)                                │
│  Formats/Validation   Cost estimator   Tags/Manifests        │   formats/, validation.py, cost.py, tags.py
│  Inference shim (OpenAI ⇄ Converse / InvokeModel+template)   │   inference.py
├──────────────────────── backends ────────────────────────────┤
│  BedrockNativeBackend    SageMakerServerlessBackend          │   backends/
│  (CreateModelCustomizationJob,   (CreateTrainingJob +        │
│   CreateCustomModelDeployment)    ServerlessJobConfig →      │
│                                   import or SM endpoint)     │
├──────────────────────── AWS ─────────────────────────────────┤
│  Session (boto3) + Bootstrap stack (S3, KMS, IAM roles)      │   session.py, bootstrap/
└──────────────────────────────────────────────────────────────┘
```

### 3.1 Backend protocol

```python
class Backend(Protocol):
    name: str

    def submit(self, req: TrainingRequest) -> JobHandle: ...
    def describe(self, handle: JobHandle) -> JobStatus: ...
    def metrics(self, handle: JobHandle) -> list[MetricPoint]: ...
    def cancel(self, handle: JobHandle) -> None: ...
    def finalize(self, handle: JobHandle) -> ModelRef: ...  # e.g. import into Bedrock
    def deploy(self, model: ModelRef, opts: DeployOptions) -> DeploymentRef: ...
```

`TrainingRequest` is backend-agnostic (technique, base model entry,
dataset refs, hyperparameters, grader, output name, KMS key, tags). Each
backend maps it to its own API and **rejects unsupported knobs explicitly**
(for example, `lora_rank` on Bedrock-native Nova raises an error instead of
being silently dropped).

For SageMaker, `FineTuningJob` is a state machine covering
`TRAINING → MERGING → IMPORTING → READY` (each phase is an AWS job), so the
user sees one job. A phase that fails can be retried without retraining.

### 3.2 Routing

`router.resolve(model, technique, region) -> Route(capability, region, serving)`

Preference order, so the common case needs no configuration:

1. **Serving**: Bedrock (serverless, pay per 5-minute CMU window or per token)
   before SageMaker endpoints (billed per instance-hour while running).
2. **Region**: home region first, then other regions if `cross_region` (default
   on). Cross-region routes are logged and shown in `skyhammer apply --plan`.
3. **Backend**: Bedrock-native (no merge/import) before SageMaker serverless.
   Native routes that would need provisioned throughput to serve (e.g. Llama
   3.1 8B) lose to SageMaker + import.

Gates enforced while routing and planning (before any spend):
* Gated licenses (Llama, Gemma) need `accept_eula=True` on SageMaker.
* Imported architectures without Converse (Qwen3) use InvokeModel with the
  chat template applied client-side; `supports_converse` is catalog data.
* Unknown hyperparameters are rejected; `hyperparameters_extra` passes
  backend-native keys through.

Overrides for platform users on every `create_*_job`: `backend=`,
`serving=`, `training_type=`; settings `cross_region` / `allowed_regions`.
* `skyhammer models list --technique rft` shows what's possible in the current
  region, and why other paths aren't.

### 3.3 Idempotency and state (no database)

* Every resource name is user-provided or derived (`support-qwen-v1`).
  `get-or-create` semantics: re-running a script reattaches to the existing
  job or model instead of launching a duplicate.
* Datasets are content-addressed: `s3://<bootstrap-bucket>/datasets/<name>/<sha256>/…`.
  Re-uploading identical data is a no-op. Uploading different data under the
  same name creates a new version.
* Tags on every AWS resource: `skyhammer:name`, `skyhammer:version`,
  `skyhammer:dataset-sha`, `skyhammer:base-model`, `skyhammer:technique`, `skyhammer:job-id`,
  plus user budget tags (`cost-center`, `project`).
* Lineage manifest `s3://<bucket>/manifests/models/<name>.json` holds full
  hyperparameters, dataset URIs, backend artifacts, eval results. Tags hold
  only short values because of tag-length limits.

### 3.4 Inference shim

`ft.chat.completions.create(...)` returns an OpenAI-shaped `ChatCompletion`
with `.raw` set to the original Bedrock response.

* Nova / native customs and base models (via `us.` inference profiles) → Converse.
* Imported models → `InvokeModel` with an OpenAI ChatCompletion body; the
  response is already OpenAI-shaped. Streaming via `InvokeModelWithResponseStream`.
* Cold starts (`ModelNotReadyException`, SageMaker scale-from-zero) are retried
  transparently with a one-time "warming up" log.
* Platform escape hatch: `ft.converse(...)`, `ft.invoke_model(body)`, `ft.model_id`.

### 3.4a SageMaker endpoints that feel serverless

Models Bedrock can't serve run on a SageMaker endpoint that hosts one
**inference component** (IC). Inference components are required for scale-to-zero.
The user just calls `ft.deploy()` and `ft.chat...`; the SDK builds everything
(`serving.py`):

* **Default `Scaling(min_copies=0, max_copies=1)`**: scale to zero when idle,
  one copy when in use. The endpoint is created with 1 instance so a broken
  model fails at deploy time, then scales in after `scale_in_cooldown_s` (300 s).
* **Autoscaling**: target tracking on `SageMakerInferenceComponentInvocationsPerCopy`
  (`target_invocations_per_copy`, default 5). Scaling out from zero uses a step policy
  plus a CloudWatch alarm on `NoCapacityInvocationFailures` (fires on the
  first failed request).
* **User rules**: `ft.deploy(scaling=Scaling(min_copies=1, max_copies=4,
  target_invocations_per_copy=10))` to keep it warm or scale wider;
  `custom_policies=[...]` takes raw Application Auto Scaling policies.
  The same fields are available as `deploy.scaling` in YAML specs.
* **Cold starts**: while at zero, requests fail for several minutes until an
  instance is up. The chat shim retries with backoff and logs "warming up"
  (timeout configurable); `ft.warm()` pre-scales before a burst.
* **Sizing**: smallest G6e/P5 instance whose GPU memory fits bf16 weights
  +30% (catalog `params_b`); `instance_type=` overrides.
* Bedrock-served models reject `scaling`/`instance_type`, since they are
  already serverless.

### 3.4b MLflow (optional, thin)

MLflow is an optional output, not a dependency or a source of truth.
Tags/manifests stay authoritative, so nothing breaks without it.

* **M1, no new dependencies**: set `mlflow_arn` (a SageMaker MLflow App or
  tracking server). SageMaker serverless jobs then get `MlflowConfig` and
  log loss curves natively (experiment = output model name, run = job name).
  The bootstrap training role has `sagemaker-mlflow:*` on the account's apps.
* **M2, optional extra `skyhammer[mlflow]`**: mirror Bedrock-native
  job metrics (from the job's S3 output), SDK params/lineage and `skyhammer eval`
  scores into the same experiment, so Nova and open-weight runs compare side
  by side.
* **Out of scope**: MLflow Model Registry as the system of record, and
  deploying from MLflow.
* The SDK never creates an MLflow App on its own, because it's a billed resource.
  `skyhammer mlflow setup` (M2) creates one explicitly.

### 3.5 Graders

```python
@grader(timeout_s=10)  # packaged + deployed as a Lambda
def exact_match(sample: dict, output: str) -> float: ...


Grader.from_lambda("arn:aws:lambda:...")  # bring your own
Grader.llm_judge(model="amazon.nova-pro-v1:0", rubric="...")  # built-in
```

* `@grader` functions are packaged with their source and pinned deps, then
  deployed by the SDK as `skyhammer-grader-<name>-<hash>` using the bootstrap
  grader role. They can be tested locally: `exact_match.test(dataset[:20], outputs)`.
* Each backend adapts the grader to its Lambda event/response contract
  (Nova RFT vs SageMaker RLVR). LLM-judge maps to Bedrock's
  model-as-a-judge grader or to SageMaker RLAIF.

### 3.6 Datasets and validation

Input is OpenAI JSONL in three shapes:

* SFT / distillation / CPT: `{"messages": [...]}` (CPT: `{"text": "..."}`)
* DPO: `{"input": {"messages": [...]}, "preferred_output": [...], "non_preferred_output": [...]}` (OpenAI preference shape)
* RFT: `{"messages": [...], <any reference fields for the grader>}`

Validation is model-aware. It uses limits from the catalog entry: minimum and
maximum sample counts, max tokens per sample, allowed roles, first/last role
rules, system prompt placement, and empty content. It returns a
`ValidationReport` with errors, warnings and suggested `fixes`.
`Dataset.from_file(..., auto_fix=True)` applies only the fixes it lists, and
the report records what changed.

Converters (pure functions, heavily unit-tested):
`openai → bedrock-conversation (Nova "bedrock-conversation-2024")`,
`openai → sagemaker-serverless`, `openai-preference → sagemaker-dpo`.

### 3.7 Cost guardrails

* `job.estimate()` (also called automatically before submit): tokens × epochs ×
  catalog price, plus storage/month and deployment notes. Token counts come
  from a heuristic tokenizer (or the HF tokenizer for open-weight models when
  `skyhammer[tokenizers]` is installed).
* `max_cost_usd=` on every `create_*_job`. Submission fails if the estimate
  exceeds it. The CLI prompts for confirmation unless run with `--yes`.
* `skyhammer deployments prune --idle 7d` deletes on-demand deployments with no
  invocations (CloudWatch) for N days. It requires `--yes`.
* Budget tags flow from config to every resource.

Prices come from the AWS Price List API (`pricing.py`, cached 7 days), keyed by
usage type. Bedrock uses `USE1-NovaMicro-Customization-Training` ($/1K tokens).
SageMaker uses `USE1-ServerlessTraining:FineTuning-SFT-LORA-<hub-model-id>`
($/1M tokens; RLVR/RLAIF are billed **per hour**, $80 LoRA / $100 full, so their
duration can't be estimated up front). Model storage is $1.95/month for both
Bedrock custom and imported models. If the lookup fails (offline, no
credentials), the estimate shows "unknown" and warns instead of blocking.

### 3.8 Bootstrap (`skyhammer bootstrap`)

Deploys CloudFormation stack `skyhammer-bootstrap` (template shipped in the package):

* S3 bucket (versioned, SSE-KMS, TLS-only, block public access)
* Customer-managed KMS key (or `--kms-key-arn` to use an existing one)
* `SkyhammerBedrockCustomizationRole` (trusted by `bedrock.amazonaws.com`, scoped to the bucket and key)
* `SkyhammerSageMakerTrainingRole` (trusted by `sagemaker.amazonaws.com`)
* `SkyhammerModelImportRole` (Bedrock import)
* `SkyhammerGraderLambdaRole`
* Stack outputs are discovered at runtime, so users never paste ARNs.
  Enterprises can skip bootstrap and set ARNs in config instead.

### 3.9 Configuration

Precedence: explicit kwargs → env (`SKYHAMMER_*`) → `./skyhammer.yaml` → `~/.config/skyhammer/config.yaml` → bootstrap stack outputs.

```yaml
profile: ml-dev
region: us-east-1
bucket: skyhammer-123456789012-us-east-1   # optional; discovered from bootstrap
kms_key_arn: arn:aws:kms:...           # optional
tags: {cost-center: "1234", project: "support-bot"}
guardrails: {max_cost_usd: 500, confirm: true}
cross_region: true                     # default; home region is always tried first
allowed_regions: [us-east-1, us-west-2]  # optional data-residency restriction
```

### 3.10 YAML job specs

```yaml
apiVersion: skyhammer/v1
kind: FineTuningJob
name: support-qwen-v1
base_model: qwen3-8b
technique: sft
dataset: {path: data/train.jsonl, validation_split: 0.05}
hyperparameters: {epochs: 3, learning_rate: 1e-4, lora_rank: 16}
deploy:
  type: on_demand
  scaling: {min_copies: 0, max_copies: 2, target_invocations_per_copy: 5}  # SageMaker-served only
eval:
  dataset: data/heldout.jsonl
  graders: [{lambda_arn: arn:aws:lambda:...}]
guardrails: {max_cost_usd: 200}
```

`skyhammer apply job.yaml` is idempotent (plan → confirm → apply), similar to `terraform apply`.

### 3.11 CLI sketch

```
skyhammer bootstrap [--kms-key-arn ARN] [--destroy]
skyhammer models list [--technique sft|dpo|rft|distill|cpt] [--region R]
skyhammer datasets validate FILE --model ALIAS [--fix]
skyhammer datasets upload FILE --name NAME
skyhammer jobs create --model ALIAS --technique sft --dataset NAME [...]
skyhammer jobs get|list|cancel|logs|wait NAME
skyhammer apply SPEC.yaml [--yes]
skyhammer deploy MODEL [--provisioned --units N]
skyhammer deployments list|delete|prune
skyhammer chat MODEL "prompt"
skyhammer eval run --model MODEL --dataset FILE [--grader ...]
```

## 4. Package layout

```
src/skyhammer/
  __init__.py          public API re-exports
  config.py            Settings (pydantic) + precedence resolution
  session.py           boto3 session/client factory, bootstrap discovery
  catalog.py           ModelEntry data + region/technique capability matrix
  router.py            (alias, technique, region) -> Route
  dataset.py           Dataset resource
  validation.py        model-aware validation + auto-fix
  formats/             pure converters between dataset schemas
  llm.py               LLM resource (create_*_job, deploy, chat)
  jobs.py              FineTuningJob + TrainingRequest/JobStatus models
  deployment.py        Deployment resource
  inference.py         OpenAI-compatible shim
  graders.py           @grader, Lambda, LLM-judge
  evaluation.py        local harness + Bedrock eval jobs
  cost.py              estimator + guardrails
  tags.py              tag schema + manifest IO
  spec.py              YAML JobSpec models
  errors.py            exception hierarchy
  backends/
    base.py            Backend protocol
    bedrock.py         BedrockNativeBackend
    sagemaker.py       SageMakerServerlessBackend
  bootstrap/
    template.yaml      CloudFormation stack
    __init__.py        deploy/destroy/discover
  cli/main.py          typer app
tests/
  unit/                pure logic (formats, validation, router, spec, tags, cost)
  integration/         marked `aws`, us-east-1, opt-in via SKYHAMMER_INTEGRATION=1
```

## 5. Error model

`SkyhammerError` → `ConfigError`, `BootstrapMissingError`, `UnsupportedRouteError`
(includes the reason and alternatives such as "available in us-west-2" or
"trainable but not importable"), `DatasetValidationError` (carries the
report), `CostLimitExceededError`, `JobFailedError` (carries the backend
failure message + log link), `AwsError` (wraps botocore errors with context).

## 6. Testing strategy

* Unit tests run offline and cover converters, validation, routing, spec
  parsing, cost math and tag encoding.
* Backend tests use botocore `Stubber` to check exact request shapes, plus
  moto for S3/IAM/CloudFormation.
* Integration tests run in us-east-1 behind `SKYHAMMER_INTEGRATION=1`: a tiny Nova
  Micro SFT and a Qwen3-0.6B LoRA SFT → import → invoke → delete. They are
  also nightly CI smoke tests.
* A golden test checks the catalog against `ListFoundationModels` /
  `ListCustomizationModels` to detect AWS changes.

## 7. Security

* No credentials handled by the SDK; standard boto3 credential chain.
* All S3 writes use SSE-KMS with the configured key; customization and
  import jobs pass `customModelKmsKeyId` / output KMS keys.
* Bootstrap IAM roles are least-privilege and scoped to the bucket prefix + key.
* Graders run as Lambdas under a dedicated role with no data-bucket write access.

## 8. Open questions

Resolved 2026-10-29: cross-region on by default (§3.2); SageMaker endpoint
serving for non-importable models; open-weight distillation in scope (M2);
call SageMaker APIs directly (§2.2); publish to PyPI eventually.

Also resolved: SageMaker endpoints scale to zero by default with
user-tunable autoscaling (§3.4a); MLflow is an optional thin integration
(§3.4b); recipe hyperparameter keys are read from `DescribeHubContent` in M1.

Still open:
1. ~~Serving container~~: resolved. Each Hub recipe carries a `HostingConfig`
   (DJL LMI vLLM image, env, instance type, compute requirements); M2 uses it.
2. **MLflow App pricing**: not confirmed from docs. Check before recommending
   one per team.
3. **SageMaker Model Package Groups** (`ModelPackageConfig`): skipped for now.
   Revisit if approval workflows are needed.

## 9. Milestones

* **M0 (this scaffold)**: package layout, config, catalog/router, converters,
  validation, spec parsing, CLI skeleton, bootstrap template, unit tests.
* **M1 (vertical slice)**: Nova Micro/Lite SFT (native) + Qwen3/Llama LoRA SFT
  (SageMaker → merge → import), `MlflowConfig` pass-through, recipe
  hyperparameter mapping, on-demand deploy, OpenAI shim, cost estimate,
  `skyhammer apply`, integration tests in us-east-1.
* **M2**: Distillation (Nova native; open-weight via teacher batch + SFT),
  full fine-tuning, **SageMaker endpoint serving** (orchestrate the `serving.py`
  builders, cold-start-aware chat shim over `sagemaker-runtime`, `ft.warm()`,
  idle pruning), SageMaker → Bedrock re-import, MLflow metric mirroring,
  `skyhammer eval` local harness, Bedrock eval jobs.
* **M3**: DPO (SageMaker), RFT (Nova 2 Lite native) + RLVR/RLAIF (SageMaker),
  `@grader` → Lambda, LLM-judge.
* **M4**: Continued pre-training (SageMaker training jobs / Nova recipes),
  provisioned throughput, async API, PyPI release.
