"""``skyhammer`` command-line interface."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from skyhammer import bootstrap as bootstrap_mod
from skyhammer import registry
from skyhammer.backends import get_backend
from skyhammer.batch import BatchJob
from skyhammer.catalog import (
    CATALOG,
    BackendKind,
    ModelEntry,
    ServingTarget,
    Technique,
    TrainingType,
    get_model,
)
from skyhammer.config import configure
from skyhammer.dataset import Dataset
from skyhammer.errors import BootstrapMissingError, SkyhammerError
from skyhammer.evaluation import evaluate
from skyhammer.graders import Grader, GraderKind
from skyhammer.jobs import FineTuningJob
from skyhammer.llm import LLM
from skyhammer.router import resolve
from skyhammer.serverless import ModelDirectory, Routing
from skyhammer.session import Session
from skyhammer.spec import JobSpec, load_spec
from skyhammer.validation import parse_jsonl

app = typer.Typer(
    help="Skyhammer: forge, tune and call models on AWS, without the heavy lifting.",
    no_args_is_help=True,
)
models_app = typer.Typer(help="Model catalog and fine-tuned models.", no_args_is_help=True)
datasets_app = typer.Typer(help="Validate and upload datasets.", no_args_is_help=True)
jobs_app = typer.Typer(help="Follow and manage fine-tuning jobs.", no_args_is_help=True)
batch_app = typer.Typer(help="Discounted asynchronous batch inference.", no_args_is_help=True)
auth_app = typer.Typer(help="Bedrock API keys.", no_args_is_help=True)
eval_app = typer.Typer(help="Score models on a dataset with graders.", no_args_is_help=True)
deployments_app = typer.Typer(help="Skyhammer-created endpoints/deployments.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(datasets_app, name="datasets")
app.add_typer(jobs_app, name="jobs")
app.add_typer(batch_app, name="batch")
app.add_typer(auth_app, name="auth")
app.add_typer(deployments_app, name="deployments")
app.add_typer(eval_app, name="eval")

console = Console()
err = Console(stderr=True)

Region = Annotated[str | None, typer.Option("--region", help="AWS region (default from config)")]
Profile = Annotated[str | None, typer.Option("--profile", help="AWS profile")]


@app.callback()
def _global(region: Region = None, profile: Profile = None) -> None:
    configure(region=region, profile=profile)


@app.command()
def bootstrap(
    kms_key_arn: Annotated[
        str | None, typer.Option(help="Use an existing KMS key (home region only)")
    ] = None,
    home_only: Annotated[bool, typer.Option(help="Skip the other customization regions")] = False,
    destroy: Annotated[
        bool, typer.Option(help="Delete the stacks (buckets/keys are retained)")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Create the bucket, KMS key and service roles in the home region and, unless
    --home-only, the other customization regions (so cross-region routes just work)."""
    settings = Session().settings
    regions = [settings.region] if home_only else bootstrap_mod.default_regions(settings)
    if destroy:
        if not yes:
            typer.confirm(
                f"Delete stack {settings.bootstrap_stack} in {', '.join(regions)}?", abort=True
            )
        for region in regions:
            bootstrap_mod.destroy(bootstrap_mod.session_for(settings, region))
        console.print("Stacks deleted (buckets and KMS keys retained).")
        return
    for region in regions:
        with console.status(f"Deploying {settings.bootstrap_stack} in {region}..."):
            outputs = bootstrap_mod.deploy(
                bootstrap_mod.session_for(settings, region),
                kms_key_arn=kms_key_arn if region == settings.region else None,
            )
        console.print(f"[bold underline]{region}[/]")
        for k, v in outputs.items():
            console.print(f"  [bold]{k}[/]: {v}")


def _serving_summary(entry: ModelEntry, backend: BackendKind) -> str:
    parts = []
    if entry.bedrock_serving_regions:
        parts.append("bedrock: " + ", ".join(sorted(entry.bedrock_serving_regions)))
    if backend is BackendKind.SAGEMAKER:
        parts.append("sagemaker endpoint")
    return "; ".join(parts)


@models_app.command("list")
def models_list(
    technique: Annotated[Technique | None, typer.Option(help="Filter by technique")] = None,
    serverless: Annotated[
        bool, typer.Option("--serverless", help="All Bedrock models you can call per token")
    ] = False,
    kind: Annotated[
        str | None, typer.Option(help="chat | embedding | rerank | image | speech")
    ] = None,
    provider: Annotated[str | None, typer.Option(help="e.g. anthropic, meta, qwen")] = None,
    routing: Annotated[str | None, typer.Option(help="geo | global | region")] = None,
) -> None:
    """Fine-tunable models (default) or every serverless Bedrock model (--serverless)."""
    if serverless:
        session = Session()
        route = (
            Routing(routing)
            if routing
            else (Routing.GEO if session.settings.cross_region else Routing.REGION)
        )
        table = Table("alias", "model id", "kind", "routes", "invokes as", "stream")
        for m in ModelDirectory.for_session(session).list(kind=kind, provider=provider):
            try:
                target = m.invocation_id(route, session.region)
            except SkyhammerError:
                target = "[red]unavailable[/]"
            table.add_row(
                m.alias or "-",
                m.model_id + (" [dim](legacy)[/]" if m.legacy else ""),
                m.kind,
                m.describe_routes(),
                target,
                "yes" if m.streaming else "",
            )
        console.print(table)
        return
    table = Table("alias", "model", "technique", "train on", "regions", "serve")
    for alias, entry in sorted(CATALOG.items()):
        for cap in entry.capabilities:
            if technique and cap.technique is not technique:
                continue
            table.add_row(
                alias,
                entry.display_name,
                cap.technique.value,
                cap.backend.value,
                ", ".join(sorted(cap.regions)),
                _serving_summary(entry, cap.backend),
            )
    console.print(table)


@datasets_app.command("list")
def datasets_list() -> None:
    """Datasets uploaded to the bootstrap bucket (each content version separately)."""
    table = Table("name", "sha256", "formats", "size", "uploaded")
    for d in Dataset.list(Session()):
        table.add_row(
            d["name"],
            d["sha256"][:12],
            ", ".join(sorted(d["formats"])),
            f"{d['bytes'] / 1024:.1f} KB",
            d["uploaded"].strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


@datasets_app.command("get")
def datasets_get(name: str) -> None:
    """A dataset's versions and the models trained on it (lineage)."""
    info = Dataset.lineage(name, Session())
    versions = Table("sha256", "formats", "size", "uploaded", title=f"{name}: versions")
    for d in info["versions"]:
        versions.add_row(
            d["sha256"][:12],
            ", ".join(sorted(d["formats"])),
            f"{d['bytes'] / 1024:.1f} KB",
            d["uploaded"].strftime("%Y-%m-%d %H:%M"),
        )
    console.print(versions)
    jobs = Table("model", "base", "technique", "rows", "data sha256", "teacher", title="trained")
    for j in info["jobs"]:
        jobs.add_row(
            j["model"],
            j["base_model"],
            j["technique"],
            str(j["rows"] or ""),
            (j["dataset_sha256"] or "")[:12],
            j["distilled_from"] or "",
        )
    console.print(jobs)


@datasets_app.command("validate")
def datasets_validate(
    path: Path,
    model: Annotated[str, typer.Option(help="Target model alias")],
    technique: Annotated[Technique, typer.Option()] = Technique.SFT,
) -> None:
    """Validate a JSONL dataset against the limits AWS enforces for this model (no upload)."""
    entry = get_model(model)
    settings = Session().settings
    route = resolve(
        entry,
        technique,
        settings.region,
        cross_region=settings.cross_region,
        allowed_regions=settings.allowed_regions,
    )
    limits = get_backend(route.backend, Session(), route.region).dataset_limits(
        entry, technique, TrainingType.LORA
    )
    report = Dataset.from_file(path).validate(entry, technique, limits=limits)
    console.print(
        f"[dim]limits: <= {limits.max_tokens_per_sample:,} tokens/sample, "
        f"{limits.min_samples}-{limits.max_samples:,} rows ({limits.source})[/]"
    )
    console.print(report.summary(limit=50))
    if not report.ok:
        if report.fixable:
            console.print(
                "[yellow]All errors are auto-fixable (pass auto_fix=True / "
                "dataset.auto_fix: true).[/]"
            )
        raise typer.Exit(1)


def _plan_from_spec(spec: JobSpec) -> tuple[LLM, Dataset, dict[str, Any]]:
    llm = LLM(spec.base_model)
    ds = Dataset.from_file(spec.dataset.path, name=spec.dataset.name or spec.name)
    kwargs: dict[str, Any] = {
        "output_model": spec.name,
        "validation_dataset": (
            Dataset.from_file(spec.dataset.validation_path)
            if spec.dataset.validation_path
            else None
        ),
        "validation_split": spec.dataset.validation_split,
        "auto_fix": spec.dataset.auto_fix,
        "backend": spec.backend,
        "serving": spec.serving,
        "accept_eula": spec.accept_eula,
        "training_type": spec.training_type,
        "teacher_model": spec.teacher_model,
        "grader": spec.grader.build() if spec.grader else None,
        "tags": spec.tags,
        **spec.hyperparameters,
    }
    return llm, ds, kwargs


def _eval_graders(spec: JobSpec) -> list[Grader]:
    if not spec.eval:
        return []
    graders = [g.build() for g in spec.eval.graders] or [Grader.builtin("exact_match")]
    if any(g.kind is GraderKind.LLM_JUDGE for g in graders):
        raise SkyhammerError("LLM-judge graders for evaluation are planned for M3")
    return graders


def _wait(job: FineTuningJob, poll_seconds: float = 60) -> None:
    """Block until done, printing each stage change and the loss as it trains (the job may
    outlive this process; re-running the command reattaches)."""
    job.wait(poll_seconds=poll_seconds, progress=True)


@app.command()
def apply(
    spec_path: Path,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
    plan_only: Annotated[bool, typer.Option("--plan", help="Show plan and exit")] = False,
    no_wait: Annotated[
        bool, typer.Option("--no-wait", help="Submit and return; re-run apply to continue")
    ] = False,
) -> None:
    """Train, deploy and evaluate the model described by a YAML spec. Idempotent: re-running
    reattaches to the job, reuses the deployment, and picks up where it left off."""
    spec = load_spec(spec_path)
    llm, ds, kwargs = _plan_from_spec(spec)
    request, estimate = llm.plan(spec.technique, ds, **kwargs)
    graders = _eval_graders(spec)
    if spec.deploy.type == "provisioned":
        raise SkyhammerError("Provisioned throughput is planned for M4; use type: on_demand")
    if spec.deploy.type != "none" and request.route.serving is ServingTarget.SAGEMAKER:
        console.print(
            "[yellow]Note:[/] this model is served from a SageMaker endpoint "
            "(billed per instance-hour; scales to zero when idle by default)."
        )

    console.print(
        f"[bold]{request.job_name}[/]: {spec.technique.value} {spec.base_model}: "
        f"{request.route.describe()}"
    )
    console.print(f"  base model id: {request.route.capability.backend_model_id}")
    console.print(f"  dataset: {ds!r}")
    lim = request.dataset.report.limits if request.dataset.report else None
    if lim:
        console.print(
            f"  limits: {lim.min_samples:,}-{lim.max_samples:,} rows, "
            f"<= {lim.max_tokens_per_sample:,} tokens/sample ({lim.source})"
        )
    console.print(f"  {estimate}")
    for note in estimate.notes[1:]:
        console.print(f"    [dim]{note}[/]")
    console.print(f"  deploy: {spec.deploy.type}")
    if spec.eval:
        console.print(
            f"  eval: {spec.eval.dataset.name} with " + ", ".join(g.name for g in graders)
        )
    if plan_only:
        return
    if not yes and llm.session.settings.guardrails.confirm:
        typer.confirm("Apply?", abort=True)

    job = llm.create_job(spec.technique, ds, max_cost_usd=spec.guardrails.max_cost_usd, **kwargs)
    console.print(f"Job {job.name} ({job.record.backend}, {job.record.region})")
    if no_wait:
        console.print(
            f"Submitted. Re-run `skyhammer apply {spec_path}` or "
            f"`skyhammer jobs wait {spec.name}` to continue."
        )
        return
    _wait(job)
    ft = job.output_llm
    if spec.deploy.type != "none":
        with console.status("Deploying..."):
            ft.deploy(
                name=spec.deploy.name,
                scaling=spec.deploy.scaling,
                instance_type=spec.deploy.instance_type,
            )
        console.print(f"Deployed: {ft.model_id}")
    if spec.eval:
        _run_eval(ft, spec.eval.dataset, graders)
    console.print(f'[green]Done.[/] Try: skyhammer chat {spec.name} "Hello"')


def _run_eval(ft: LLM, path: Path, graders: list[Grader]) -> None:
    with path.open(encoding="utf-8") as f:
        records = [r for _, r in parse_jsonl(f) if isinstance(r, dict)]
    with console.status(f"Evaluating on {len(records)} records..."):
        result = evaluate(ft, records, graders)
    table = Table("grader", "mean score", "n")
    for name, mean in result.means.items():
        table.add_row(name, f"{mean:.3f}", str(len(result.scores[name])))
    console.print(table)
    ft._record_in_manifest(eval={"dataset": path.name, "means": result.means, "at": time.time()})


# ---- jobs ----


@jobs_app.command("list")
def jobs_list() -> None:
    """Latest job for each fine-tuned model."""
    table = Table("model", "job", "technique", "train on", "region", "status")
    for job in FineTuningJob.list():
        st = job.status()
        table.add_row(
            job.record.output_model,
            job.name,
            job.record.technique,
            job.record.backend,
            job.record.region,
            st.state.value,
        )
    console.print(table)


@jobs_app.command("get")
def jobs_get(name: str) -> None:
    """Details and live status for a model's latest job (or a job by exact name)."""
    job = FineTuningJob.get(name)
    st = job.status()
    r = job.record
    rows = [
        ("job", r.job_name),
        ("model", r.output_model),
        ("base", r.base_model),
        ("technique", f"{r.technique} ({r.training_type})"),
        ("route", f"train on {r.backend} in {r.region}, serve on {r.serving}"),
        ("dataset", f"{r.dataset.get('name')} ({r.dataset.get('rows')} rows)"),
        ("estimate", r.estimate or "-"),
        ("submitted", time.strftime("%Y-%m-%d %H:%M", time.localtime(r.created_at))),
        ("status", st.state.value + (f": {st.message}" if st.message else "")),
    ]
    if st.output_model_arn:
        rows.append(("model arn", st.output_model_arn))
    if st.log_url:
        rows.append(("logs", st.log_url))
    for k, v in rows:
        console.print(f"[bold]{k:>10}[/]  {v}")


@jobs_app.command("wait")
def jobs_wait(
    name: str, poll: Annotated[float, typer.Option(help="Seconds between checks")] = 60
) -> None:
    """Block until the job (including any Bedrock import) finishes."""
    job = FineTuningJob.get(name)
    console.print(f"Waiting on {job.name}")
    _wait(job, poll)
    console.print("[green]Completed.[/]")


@jobs_app.command("logs")
def jobs_logs(name: str, limit: Annotated[int, typer.Option(help="Lines to show")] = 50) -> None:
    """Training progress (SageMaker log stream or Bedrock step metrics)."""
    for line in FineTuningJob.get(name).logs(limit):
        console.print(line, highlight=False, markup=False)


@jobs_app.command("metrics")
def jobs_metrics(name: str, last: Annotated[int, typer.Option(help="Rows to show")] = 20) -> None:
    """Per-step training metrics (loss, learning rate...) and the loss curve."""
    from skyhammer.progress import loss_series, sparkline

    rows = FineTuningJob.get(name).metrics()
    if not rows:
        console.print("No metrics yet.")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r if k != "step"))
    table = Table("step", *keys)
    for r in rows[-last:]:
        table.add_row(f"{r['step']:g}", *(f"{r[k]:.4g}" if k in r else "" for k in keys))
    console.print(table)
    losses = loss_series(rows)
    if losses:
        console.print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}  {sparkline(losses)}")


@jobs_app.command("cancel")
def jobs_cancel(name: str, yes: Annotated[bool, typer.Option("--yes", "-y")] = False) -> None:
    """Stop a running training job."""
    job = FineTuningJob.get(name)
    if not yes:
        typer.confirm(f"Stop {job.name}?", abort=True)
    job.cancel()
    console.print(f"Stopping {job.name}.")


# ---- models / serving ----


@models_app.command("delete")
def models_delete(name: str, yes: Annotated[bool, typer.Option("--yes", "-y")] = False) -> None:
    """Delete a fine-tuned model and its deployment (stops storage/serving charges)."""
    ft = LLM.get(name)
    if not yes:
        typer.confirm(
            f"Delete {name} ({ft.model_ref.model_arn if ft.model_ref else ''})?", abort=True
        )
    ft.delete()
    console.print(f"Deleted {name}.")


@app.command()
def deploy(
    name: str,
    min_copies: Annotated[int | None, typer.Option(help="SageMaker-served only")] = None,
    max_copies: Annotated[int | None, typer.Option(help="SageMaker-served only")] = None,
    instance_type: Annotated[str | None, typer.Option(help="SageMaker-served only")] = None,
) -> None:
    """Make a fine-tuned model invocable (idempotent)."""
    scaling = None
    if min_copies is not None or max_copies is not None:
        scaling = {
            k: v
            for k, v in (("min_copies", min_copies), ("max_copies", max_copies))
            if v is not None
        }
    ft = LLM.get(name)
    with console.status(f"Deploying {name}..."):
        ft.deploy(scaling=scaling, instance_type=instance_type)
    console.print(f"Deployed: {ft.model_id}")


def _resolve_llm(name: str) -> LLM:
    """Your fine-tuned model by that name if one exists, else a Bedrock model alias/ID."""
    session = Session()
    try:
        if registry.load_model(session, name):
            return LLM.get(name, session=session)
    except BootstrapMissingError:
        pass  # no bootstrap: serverless calls still work
    return LLM(name, session=session)


@app.command()
def chat(
    name: str,
    prompt: str,
    system: Annotated[str | None, typer.Option(help="System prompt")] = None,
    max_tokens: Annotated[int, typer.Option()] = 512,
    temperature: Annotated[float | None, typer.Option()] = None,
    stream: Annotated[bool, typer.Option(help="Print tokens as they arrive")] = False,
    tier: Annotated[str | None, typer.Option(help="priority | standard | flex")] = None,
    routing: Annotated[str | None, typer.Option(help="geo | global | region")] = None,
    reasoning: Annotated[str | None, typer.Option(help="low | medium | high")] = None,
) -> None:
    """Prompt a fine-tuned model (by name) or any Bedrock model (alias or ID)."""
    llm = _resolve_llm(name)
    if routing:
        llm._routing = Routing(routing)
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    params: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "service_tier": tier,
        "reasoning_effort": reasoning,
    }
    params = {k: v for k, v in params.items() if v is not None}
    if stream:
        s = llm.chat.completions.create(messages=messages, stream=True, **params)
        for chunk in s:
            for ch in chunk.choices:
                if ch.delta.content:
                    console.print(ch.delta.content, end="", highlight=False, markup=False)
        console.print()
        resp = s.final()
    else:
        resp = llm.chat.completions.create(messages=messages, **params)
        msg = resp.choices[0].message
        if msg.reasoning_content:
            console.print(f"[dim]{msg.reasoning_content.strip()[:500]}[/]", highlight=False)
        console.print(msg.content or "", highlight=False, markup=False)
    u = resp.usage
    cost = f", ${u.cost_usd:.6f}" if u.cost_usd is not None else ""
    console.print(
        f"[dim]{llm.model_id}: {u.prompt_tokens} in / {u.completion_tokens} out"
        f"{cost}, finish={resp.choices[0].finish_reason}[/]"
    )


@app.command()
def embed(
    name: str,
    texts: list[str],
    dimensions: Annotated[int | None, typer.Option()] = None,
    input_type: Annotated[str, typer.Option(help="search_document | search_query")] = (
        "search_document"
    ),
) -> None:
    """Embed texts with any Bedrock embedding model (prints dimensions and a preview)."""
    resp = LLM(name).embeddings.create(input=texts, dimensions=dimensions, input_type=input_type)
    for e in resp.data:
        preview = ", ".join(f"{x:.4f}" for x in e.embedding[:4])
        console.print(f"[{e.index}] dim={len(e.embedding)} [{preview}, ...]")
    cost = f", ${resp.usage.cost_usd:.6f}" if resp.usage.cost_usd is not None else ""
    console.print(f"[dim]{resp.usage.total_tokens} tokens{cost}[/]")


@app.command("rerank")
def rerank_cmd(
    name: str,
    query: str,
    documents: list[str],
    top_n: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Rank documents by relevance to a query (e.g. rerank-v3-5)."""
    resp = LLM(name).rerank(query, documents, top_n=top_n, return_documents=True)
    for r in resp.results:
        console.print(f"{r.relevance_score:.3f}  [{r.index}] {r.document}", highlight=False)


@models_app.command("quotas")
def models_quotas(name: str) -> None:
    """Service Quotas for a model (tokens/requests per minute, batch limits)."""
    table = Table("quota", "value", "adjustable")
    for q in LLM(name).quotas():
        table.add_row(q["name"], f"{q['value']:,.0f}", "yes" if q["adjustable"] else "")
    console.print(table)


@auth_app.command("policy")
def auth_policy(
    account: Annotated[str | None, typer.Option(help="Account ID (default: current)")] = None,
    bootstrap: Annotated[
        bool, typer.Option(help="Include managing the skyhammer-bootstrap stack")
    ] = True,
) -> None:
    """Print a least-privilege IAM policy for Skyhammer users (attach to a role or group)."""
    import json

    from skyhammer.iam import user_policy

    account = account or Session().client("sts").get_caller_identity()["Account"]
    typer.echo(json.dumps(user_policy(account, bootstrap=bootstrap), indent=2))


@auth_app.command("api-key")
def auth_api_key(
    hours: Annotated[float, typer.Option(help="Validity, up to 12 hours")] = 12,
) -> None:
    """Print a short-term Bedrock API key from your AWS credentials (bearer token)."""
    from skyhammer.auth import generate_api_key

    typer.echo(generate_api_key(Session(), hours=hours))


@batch_app.command("submit")
def batch_submit(
    name: str,
    requests_path: Path,
    job_name: Annotated[str | None, typer.Option("--job-name")] = None,
    max_tokens: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Submit JSONL requests ({"custom_id"?, "messages": [...], ...}) as a batch job."""
    with requests_path.open(encoding="utf-8") as f:
        reqs = [r for _, r in parse_jsonl(f) if isinstance(r, dict)]
    defaults = {"max_tokens": max_tokens} if max_tokens else {}
    job = LLM(name).batch(reqs, name=job_name, **defaults)
    console.print(f"Batch {job.name} ({len(reqs):,} records) on {job.model_id}")


@batch_app.command("get")
def batch_get(job_name: str) -> None:
    """Status and record counts."""
    st = BatchJob.get(job_name).status()
    console.print(
        f"{job_name}: {st.status} {st.processed or 0}/{st.total or '?'} processed, "
        f"{st.succeeded or 0} ok, {st.failed or 0} failed"
        + (f" ({st.message})" if st.message else "")
    )


@batch_app.command("wait")
def batch_wait(job_name: str, poll: Annotated[float, typer.Option()] = 60) -> None:
    """Block until the batch job finishes."""
    BatchJob.get(job_name).wait(
        poll_seconds=poll,
        on_status=lambda st: console.print(
            f"  {time.strftime('%H:%M:%S')} {st.status} {st.processed or 0}/{st.total or '?'}"
        ),
    )


@batch_app.command("results")
def batch_results(job_name: str, output: Annotated[Path | None, typer.Option("-o")] = None) -> None:
    """Write results as JSONL ({custom_id, content, error, usage})."""
    import json

    lines = []
    for r in BatchJob.get(job_name).results():
        msg = r.completion.choices[0].message if r.completion else None
        lines.append(
            json.dumps(
                {
                    "custom_id": r.custom_id,
                    "content": msg.content if msg else None,
                    "error": r.error,
                    "usage": r.completion.usage.__dict__ if r.completion else None,
                }
            )
        )
    if output:
        output.write_text("\n".join(lines) + "\n")
        console.print(f"Wrote {len(lines):,} results to {output}")
    else:
        for line in lines:
            typer.echo(line)


@batch_app.command("cancel")
def batch_cancel(job_name: str, yes: Annotated[bool, typer.Option("--yes", "-y")] = False) -> None:
    if not yes:
        typer.confirm(f"Stop batch {job_name}?", abort=True)
    BatchJob.get(job_name).cancel()
    console.print(f"Stopping {job_name}.")


@deployments_app.command("list")
def deployments_list(
    idle_days: Annotated[float, typer.Option(help="Invocation window")] = 7,
) -> None:
    """Endpoints and on-demand deployments Skyhammer created, with recent invocations."""
    from skyhammer.deployments import list_deployments

    table = Table("kind", "name", "model", "status", f"invocations ({idle_days:g}d)")
    for d in list_deployments(Session(), idle_days=idle_days):
        table.add_row(d.kind, d.name, d.model, d.status, str(d.invocations))
    console.print(table)


@deployments_app.command("prune")
def deployments_prune(
    idle_days: Annotated[float, typer.Option(help="Delete if no invocations for this long")] = 7,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Delete idle Skyhammer deployments (never touches untagged resources)."""
    from skyhammer.deployments import delete_deployment, list_deployments

    session = Session()
    idle = [d for d in list_deployments(session, idle_days=idle_days) if d.invocations == 0]
    if not idle:
        console.print("Nothing idle.")
        return
    for d in idle:
        console.print(f"  {d.kind} {d.name} ({d.model})")
    if not yes:
        typer.confirm(f"Delete {len(idle)} idle deployment(s)?", abort=True)
    for d in idle:
        delete_deployment(session, d)
        console.print(f"Deleted {d.name}")


@app.command("cleanup")
def cleanup_cmd(
    older_than_hours: Annotated[float, typer.Option(help="Only resources older than this")] = 6,
    prefix: Annotated[
        list[str] | None, typer.Option(help="Name prefixes to sweep (repeatable)")
    ] = None,
    region: Annotated[
        list[str] | None, typer.Option(help="Regions (default: the bootstrap regions)")
    ] = None,
    graders: Annotated[
        bool, typer.Option("--graders", help="Also delete grader Lambdas if no RL job runs")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Delete without asking")] = False,
) -> None:
    """Delete leftovers from tests and smoke runs: names starting with skyhammer-it- or
    skyhammer-smoke- (or --prefix), tagged skyhammer:name, older than --older-than-hours."""
    from skyhammer.bootstrap import default_regions
    from skyhammer.sweeper import DEFAULT_PREFIXES, sweep

    session = Session()
    kwargs: dict[str, Any] = {
        "older_than_hours": older_than_hours,
        "prefixes": tuple(prefix) if prefix else DEFAULT_PREFIXES,
        "include_graders": graders,
    }
    regions = region or default_regions(session.settings)
    found = sweep(session, regions, dry_run=True, **kwargs)
    if not found:
        console.print("Nothing to clean up.")
        return
    table = Table("kind", "name", "region", "age (h)")
    for x in found:
        table.add_row(x.kind, x.name, x.region, f"{x.age_hours:.1f}")
    console.print(table)
    if not yes:
        typer.confirm(f"Delete {len(found)} resource(s)?", abort=True)
    sweep(session, regions, dry_run=False, **kwargs)
    console.print(f"Deleted {len(found)} resource(s).")


@app.command("metrics")
def metrics_cmd(
    model: str,
    hours: Annotated[float, typer.Option(help="Look-back window")] = 24,
) -> None:
    """Serving metrics from CloudWatch: invocations, latency, tokens, errors."""
    m = _resolve_llm(model).metrics(hours=hours)

    def ms(v: float | None) -> str:
        return f"{v:,.0f} ms" if v is not None else "-"

    table = Table("metric", f"last {m.hours:g}h", title=m.model_id)
    table.add_row("invocations", f"{m.invocations:,}")
    table.add_row(
        "latency avg / p50 / p99",
        " / ".join(map(ms, (m.latency_ms_avg, m.latency_ms_p50, m.latency_ms_p99))),
    )
    if m.input_tokens is not None:
        table.add_row("tokens in / out", f"{m.input_tokens:,} / {m.output_tokens or 0:,}")
    table.add_row("errors (client / server)", f"{m.client_errors} / {m.server_errors}")
    if m.throttles is not None:
        table.add_row("throttles", str(m.throttles))
    console.print(table)


@eval_app.command("compare")
def eval_compare(
    models: list[str],
    dataset: Annotated[Path, typer.Option(help="JSONL with messages (+ answer fields)")],
    grader: Annotated[
        list[str] | None,
        typer.Option(help="exact_match | contains | json_valid | module:function | Lambda ARN"),
    ] = None,
    max_tokens: Annotated[int, typer.Option()] = 512,
    limit: Annotated[int | None, typer.Option(help="Use only the first N records")] = None,
) -> None:
    """Run the same records and graders against several models (e.g. base vs fine-tuned)."""
    from skyhammer.graders import BUILTINS
    from skyhammer.spec import GraderSpec

    if grader is None:
        grader = ["contains"]
    graders = [
        GraderSpec(builtin=g).build()
        if g in BUILTINS
        else GraderSpec(lambda_arn=g).build()
        if g.startswith("arn:")
        else GraderSpec(python=g).build()
        for g in grader
    ]
    with dataset.open(encoding="utf-8") as f:
        records = [r for _, r in parse_jsonl(f) if isinstance(r, dict)][:limit]
    table = Table("model", *[g.name for g in graders], "cost")
    for name in models:
        llm = _resolve_llm(name)
        before = llm.session.spend.total_usd
        with console.status(f"{name}: {len(records)} records..."):
            result = evaluate(llm, records, graders, max_tokens=max_tokens)
        spent = llm.session.spend.total_usd - before
        table.add_row(
            name, *[f"{result.means.get(g.name, 0):.3f}" for g in graders], f"${spent:.5f}"
        )
    console.print(table)


@eval_app.command("run")
def eval_run(
    model: str,
    dataset: Annotated[Path, typer.Option()],
    grader: Annotated[list[str] | None, typer.Option()] = None,
    max_tokens: Annotated[int, typer.Option()] = 512,
    limit: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Score one model (same options as `eval compare`)."""
    if grader is None:
        grader = ["contains"]
    eval_compare([model], dataset=dataset, grader=grader, max_tokens=max_tokens, limit=limit)


@eval_app.command("judge")
def eval_judge(
    model: str,
    dataset: Annotated[Path, typer.Option(help="JSONL with messages (+ reference answers)")],
    metric: Annotated[
        list[str] | None, typer.Option(help="correctness, completeness, helpfulness, ...")
    ] = None,
    judge: Annotated[str, typer.Option(help="Judge model")] = "nova-pro",
    max_tokens: Annotated[int, typer.Option()] = 1024,
    no_wait: Annotated[bool, typer.Option("--no-wait")] = False,
) -> None:
    """Managed Bedrock evaluation job: a judge model scores MODEL (up to 1,000 prompts)."""
    with dataset.open(encoding="utf-8") as f:
        records = [r for _, r in parse_jsonl(f) if isinstance(r, dict)]
    job = _resolve_llm(model).create_eval_job(
        records,
        metrics=metric or ["correctness", "completeness", "helpfulness"],
        judge=judge,
        max_tokens=max_tokens,
    )
    console.print(f"Evaluation job {job.name}")
    if no_wait:
        return
    with console.status("judging (typically 10-30 min)..."):
        job.wait(poll_seconds=30)
    report = job.results()
    table = Table("metric", "mean (0-1)")
    for k, v in report.means.items():
        table.add_row(k, f"{v:.3f}")
    console.print(table)


def main() -> None:
    try:
        app()
    except SkyhammerError as e:
        err.print(f"[red]error:[/] {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
