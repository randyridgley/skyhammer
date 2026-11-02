"""``LLM`` resource: a base model you can fine-tune, or a fine-tuned model you can serve."""

from __future__ import annotations

import logging
import re
from typing import Any

from skyhammer import cost, pretrain, pricing, registry
from skyhammer import tags as tagging
from skyhammer.backends import DeploymentRef, DeploymentType, DeployOptions, get_backend
from skyhammer.backends.sagemaker import resolve_model_dir
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
from skyhammer.dataset import Dataset
from skyhammer.deployment import Deployment
from skyhammer.embeddings import Embeddings, RerankResponse, rerank
from skyhammer.errors import (
    CostLimitExceededError,
    SkyhammerError,
    UnsupportedParameterError,
    UnsupportedRouteError,
)
from skyhammer.evaluation import EvalJob
from skyhammer.graders import Grader
from skyhammer.imports import import_to_bedrock
from skyhammer.inference import Chat, Usage
from skyhammer.jobs import (
    FineTuningJob,
    Hyperparameters,
    JobRecord,
    ModelRef,
    TrainingRequest,
)
from skyhammer.metrics import ServingMetrics
from skyhammer.pricing import InferencePrices
from skyhammer.responses import Responses
from skyhammer.router import resolve
from skyhammer.serverless import ModelDirectory, Routing, ServerlessModel
from skyhammer.serving import EndpointDeployer, EndpointNames, Scaling
from skyhammer.session import Session, get_session

log = logging.getLogger(__name__)

FULL_FT_LEARNING_RATE = 1e-5  # SageMaker full fine-tuning default (recipes ship 1e-4)

# Bedrock job names are <=63 chars and the SDK appends "-<8 hex>".
_NAME = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9])*$")
_MAX_NAME = 54


def _check_name(name: str) -> None:
    if len(name) > _MAX_NAME or not _NAME.match(name):
        raise SkyhammerError(
            f"Invalid model name {name!r}: use letters, digits and hyphens (no dots or "
            f"underscores), at most {_MAX_NAME} characters"
        )


class LLM:
    """``LLM("nova-micro")`` is a base model; ``job.output_llm`` is a fine-tuned one."""

    def __init__(
        self,
        model: str,
        *,
        session: Session | None = None,
        routing: str | Routing | None = None,
        deployment_type: str = "auto",
    ) -> None:
        """``model`` is a fine-tunable catalog alias (``qwen3-8b``) or any Bedrock model alias
        or ID for serverless calls (``claude-sonnet-4-6``, ``meta.llama3-3-70b-instruct-v1:0``).

        ``routing``: ``geo`` (default; ``us.`` profiles), ``global``, or ``region``.
        ``deployment_type``: ``serverless`` (Bedrock per-token), ``dedicated`` (SageMaker
        endpoint via ``deploy()``), or ``auto`` (serverless when Bedrock serves the model,
        else dedicated)."""
        if deployment_type not in ("serverless", "dedicated", "auto"):
            raise SkyhammerError("deployment_type must be serverless, dedicated or auto")
        self.deployment_type = deployment_type
        self._session = session
        self.name = model
        try:
            self.entry: ModelEntry | None = get_model(model)
        except UnsupportedRouteError:
            self.entry = None  # serverless-only model; resolved lazily by the model directory
        self._routing = Routing(routing) if routing else None
        self.model_ref: ModelRef | None = None
        self.deployment: Deployment | None = None

    @property
    def routing(self) -> Routing:
        if self._routing:
            return self._routing
        s = self.session.settings
        if s.inference_routing:
            return Routing(s.inference_routing)
        return Routing.GEO if s.cross_region else Routing.REGION

    def _continuation_source(self, ref: ModelRef) -> str:
        """What a continued job trains from: the Bedrock custom model, or the model package
        SageMaker registered for the job that produced this model."""
        if ref.backend == BackendKind.BEDROCK.value:
            if ":custom-model/" not in ref.model_arn:
                raise SkyhammerError(f"{ref.name} isn't a Bedrock custom model ({ref.model_arn})")
            return ref.model_arn
        if ref.backend == BackendKind.SAGEMAKER.value:
            record = registry.find_job(self.session, ref.name)
            sm = self.session.client("sagemaker", region=ref.region)
            job = sm.describe_training_job(TrainingJobName=record.job_name)
            if not job.get("OutputModelPackageArn"):
                raise SkyhammerError(f"{record.job_name} registered no model package to continue")
            return str(job["OutputModelPackageArn"])
        raise SkyhammerError(f"Continued customization isn't supported on {ref.backend}")

    def _tunable(self) -> ModelEntry:
        if self.entry is None:
            raise SkyhammerError(
                f"{self.name!r} can be called serverless but isn't in the fine-tuning catalog. "
                "Run `skyhammer models list` for fine-tunable models."
            )
        return self.entry

    @property
    def info(self) -> ServerlessModel:
        """Bedrock metadata for the base model behind this LLM (serverless calls)."""
        target = self.entry.inference_id if self.entry and self.entry.inference_id else self.name
        return ModelDirectory.for_session(self.session).find(target)

    @classmethod
    def get(cls, name: str, *, session: Session | None = None) -> LLM:
        """A fine-tuned model by name, ready to chat if it was deployed before. An
        environment (``support-qwen@prod``) resolves to the model it serves."""
        session = session or get_session()
        if "@" in name:
            from skyhammer.environments import Environment

            return Environment(name, session=session).llm()
        manifest = registry.load_model(session, name)
        if manifest and manifest.get("deleted"):
            raise SkyhammerError(f"Model {name!r} was deleted")
        llm = FineTuningJob.get(name, session=session).output_llm
        dep = (manifest or {}).get("deployment")
        if dep and llm.model_ref:
            backend = get_backend(
                BackendKind(llm.model_ref.backend), llm.session, llm.model_ref.region
            )
            llm.deployment = Deployment(DeploymentRef(dep["name"], dep["arn"], "Unknown"), backend)
        return llm

    @classmethod
    def from_model_ref(cls, ref: ModelRef, session: Session | None = None) -> LLM:
        llm = cls(ref.base_alias, session=session)
        llm.model_ref = ref
        return llm

    @property
    def session(self) -> Session:
        return self._session or get_session()

    @property
    def is_fine_tuned(self) -> bool:
        return self.model_ref is not None

    def __repr__(self) -> str:
        if self.model_ref:
            return f"LLM(fine_tuned={self.model_ref.name!r}, base={self.model_ref.base_alias!r})"
        return f"LLM({self.name!r})"

    # ---- fine-tuning ----

    def create_sft_job(
        self, dataset: Dataset, *, output_model: str, **kwargs: Any
    ) -> FineTuningJob:
        return self._create_job(Technique.SFT, dataset, output_model=output_model, **kwargs)

    def create_distillation_job(
        self,
        dataset: Dataset,
        *,
        teacher: str | LLM,
        output_model: str,
        teacher_max_tokens: int = 1024,
        teacher_temperature: float | None = None,
        use_batch: bool | None = None,
        **kwargs: Any,
    ) -> FineTuningJob:
        """Train this model on a teacher's answers to ``dataset``'s prompts.

        Nova students use Bedrock-native distillation. Other students (open-weight models)
        get teacher answers first (live calls; a 50%-off batch job from 1,000 prompts, or set
        ``use_batch``), then a normal SFT job; this call blocks until the
        answers are ready. Answers are cached in S3, so re-running reattaches.
        """
        if self.is_fine_tuned or self._tunable().capabilities_for(Technique.DISTILLATION):
            return self._create_job(
                Technique.DISTILLATION,
                dataset,
                output_model=output_model,
                teacher_model=teacher if isinstance(teacher, str) else teacher.model_id,
                **kwargs,
            )
        from skyhammer import distill

        teacher_llm = teacher if isinstance(teacher, LLM) else LLM(teacher, session=self.session)

        convs = [distill.prompt_messages(r) for _, r in dataset._rows]
        batch = len(convs) >= distill.BATCH_FROM if use_batch is None else use_batch
        bound = distill.estimate_teacher_cost(teacher_llm, convs, teacher_max_tokens, batch=batch)
        limit = kwargs.get("max_cost_usd", self.session.settings.guardrails.max_cost_usd)
        if bound is not None:
            log.info("Teacher %s: at most ~$%.4f for %d prompts", teacher, bound, len(convs))
            if limit is not None and bound > limit:
                raise CostLimitExceededError(bound, limit)
        answers = distill.teacher_dataset(
            teacher_llm,
            dataset,
            max_tokens=teacher_max_tokens,
            temperature=teacher_temperature,
            use_batch=use_batch,
        )
        tags = {"distilled-from": teacher_llm.model_id, **(kwargs.pop("tags", None) or {})}
        return self._create_job(
            Technique.SFT, answers, output_model=output_model, tags=tags, **kwargs
        )

    def create_dpo_job(
        self, dataset: Dataset, *, output_model: str, **kwargs: Any
    ) -> FineTuningJob:
        return self._create_job(Technique.DPO, dataset, output_model=output_model, **kwargs)

    def create_rft_job(
        self, dataset: Dataset, *, grader: Grader, output_model: str, **kwargs: Any
    ) -> FineTuningJob:
        return self._create_job(
            Technique.RFT, dataset, output_model=output_model, grader=grader, **kwargs
        )

    def create_cpt_job(
        self, dataset: Dataset, *, output_model: str, **kwargs: Any
    ) -> FineTuningJob:
        return self._create_job(Technique.CPT, dataset, output_model=output_model, **kwargs)

    def plan(
        self,
        technique: str | Technique,
        dataset: Dataset,
        *,
        output_model: str,
        validation_dataset: Dataset | None = None,
        validation_split: float | None = None,
        auto_fix: bool = False,
        backend: str | BackendKind | None = None,
        serving: str | ServingTarget | None = None,
        accept_eula: bool = False,
        training_type: str | TrainingType = TrainingType.LORA,
        teacher_model: str | None = None,
        grader: Grader | None = None,
        tags: dict[str, str] | None = None,
        hyperparameters_extra: dict[str, Any] | None = None,
        max_hours: float | None = None,
        **hyperparameters: Any,
    ) -> tuple[TrainingRequest, cost.CostEstimate]:
        """Resolve route, validate data, and estimate cost without touching AWS.

        On a fine-tuned model (``LLM.get(name)``) this plans *continued* customization: the
        job trains that model further, on the backend and region that produced it."""
        source_model = None
        if self.model_ref is not None:
            ref = self.model_ref
            if backend and BackendKind(backend).value != ref.backend:
                raise SkyhammerError(
                    f"{ref.name} was trained on {ref.backend}; continued customization stays there"
                )
            backend = BackendKind(ref.backend)
            serving = serving or ref.serving
            source_model = self._continuation_source(ref)
        entry = self._tunable()
        technique = Technique(technique)
        if technique is Technique.CPT and entry.family == "nova":
            raise UnsupportedRouteError(
                f"{entry.display_name} continued pre-training runs only as SageMaker HyperPod "
                "recipes on 8-24 ml.p5.48xlarge instances, which Skyhammer doesn't manage. "
                "Skyhammer CPT covers open-weight models",
                alternatives=[
                    e.alias for e in CATALOG.values() if e.capabilities_for(Technique.CPT)
                ],
            )
        if technique is Technique.DISTILLATION and not entry.capabilities_for(technique):
            raise SkyhammerError(
                f"{entry.display_name} distills in two steps (teacher answers, then SFT), so "
                "there is no single plan up front. create_distillation_job() logs the teacher "
                "cost bound, then plans the SFT job on the answers."
            )
        _check_name(output_model)
        settings = self.session.settings
        ttype = TrainingType(training_type)
        route = resolve(
            entry,
            technique,
            self.model_ref.region if self.model_ref else settings.region,
            backend=BackendKind(backend) if backend else None,
            training_type=ttype,
            serving=ServingTarget(serving) if serving else None,
            cross_region=settings.cross_region,
            allowed_regions=settings.allowed_regions,
        )
        if route.cross_region:
            log.warning("%s: %s", entry.alias, route.describe())
        if route.backend is BackendKind.SAGEMAKER and entry.requires_eula and not accept_eula:
            raise SkyhammerError(
                f"{entry.display_name} has a gated license. Review it, then pass "
                "accept_eula=True (or `accept_eula: true` in the job spec)."
            )

        if validation_split and validation_dataset:
            raise SkyhammerError("Pass validation_dataset or validation_split, not both")
        if validation_split:
            dataset, validation_dataset = dataset.split(validation_split)
        limits = get_backend(route.backend, self.session, route.region).dataset_limits(
            entry, technique, ttype
        )
        dataset.prepare(entry, technique, auto_fix=auto_fix, limits=limits)
        if validation_dataset:
            validation_dataset.prepare(
                entry, technique, auto_fix=auto_fix, is_validation_set=True, limits=limits
            )
        assert dataset.report is not None
        for w in dataset.report.warnings:
            log.warning("dataset %s: %s", dataset.name, w.message)

        if (
            route.backend is BackendKind.SAGEMAKER
            and ttype is TrainingType.FULL
            and hyperparameters.get("learning_rate") is None
        ):
            # The recipes default to 1e-4, a LoRA-sized rate. For full fine-tuning it made
            # Qwen3-0.6B memorize answers and invent details (live, 2026-10-30); 1e-5
            # generalized and stopped cleanly.
            hyperparameters = {**hyperparameters, "learning_rate": FULL_FT_LEARNING_RATE}
            log.info(
                "full fine-tuning: learning_rate=%g (set learning_rate to change)",
                FULL_FT_LEARNING_RATE,
            )
        hp = Hyperparameters(**hyperparameters, extra=hyperparameters_extra or {})
        request = TrainingRequest(
            technique=technique,
            route=route,
            dataset=dataset,
            output_model=output_model,
            hyperparameters=hp,
            training_type=ttype,
            validation_dataset=validation_dataset,
            teacher_model=teacher_model,
            grader=grader,
            accept_eula=accept_eula,
            source_model=source_model,
        )
        request.tags = tagging.build_tags(
            {
                "name": output_model,
                "base-model": entry.alias,
                "technique": technique.value,
                "serving": route.serving.value,
                "dataset": dataset.name,
                "dataset-sha": dataset.sha256[:16],
                "fingerprint": request.fingerprint()[:16],
            },
            {**settings.tags, **(tags or {})},
        )
        if technique is Technique.CPT:  # a training job, billed per instance-hour
            price = pricing.instance_price(
                self.session, route.region, pretrain.instance_type(request)
            )
        else:
            price = pricing.training_price(self.session, route, technique, ttype)
        if price.usd_per_hour is not None or technique in (Technique.RFT, Technique.CPT):
            if route.backend is not BackendKind.SAGEMAKER and max_hours is not None:
                raise UnsupportedParameterError(
                    "max_hours applies to time-billed SageMaker jobs (RL, CPT) only"
                )
            if route.backend in (BackendKind.BEDROCK, BackendKind.MANTLE):
                log.warning(
                    "Bedrock RFT is billed by training time%s and has no runtime cap; "
                    "keep epochs and prompts small, and job.cancel() stops it",
                    f" (${price.usd_per_hour:g}/h)" if price.usd_per_hour else "",
                )
            if route.backend is BackendKind.SAGEMAKER and max_hours is None:
                max_hours = cost.DEFAULT_RL_MAX_HOURS
                log.warning(
                    "%s is billed by time; stopping after %gh (set max_hours to change)",
                    technique.value,
                    max_hours,
                )
        request.max_hours = max_hours
        if route.backend is BackendKind.SAGEMAKER:  # recipe checks (batch sizes, bounds) now
            get_backend(route.backend, self.session, route.region).check(request)  # type: ignore[attr-defined]
        estimate = cost.estimate_training(
            dataset.report.estimated_tokens,
            hp.epochs,
            route.backend.value,
            entry.alias,
            usd_per_1k_tokens=price.usd_per_1k_tokens,
            usd_per_hour=price.usd_per_hour,
            max_hours=max_hours,
            exact_tokens=dataset.report.exact_tokens,
        )
        if technique is Technique.DISTILLATION:
            estimate = cost.CostEstimate(
                estimate.training_tokens,
                estimate.usd_per_1k_tokens,
                (
                    *estimate.notes,
                    f"excludes teacher inference ({teacher_model}), billed at its "
                    "on-demand rates for each prompt",
                ),
            )
        return request, estimate

    def create_job(
        self, technique: str | Technique, dataset: Dataset, *, output_model: str, **kwargs: Any
    ) -> FineTuningJob:
        """Generic entry point (used by specs/CLI); prefer the create_<technique>_job helpers."""
        if Technique(technique) is Technique.DISTILLATION:
            teacher = kwargs.pop("teacher_model", None)
            if not teacher:
                raise SkyhammerError("Distillation needs a teacher model")
            return self.create_distillation_job(
                dataset, teacher=teacher, output_model=output_model, **kwargs
            )
        return self._create_job(Technique(technique), dataset, output_model=output_model, **kwargs)

    def _create_job(
        self,
        technique: Technique,
        dataset: Dataset,
        *,
        output_model: str,
        max_cost_usd: float | None = None,
        **kwargs: Any,
    ) -> FineTuningJob:
        request, estimate = self.plan(technique, dataset, output_model=output_model, **kwargs)
        cost.check_guardrails(estimate, self.session.settings.guardrails, max_cost_usd)
        backend = get_backend(request.route.backend, self.session, request.route.region)

        handle = backend.find(request)
        record = registry.load_job(self.session, handle.name) if handle else None
        if handle:
            log.info("Reattaching to existing job %s", handle.name)
        else:
            log.info(
                "Submitting %s via %s in %s: %s",
                request.job_name,
                backend.name,
                request.route.region,
                estimate,
            )
            handle = backend.submit(request)
        if record is None:
            record = JobRecord.from_request(request, handle, str(estimate))
            registry.save_job(self.session, record)
        return FineTuningJob(record, backend, request)

    # ---- serving ----

    def deploy(
        self,
        name: str | None = None,
        *,
        type: str | DeploymentType = "on_demand",
        serving: str | ServingTarget | None = None,
        scaling: Scaling | dict[str, Any] | None = None,
        instance_type: str | list[str] | None = None,
        accept_eula: bool = False,
        wait: bool = True,
        on: str | None = None,
        adapters: bool = False,
        model_units: int = 1,
        commitment: str | None = None,
        max_hourly_usd: float | None = None,
        accept_commitment: bool = False,
    ) -> LLM:
        """Make the model invocable.

        Bedrock-served models are serverless. SageMaker-served models get an endpoint that
        scales to zero when idle by default; pass ``scaling=Scaling(min_copies=1, ...)`` to
        keep it warm or tune autoscaling. ``serving="bedrock"`` on a SageMaker-served model
        re-imports its stored weights into Bedrock (no retraining).

        ``instance_type`` (SageMaker only): leave unset to use AWS's recommended instance
        with automatic fallback to other GPU types when it has no capacity; pass a list to
        set your own fallback order, or one type to pin it.

        Multi-LoRA (SageMaker): ``deploy(adapters=True)`` on a base model starts a shared
        endpoint that can host LoRA adapters; ``LLM.get("my-lora").deploy(on="shared")``
        adds a fine-tune's adapter to it (no new GPUs). Adapters come from LoRA jobs with
        ``serving="sagemaker"`` and ``hyperparameters_extra={"merge_weights": False}``.

        Provisioned Throughput (Bedrock): ``type="provisioned"`` buys ``model_units`` of
        reserved capacity for this fine-tune or base model, billed by the hour whether used
        or not. ``max_hourly_usd`` is required and must cover the Price List rate (see
        ``skyhammer.provisioned.price``). ``commitment="1_month"`` or ``"6_months"`` is
        cheaper per hour but can't be cancelled, so it also needs ``accept_commitment=True``.
        Deploying another fine-tune of the same base under an existing name swaps it in."""
        if DeploymentType(type) is DeploymentType.PROVISIONED:
            return self._deploy_provisioned(
                name,
                model_units=model_units,
                commitment=commitment,
                max_hourly_usd=max_hourly_usd,
                accept_commitment=accept_commitment,
                wait=wait,
            )
        if self.model_ref is None:
            if on:
                raise SkyhammerError("on= adds a fine-tuned LoRA adapter; deploy the base first")
            return self._deploy_base(
                name,
                scaling=scaling,
                instance_type=instance_type,
                accept_eula=accept_eula,
                adapters=adapters,
            )
        if adapters:
            raise SkyhammerError("adapters=True starts a shared base-model endpoint")
        ref = self.model_ref
        if serving and ServingTarget(serving).value != ref.serving:
            return (
                self._move_to_bedrock()
                if ServingTarget(serving) is ServingTarget.BEDROCK
                else self._unsupported_move()
            )
        backend = get_backend(BackendKind(ref.backend), self.session, ref.region)
        opts = DeployOptions(
            name=name or ref.name,
            type=DeploymentType(type),
            tags=tagging.build_tags(
                {"name": ref.name, "base-model": ref.base_alias}, self.session.settings.tags
            ),
            scaling=Scaling.model_validate(scaling) if isinstance(scaling, dict) else scaling,
            instance_type=instance_type,
            on=on,
        )
        if on and (opts.scaling or instance_type):
            raise SkyhammerError("Adapters share the endpoint's scaling and instances")
        if (on or opts.scaling or instance_type) and ref.serving != ServingTarget.SAGEMAKER.value:
            raise SkyhammerError(
                "scaling/instance_type apply only to SageMaker-served models; "
                "Bedrock serving is already serverless"
            )
        self.deployment = Deployment(backend.deploy(ref, opts), backend)
        if wait:
            self.deployment.wait()
        self._record_in_manifest(
            deployment={"name": self.deployment.name, "arn": self.deployment.arn}
        )
        return self

    def _unsupported_move(self) -> LLM:
        raise SkyhammerError(
            "Moving a Bedrock-served model to SageMaker isn't supported; "
            "train with serving='sagemaker' instead"
        )

    def _move_to_bedrock(self) -> LLM:
        """SageMaker-served -> Bedrock Custom Model Import, from the stored weights."""
        ref = self.model_ref
        assert ref is not None
        if not ref.artifacts_uri:
            raise SkyhammerError(f"{ref.name} has no stored weights to import")
        model_dir = resolve_model_dir(
            self.session.client("s3", region=ref.region), ref.artifacts_uri
        )
        arn = import_to_bedrock(
            self.session,
            ref.region,
            model_dir,
            ref.name,
            tags=tagging.build_tags(
                {"name": ref.name, "base-model": ref.base_alias}, self.session.settings.tags
            ),
            log=lambda m: log.info("%s: %s", ref.name, m),
        )
        self.model_ref = ModelRef(
            ref.name,
            arn,
            "sagemaker",
            ref.base_alias,
            ref.region,
            api="openai_chat",
            serving=ServingTarget.BEDROCK.value,
            artifacts_uri=ref.artifacts_uri,
        )
        old, self.deployment = self.deployment, None
        if old is not None:
            old.delete()  # the SageMaker endpoint is no longer needed
        self._record_in_manifest(deployment=None, serving="bedrock")
        return self

    @classmethod
    def import_weights(
        cls,
        s3_uri: str,
        *,
        name: str,
        base: str | None = None,
        session: Session | None = None,
        wait: bool = True,
    ) -> LLM:
        """Bring your own Hugging Face weights (in S3) into Bedrock, served on demand."""
        _check_name(name)
        session = session or get_session()
        arn = import_to_bedrock(
            session,
            session.region,
            s3_uri,
            name,
            wait=wait,
            tags=tagging.build_tags({"name": name, "kind": "imported"}, session.settings.tags),
        )
        return cls.from_model_ref(
            ModelRef(
                name,
                arn,
                "sagemaker",
                base or name,
                session.region,
                api="openai_chat",
                serving="bedrock",
                artifacts_uri=s3_uri,
            ),
            session=session,
        )

    def serverless_available(self) -> bool:
        try:
            self.info.invocation_id(self.routing, self.session.region)
        except SkyhammerError:
            return False
        return True

    def _deploy_provisioned(
        self,
        name: str | None,
        *,
        model_units: int,
        commitment: str | None,
        max_hourly_usd: float | None,
        accept_commitment: bool,
        wait: bool,
    ) -> LLM:
        from skyhammer import provisioned

        ref = self.model_ref
        if ref and (ref.backend != BackendKind.BEDROCK.value or ref.api != "converse"):
            raise SkyhammerError(
                f"{ref.name} isn't a Bedrock custom model; Provisioned Throughput serves "
                "Bedrock fine-tunes (Nova, Titan) and base models"
            )
        region = ref.region if ref else self.session.region
        term = provisioned.commitment_term(commitment)
        estimate = provisioned.price(
            self.session,
            self.info.name,
            model_units=model_units,
            commitment=term,
            custom=ref is not None,
            region=region,
        )
        provisioned.check_budget(estimate, max_hourly_usd, accept_commitment=accept_commitment)
        label = ref.name if ref else self.name
        log.warning("Provisioned Throughput for %s: %s", label, estimate)
        backend = get_backend(BackendKind.BEDROCK, self.session, region)
        if ref:
            target = ref
        else:
            pid = provisioned.Provisioned(self.session, region).base_model_id(self.info.model_id)
            target = ModelRef(self.name, pid, BackendKind.BEDROCK.value, self.name, region)
        opts = DeployOptions(
            name=name or (ref.name if ref else f"skyhammer-{label}-provisioned".replace(".", "-")),
            type=DeploymentType.PROVISIONED,
            model_units=model_units,
            commitment=term,
            tags=tagging.build_tags(
                {"name": label, "base-model": ref.base_alias if ref else self.name},
                self.session.settings.tags,
            ),
        )
        self.deployment = Deployment(backend.deploy(target, opts), backend)
        if wait:
            self.deployment.wait(poll_seconds=30)
        self._record_in_manifest(
            deployment={"name": self.deployment.name, "arn": self.deployment.arn}
        )
        return self

    @classmethod
    def provisioned(cls, name: str, *, session: Session | None = None) -> LLM:
        """A base model's Provisioned Throughput by the name ``deploy`` gave it (fine-tunes
        reattach to theirs with ``LLM.get``)."""
        from skyhammer.provisioned import Provisioned

        session = session or get_session()
        pts = Provisioned(session)
        for pt, tags in pts.tagged():
            if pt.name == name:
                llm = cls(tags.get("skyhammer:base-model", tags["skyhammer:name"]), session=session)
                backend = get_backend(BackendKind.BEDROCK, session, pts.region)
                llm.deployment = Deployment(DeploymentRef(pt.name, pt.arn, pt.status), backend)
                return llm
        raise SkyhammerError(f"No Skyhammer Provisioned Throughput named {name!r}")

    @property
    def is_provisioned(self) -> bool:
        """Served by Provisioned Throughput (billed per model-unit hour, not per token)."""
        from skyhammer.provisioned import is_provisioned_arn

        return bool(self.deployment and is_provisioned_arn(self.deployment.arn))

    def _deploy_base(
        self,
        name: str | None,
        *,
        scaling: Scaling | dict[str, Any] | None,
        instance_type: str | list[str] | None,
        accept_eula: bool,
        adapters: bool = False,
    ) -> LLM:
        """Base models: serverless needs no deploy; dedicated gets a SageMaker endpoint."""
        if self.deployment_type == "serverless" or (
            self.deployment_type == "auto" and self.serverless_available()
        ):
            log.info("%s is served serverless by Bedrock; no deployment needed", self.name)
            return self
        entry = self._tunable()
        backend = get_backend(BackendKind.SAGEMAKER, self.session)
        opts = DeployOptions(
            name=name or f"skyhammer-{entry.alias}".replace(".", "-"),
            scaling=Scaling.model_validate(scaling) if isinstance(scaling, dict) else scaling,
            instance_type=instance_type,
            adapters=adapters,
            tags=tagging.build_tags(
                {"name": name or entry.alias, "base-model": entry.alias, "kind": "dedicated-base"},
                self.session.settings.tags,
            ),
        )
        dep, ref = backend.deploy_base(entry.alias, opts, accept_eula=accept_eula)  # type: ignore[attr-defined]
        self.model_ref, self.deployment = ref, Deployment(dep, backend)
        return self

    def _record_in_manifest(self, **changes: Any) -> None:
        if self.model_ref is None:
            return
        try:
            registry.update_model(self.session, self.model_ref.name, **changes)
        except SkyhammerError:
            log.debug("no manifest for %s; not recording %s", self.model_ref.name, changes)

    def delete(self) -> None:
        """Delete this fine-tuned model's deployment (if any) and the model itself.
        Training data, artifacts and manifests in S3 are kept."""
        if self.model_ref is None:
            raise SkyhammerError("Base models can't be deleted")
        backend = get_backend(
            BackendKind(self.model_ref.backend), self.session, self.model_ref.region
        )
        if self.deployment:
            self.deployment.delete()
            self.deployment = None
        backend.delete_model(self.model_ref)
        self._record_in_manifest(deleted=True, deployment=None)

    @property
    def model_id(self) -> str:
        """The ``modelId`` to pass to Bedrock runtime APIs."""
        if self.deployment:
            return self.deployment.arn
        if self.model_ref:
            if self.model_ref.api in ("openai_chat", "mantle") and (
                self.model_ref.serving != "sagemaker"
            ):
                return self.model_ref.model_arn  # imported/mantle models are invoked directly
            raise SkyhammerError(f"{self.model_ref.name} must be deployed first: call .deploy()")
        # Base/serverless model: the inference profile or model ID per routing policy.
        if self.deployment_type == "dedicated":
            raise SkyhammerError(f"{self.name} is set to a dedicated deployment: call .deploy()")
        try:
            return self.info.invocation_id(self.routing, self.session.region)
        except SkyhammerError:
            if (
                self.deployment_type == "auto"
                and self.entry
                and any(c.backend is BackendKind.SAGEMAKER for c in self.entry.capabilities)
            ):
                raise SkyhammerError(
                    f"Bedrock doesn't serve base {self.name} serverless; call .deploy() for a "
                    "dedicated SageMaker endpoint (scales to zero when idle)"
                ) from None
            raise

    def _record_usage(self, model_id: str, usage: Usage, tier: str | None) -> None:
        """Fill usage.cost_usd from live prices and add it to the session's spend."""
        if (
            self.session.settings.track_costs
            and not self.is_provisioned  # billed per model-unit hour, not per token
            and not (self.model_ref and self.model_ref.api in ("openai_chat", "mantle"))
        ):  # imports bill per CMU; mantle fine-tunes aren't in the Price List
            try:
                usage.cost_usd = InferencePrices.for_session(self.session).cost(
                    self.info.name,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                    tier="standard" if tier in (None, "default") else tier,
                    routing="global" if model_id.startswith("global.") else "regional",
                )
            except Exception as e:  # pricing is best-effort; never fail the call
                log.debug("cost lookup for %s failed: %s", model_id, e)
        self.session.spend.record(model_id, usage.cost_usd)

    def batch(
        self,
        requests: list[dict[str, Any]],
        *,
        name: str | None = None,
        timeout_hours: int = 24,
        **defaults: Any,
    ) -> BatchJob:
        """Submit a discounted batch job: OpenAI-style requests (``messages`` + per-request
        params, optional ``custom_id``). ``defaults`` apply to every request."""
        model_id = self.model_id
        try:
            model_name = self.info.name
        except SkyhammerError:
            model_name = None
        return BatchJob.submit(
            self.session,
            model_id,
            requests,
            name=name,
            model_name=model_name,
            timeout_hours=timeout_hours,
            defaults=defaults,
        )

    def create_eval_job(
        self,
        records: list[dict[str, Any]] | Dataset,
        *,
        metrics: list[str],
        judge: str | LLM = "nova-pro",
        name: str | None = None,
        max_tokens: int = 1024,
        precomputed: bool | None = None,
    ) -> EvalJob:
        """Managed evaluation on Bedrock: a judge model scores this model's answers.

        ``metrics`` are built-in judge metrics such as ``correctness``, ``completeness``,
        ``helpfulness``, ``faithfulness``, ``harmfulness`` (see ``JUDGE_METRICS``). Up to
        1,000 prompts. SageMaker-served models are answered locally first and the
        responses are judged (``precomputed``). Same inputs → same job (idempotent).
        """
        rows = [r for _, r in records._rows] if isinstance(records, Dataset) else records
        return EvalJob.submit(
            self,
            rows,
            metrics=metrics,
            judge=judge,
            name=name,
            max_tokens=max_tokens,
            precomputed=precomputed,
        )

    def quotas(self) -> list[dict[str, Any]]:
        """Bedrock Service Quotas for this model: tokens/requests per minute (on-demand, geo
        and global cross-region), daily tokens, batch limits."""
        info = self.info
        pattern = re.compile(
            rf" for (?:{re.escape(info.provider)}\s+)?{re.escape(info.name)}(?=$|[\s(])"
        )
        sq = self.session.client("service-quotas")
        out = []
        for page in sq.get_paginator("list_service_quotas").paginate(ServiceCode="bedrock"):
            for q in page["Quotas"]:
                if pattern.search(q["QuotaName"]):
                    out.append(
                        {
                            "name": q["QuotaName"],
                            "value": q["Value"],
                            "adjustable": q.get("Adjustable", False),
                            "code": q["QuotaCode"],
                        }
                    )
        return sorted(out, key=lambda q: q["name"])

    @property
    def embeddings(self) -> Embeddings:
        """OpenAI-style ``embeddings.create(input=..., dimensions=...)`` for embedding models."""
        if self.info.kind != "embedding":
            raise SkyhammerError(f"{self.name} is a {self.info.kind} model, not an embedding model")
        model_id = self.model_id

        def on_usage(tokens: int) -> float | None:
            cost = None
            if self.session.settings.track_costs:
                try:
                    cost = InferencePrices.for_session(self.session).cost(self.info.name, tokens, 0)
                except Exception as e:
                    log.debug("embedding cost lookup failed: %s", e)
            self.session.spend.record(model_id, cost)
            return cost

        return Embeddings(
            self.session.client("bedrock-runtime"),
            model_id,
            on_usage=on_usage,
            before_call=self.session.spend.check,
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        return_documents: bool = False,
    ) -> RerankResponse:
        """Order ``documents`` by relevance to ``query`` (Cohere Rerank, Amazon Rerank)."""
        if self.info.kind != "rerank":
            raise SkyhammerError(f"{self.name} is a {self.info.kind} model, not a rerank model")
        self.session.spend.check()
        price = None
        if self.session.settings.track_costs:
            try:
                price = InferencePrices.for_session(self.session).search_unit_price(self.info.name)
            except Exception as e:
                log.debug("rerank price lookup failed: %s", e)
        resp = rerank(
            self.session.client("bedrock-agent-runtime"),
            self.session.region,
            self.info.model_id,
            query,
            documents,
            top_n=top_n,
            return_documents=return_documents,
            price_per_search_unit=price,
        )
        self.session.spend.record(self.info.model_id, resp.usage.cost_usd)
        return resp

    @property
    def completions(self) -> Any:
        """OpenAI-style ``completions.create(prompt=...)`` (prompt sent as one user turn)."""
        return self.chat.text_completions

    def metrics(self, hours: float = 24) -> ServingMetrics:
        """Traffic, latency (avg/p50/p99 ms), tokens and errors from CloudWatch."""
        from skyhammer import metrics

        ref = self.model_ref
        if ref and ref.serving == ServingTarget.SAGEMAKER.value and self.deployment:
            names = EndpointNames.for_deployment(self.deployment.name)
            return metrics.fetch(
                self.session, self.model_id, component=names.component, hours=hours
            )
        return metrics.fetch(self.session, self.model_id, hours=hours)

    def warm(self, copies: int = 1) -> LLM:
        """Pre-scale a scale-to-zero SageMaker endpoint before a burst of traffic."""
        ref = self.model_ref
        if not ref or ref.serving != ServingTarget.SAGEMAKER.value or not self.deployment:
            return self  # Bedrock serving restores on demand; nothing to pre-scale
        EndpointDeployer(self.session, ref.region).warm(
            EndpointNames.for_deployment(self.deployment.name), copies
        )
        return self

    @property
    def responses(self) -> Responses:
        """The OpenAI Responses API: native where Bedrock serves it, emulated elsewhere."""
        return Responses(self)

    @property
    def chat(self) -> Chat:
        if self.model_ref and self.model_ref.serving == ServingTarget.SAGEMAKER.value:
            if not self.deployment:
                raise SkyhammerError(f"{self.model_ref.name} must be deployed first: .deploy()")
            names = EndpointNames.for_deployment(self.deployment.name)
            return Chat(
                self.session.client("sagemaker-runtime", region=self.model_ref.region),
                names.endpoint,
                api="sagemaker",
                component=names.component,
                on_usage=lambda usage, tier: self.session.spend.record(
                    names.endpoint, None
                ),  # billed per instance-hour, not per token
                before_call=self.session.spend.check,
                guardrail_runtime=self.session.client(
                    "bedrock-runtime", region=self.model_ref.region
                ),
            )
        region = self.model_ref.region if self.model_ref else None
        model_id = self.model_id
        if self.model_ref and self.model_ref.api == "mantle":
            from skyhammer.mantle import MantleClient

            return Chat(
                MantleClient(self.session, self.model_ref.region),
                model_id,
                api="mantle",
                on_usage=lambda usage, tier: self._record_usage(model_id, usage, tier),
                before_call=self.session.spend.check,
                guardrail_runtime=self.session.client("bedrock-runtime", region=region),
            )
        return Chat(
            self.session.client("bedrock-runtime", region=region),
            model_id,
            api=self.model_ref.api if self.model_ref else "converse",
            on_usage=lambda usage, tier: self._record_usage(model_id, usage, tier),
            before_call=self.session.spend.check,
        )
