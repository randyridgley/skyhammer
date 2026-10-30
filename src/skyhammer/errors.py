"""Exception hierarchy. Every error raised by the SDK derives from ``SkyhammerError``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skyhammer.validation import ValidationReport


class SkyhammerError(Exception):
    """Base class for all skyhammer errors."""


class ConfigError(SkyhammerError):
    """Configuration is missing or invalid."""


class BootstrapMissingError(ConfigError):
    """The bootstrap stack was not found and no explicit resources were configured."""

    def __init__(self, region: str) -> None:
        super().__init__(
            f"No skyhammer bootstrap stack found in {region}. Run "
            f"`skyhammer bootstrap --region {region}` or set bucket/role ARNs explicitly in "
            "skyhammer.yaml."
        )
        self.region = region


class UnsupportedRouteError(SkyhammerError):
    """No backend can run the requested (model, technique, region) combination."""

    def __init__(self, message: str, alternatives: list[str] | None = None) -> None:
        self.alternatives = alternatives or []
        if self.alternatives:
            message += "\nAlternatives:\n" + "\n".join(f"  - {a}" for a in self.alternatives)
        super().__init__(message)


class UnsupportedParameterError(SkyhammerError):
    """A hyperparameter was passed that the selected backend does not support."""


class DatasetValidationError(SkyhammerError):
    """Dataset failed validation. Inspect ``.report`` for details and suggested fixes."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(report.summary())


class CostLimitExceededError(SkyhammerError):
    """Estimated cost exceeds the configured ``max_cost_usd``."""

    def __init__(self, estimate_usd: float, limit_usd: float) -> None:
        self.estimate_usd = estimate_usd
        self.limit_usd = limit_usd
        super().__init__(
            f"Estimated cost ${estimate_usd:,.2f} exceeds limit ${limit_usd:,.2f}. "
            "Raise max_cost_usd or reduce epochs/dataset size."
        )


class JobFailedError(SkyhammerError):
    """A training, merge, or import job ended in a failed state."""

    def __init__(self, job_name: str, reason: str, log_url: str | None = None) -> None:
        self.job_name = job_name
        self.reason = reason
        self.log_url = log_url
        msg = f"Job {job_name!r} failed: {reason}"
        if log_url:
            msg += f"\nLogs: {log_url}"
        super().__init__(msg)


class ThrottledError(SkyhammerError):
    """Bedrock throttled the model after retries (tokens/requests per minute quota)."""

    def __init__(self, model_id: str, message: str) -> None:
        self.model_id = model_id
        hint = []
        if not model_id.startswith("global."):
            hint.append("routing='global' for more capacity")
        hint += [
            "service_tier='priority' for latency-critical traffic",
            "service_tier='flex' or llm.batch(...) for offline work",
            f"`skyhammer models quotas {model_id}` to see limits and request an increase",
        ]
        super().__init__(f"Bedrock throttled {model_id}: {message}\nTry: " + "; ".join(hint))


class SpendLimitExceededError(SkyhammerError):
    """Serverless spend in this process reached ``max_spend_usd``."""

    def __init__(self, spent_usd: float, limit_usd: float) -> None:
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        super().__init__(
            f"Serverless spend ${spent_usd:,.4f} reached the limit ${limit_usd:,.4f}. "
            "Raise max_spend_usd or reset the tracker (session.spend)."
        )
