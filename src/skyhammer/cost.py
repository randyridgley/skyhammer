"""Pre-flight cost estimates and guardrails.

SFT/DPO/distillation are billed per training token (dataset tokens x epochs) plus monthly
model storage; RL techniques on SageMaker bill per hour. Prices come from the AWS Price
List API (``pricing.py``) or from the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from skyhammer.config import Guardrails
from skyhammer.errors import CostLimitExceededError
from skyhammer.pricing import CUSTOM_MODEL_STORAGE_USD_MONTH

log = logging.getLogger(__name__)

# Epochs assumed when the user doesn't set them. SageMaker recipes default to 5 (from the
# Hub override params); TODO(M1): confirm Bedrock's default from a real job.
DEFAULT_EPOCHS = {"bedrock": 2, "sagemaker": 5, "mantle": 1}  # mantle: "auto" in the API
# Runtime cap for time-billed (RL) jobs when the caller doesn't set max_hours.
DEFAULT_RL_MAX_HOURS = 4.0


@dataclass(frozen=True)
class CostEstimate:
    training_tokens: int
    usd_per_1k_tokens: float | None
    notes: tuple[str, ...] = ()
    usd_per_hour: float | None = None  # time-billed (RL) jobs
    max_hours: float | None = None  # their runtime cap; usd is then an upper bound

    @property
    def usd(self) -> float | None:
        if self.usd_per_hour is not None:
            return self.usd_per_hour * self.max_hours if self.max_hours is not None else None
        if self.usd_per_1k_tokens is None:
            return None
        return self.training_tokens / 1000 * self.usd_per_1k_tokens

    def _repr_html_(self) -> str:
        from skyhammer.progress import html_table

        return html_table([("estimate", str(self)), *(("note", n) for n in self.notes)])

    def __str__(self) -> str:
        if self.usd_per_hour is not None and self.usd is not None:
            return (
                f"billed by time at ${self.usd_per_hour:,.2f}/hour, at most "
                f"${self.usd:,.2f} (stops after {self.max_hours:g}h)"
            )
        if self.usd_per_hour is not None:  # time-billed with no runtime cap (Bedrock RFT)
            return f"billed by time at ${self.usd_per_hour:,.2f}/hour, no runtime cap"
        cost = f"${self.usd:,.4f}" if self.usd is not None else "unknown (no price data)"
        return f"~{self.training_tokens:,} training tokens, estimated {cost}"


def estimate_training(
    dataset_tokens: int,
    epochs: int | None,
    backend: str,
    model_alias: str,
    usd_per_1k_tokens: float | None = None,
    usd_per_hour: float | None = None,
    *,
    max_hours: float | None = None,
    exact_tokens: bool = False,
) -> CostEstimate:
    notes = [] if exact_tokens else ["token count is a heuristic (~4 chars/token)"]
    if usd_per_hour is not None:
        notes.append("time-billed: the estimate is the cost of running until max_hours")
    notes.append(f"plus ${CUSTOM_MODEL_STORAGE_USD_MONTH:.2f}/month model storage until deleted")
    if epochs is None:
        epochs = DEFAULT_EPOCHS[backend]
        notes.append(f"assumes {epochs} epochs ({backend} default)")
    return CostEstimate(
        training_tokens=dataset_tokens * epochs,
        usd_per_1k_tokens=usd_per_1k_tokens,
        notes=tuple(notes),
        usd_per_hour=usd_per_hour,
        max_hours=max_hours,
    )


def check_guardrails(
    estimate: CostEstimate, guardrails: Guardrails, max_cost_usd: float | None = None
) -> None:
    limit = max_cost_usd if max_cost_usd is not None else guardrails.max_cost_usd
    if limit is None:
        return
    if estimate.usd is None:
        log.warning(
            "Cost limit $%.2f set but no price data is available; cannot enforce. %s",
            limit,
            estimate,
        )
        return
    if estimate.usd > limit:
        raise CostLimitExceededError(estimate.usd, limit)
