"""Eval results as flat rows, shared by the experiment-tracking integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skyhammer.errors import SkyhammerError
from skyhammer.evaluation import EvalReport, EvalResult
from skyhammer.formats import message_text


@dataclass
class EvalRow:
    prompt: Any  # the messages sent (local eval) or the prompt text (managed eval job)
    output: str | None
    expected: str | None = None
    category: str | None = None
    scores: dict[str, float | None] = field(default_factory=dict)
    comments: dict[str, str] = field(default_factory=dict)


def eval_rows(
    result: EvalResult | EvalReport, records: list[dict[str, Any]] | None = None
) -> list[EvalRow]:
    """One row per sample. ``records`` (the rows ``evaluate`` ran) adds each prompt and
    reference answer to a local ``EvalResult``; a managed ``EvalReport`` carries its own."""
    if isinstance(result, EvalReport):
        return [
            EvalRow(r.prompt, r.response, None, r.category, dict(r.scores), dict(r.explanations))
            for r in result.records
        ]
    if records is not None and len(records) != len(result.outputs):
        raise SkyhammerError(
            f"records has {len(records)} rows but the result has {len(result.outputs)} outputs"
        )
    rows = []
    for i, output in enumerate(result.outputs):
        prompt: Any = None
        expected = None
        if records is not None:
            messages = records[i]["messages"]
            if messages and messages[-1]["role"] == "assistant":
                expected = message_text(messages[-1]["content"])
                messages = messages[:-1]
            prompt = messages
        scores: dict[str, float | None] = {k: v[i] for k, v in result.scores.items()}
        rows.append(EvalRow(prompt, output, expected, None, scores))
    return rows
