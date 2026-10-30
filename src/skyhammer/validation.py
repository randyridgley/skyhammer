"""Model-aware dataset validation with explicit, opt-in fixes.

Validation never mutates silently: ``validate`` reports issues and the fixes it *would*
apply; ``apply_fixes`` performs exactly those fixes and records them in the report.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from skyhammer.catalog import DatasetLimits, Technique
from skyhammer.formats.converters import message_text
from skyhammer.tokenization import TokenCounter, estimate_tokens

__all__ = ["estimate_tokens"]

Record = dict[str, Any]


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class Fix(StrEnum):
    DROP_ROW = "drop_row"


@dataclass(frozen=True)
class Issue:
    line: int | None  # 1-based line number; None for dataset-level issues
    severity: Severity
    message: str
    fix: Fix | None = None


@dataclass
class ValidationReport:
    technique: Technique
    total_rows: int = 0
    issues: list[Issue] = field(default_factory=list)
    applied_fixes: list[Issue] = field(default_factory=list)
    estimated_tokens: int = 0  # exact when ``exact_tokens`` (counted by the model's tokenizer)
    max_sample_tokens: int = 0
    limits: DatasetLimits | None = None
    exact_tokens: bool = False

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def fixable(self) -> bool:
        """True if every error has a fix and applying them leaves a valid dataset."""
        return all(i.fix is not None for i in self.errors)

    def summary(self, limit: int = 10) -> str:
        lines = [
            f"{self.total_rows} rows, {'' if self.exact_tokens else '~'}"
            f"{self.estimated_tokens:,} tokens; "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings"
        ]
        for i in self.issues[:limit]:
            where = f"line {i.line}" if i.line else "dataset"
            fix = f" [fix: {i.fix.value}]" if i.fix else ""
            lines.append(f"  {i.severity.value}: {where}: {i.message}{fix}")
        if len(self.issues) > limit:
            lines.append(f"  ... and {len(self.issues) - limit} more")
        return "\n".join(lines)


def _check_chat(messages: Any, *, require_final: str) -> list[str]:
    """Return a list of problems with an OpenAI ``messages`` list."""
    if not isinstance(messages, list) or not messages:
        return ["'messages' must be a non-empty list"]
    problems: list[str] = []
    turns = list(messages)
    if isinstance(turns[0], dict) and turns[0].get("role") == "system":
        turns = turns[1:]
    expected = "user"
    for idx, m in enumerate(turns):
        if not isinstance(m, dict):
            return ["each message must be an object"]
        role = m.get("role")
        if role == "system":
            problems.append("system message is only allowed first")
        elif role not in ("user", "assistant"):
            problems.append(f"unsupported role {role!r}")
        elif role != expected:
            problems.append(f"message {idx + 1}: expected role {expected!r}, got {role!r}")
            break
        if not message_text(m.get("content")).strip():
            problems.append(f"message {idx + 1}: empty content")
        expected = "assistant" if expected == "user" else "user"
    if not turns:
        problems.append("conversation has no user/assistant turns")
    elif not problems and turns[-1].get("role") != require_final:
        problems.append(f"conversation must end with a {require_final!r} message")
    return problems


def _record_text(record: Record, technique: Technique) -> str:
    if technique is Technique.CPT:
        return str(record.get("text", ""))
    if technique is Technique.DPO:
        parts = [
            *record["input"]["messages"],
            *record["preferred_output"],
            *record["non_preferred_output"],
        ]
    else:
        parts = record["messages"]
    return "".join(message_text(m.get("content")) for m in parts)


def _check_record(record: Any, technique: Technique) -> list[str]:
    if not isinstance(record, dict):
        return ["row must be a JSON object"]
    if technique is Technique.CPT:
        text = record.get("text")
        return (
            [] if isinstance(text, str) and text.strip() else ["'text' must be a non-empty string"]
        )
    if technique is Technique.DPO:
        inp = record.get("input")
        if not isinstance(inp, dict):
            return ["'input' must be an object with 'messages'"]
        problems = _check_chat(inp.get("messages"), require_final="user")
        for key in ("preferred_output", "non_preferred_output"):
            out = record.get(key)
            if (
                not isinstance(out, list)
                or not out
                or not isinstance(out[-1], dict)
                or out[-1].get("role") != "assistant"
                or not message_text(out[-1].get("content")).strip()
            ):
                problems.append(f"'{key}' must be a non-empty list ending in an assistant message")
        return problems
    # RFT rows are prompts; the model generates the completion during training.
    final = "user" if technique is Technique.RFT else "assistant"
    if technique is Technique.DISTILLATION:
        # Distillation accepts prompt-only rows (teacher generates) or full conversations.
        msgs = record.get("messages")
        last = (
            msgs[-1].get("role")
            if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict)
            else None
        )
        final = "assistant" if last == "assistant" else "user"
    return _check_chat(record.get("messages"), require_final=final)


def parse_jsonl(lines: Iterable[str]) -> list[tuple[int, Any]]:
    """Parse JSONL into (line_number, value) pairs; invalid JSON yields a ``_ParseError``."""
    out: list[tuple[int, Any]] = []
    for n, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            out.append((n, json.loads(raw)))
        except json.JSONDecodeError as e:
            out.append((n, _ParseError(str(e))))
    return out


@dataclass(frozen=True)
class _ParseError:
    message: str


def validate(
    rows: list[tuple[int, Any]],
    technique: Technique,
    limits: DatasetLimits,
    count_tokens: TokenCounter | None = None,
) -> ValidationReport:
    """``count_tokens`` is the model's tokenizer; without it counts are estimated."""
    report = ValidationReport(technique=technique, total_rows=len(rows), limits=limits)
    report.exact_tokens = count_tokens is not None
    count = count_tokens or estimate_tokens
    approx = "" if count_tokens else "~"
    valid_rows = 0
    for line, record in rows:
        if isinstance(record, _ParseError):
            report.issues.append(
                Issue(line, Severity.ERROR, f"invalid JSON: {record.message}", Fix.DROP_ROW)
            )
            continue
        problems = _check_record(record, technique)
        if problems:
            report.issues.append(Issue(line, Severity.ERROR, "; ".join(problems), Fix.DROP_ROW))
            continue
        text = _record_text(record, technique)
        reserved = [t for t in limits.reserved_terms if t in text]
        if reserved:
            report.issues.append(
                Issue(
                    line,
                    Severity.ERROR,
                    f"contains reserved term(s) {', '.join(map(repr, reserved))} that "
                    "fail training for this model; rephrase them",
                    Fix.DROP_ROW,
                )
            )
            continue
        tokens = count(text)
        if tokens > limits.max_tokens_per_sample:
            report.issues.append(
                Issue(
                    line,
                    Severity.ERROR,
                    f"{approx}{tokens:,} tokens exceeds model limit "
                    f"{limits.max_tokens_per_sample:,}",
                    Fix.DROP_ROW,
                )
            )
            continue
        valid_rows += 1
        report.estimated_tokens += tokens
        report.max_sample_tokens = max(report.max_sample_tokens, tokens)

    if valid_rows < limits.min_samples:
        report.issues.append(
            Issue(
                None,
                Severity.ERROR,
                f"{valid_rows} valid rows; at least {limits.min_samples} required",
            )
        )
    elif valid_rows > limits.max_samples:
        report.issues.append(
            Issue(None, Severity.ERROR, f"{valid_rows} rows exceeds maximum {limits.max_samples:,}")
        )
    if report.errors and valid_rows:
        dropped = sum(1 for i in report.errors if i.fix is Fix.DROP_ROW)
        if dropped / report.total_rows > 0.1:
            report.issues.append(
                Issue(
                    None,
                    Severity.WARNING,
                    f"{dropped} of {report.total_rows} rows invalid (>10%); "
                    "check your export before auto-fixing",
                )
            )
    return report


def apply_fixes(rows: list[tuple[int, Any]], report: ValidationReport) -> list[Record]:
    """Drop rows the report marked ``DROP_ROW``; return the remaining records."""
    drop = {i.line for i in report.issues if i.fix is Fix.DROP_ROW}
    report.applied_fixes = [i for i in report.issues if i.fix is Fix.DROP_ROW]
    return [r for line, r in rows if line not in drop]
