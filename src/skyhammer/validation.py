"""Model-aware dataset validation with explicit, opt-in fixes.

Validation never mutates silently: ``validate`` reports issues and the fixes it *would*
apply; ``apply_fixes`` performs exactly those fixes and records them in the report.

Fixes, tried in order for each bad chat row:

- ``REPAIR_ROLES``: map role aliases (``human``/``gpt``/``bot``/``model``), drop empty
  messages and leading assistant turns, merge consecutive same-role messages.
- ``TRUNCATE``: drop the oldest user/assistant exchanges (keeping the system prompt and
  the final exchange) until the row fits the model's token limit.
- ``DROP_ROW``: anything else that can't be repaired.
"""

from __future__ import annotations

import copy
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
    REPAIR_ROLES = "repair_roles"
    TRUNCATE = "truncate"


@dataclass(frozen=True)
class Issue:
    line: int | None  # 1-based line number; None for dataset-level issues
    severity: Severity
    message: str
    fix: Fix | None = None
    # The fixed row for REPAIR_ROLES / TRUNCATE; ``apply_fixes`` uploads it instead.
    replacement: Record | None = field(default=None, compare=False, repr=False)


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

    def _repr_html_(self) -> str:
        import html

        approx = "" if self.exact_tokens else "~"
        head = (
            f"<p>{self.total_rows} rows, {approx}{self.estimated_tokens:,} tokens: "
            f"<b>{len(self.errors)} errors</b>, {len(self.warnings)} warnings"
            f"{'' if self.ok else (' (auto-fixable)' if self.fixable else '')}</p>"
        )
        rows = "".join(
            f"<tr><td>{i.line or 'dataset'}</td><td>{i.severity.value}</td>"
            f"<td>{html.escape(i.message)}</td><td>{i.fix.value if i.fix else ''}</td></tr>"
            for i in self.issues[:50]
        )
        table = (
            "<table><tr><th>line</th><th>severity</th><th>problem</th><th>fix</th></tr>"
            f"{rows}</table>"
            if rows
            else ""
        )
        return head + table

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


# Role names used by ShareGPT, Gemini and similar exports.
_ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "ai": "assistant",
}


def _as_parts(content: Any) -> list[Any]:
    return content if isinstance(content, list) else [{"type": "text", "text": str(content)}]


def _merge_content(a: Any, b: Any) -> Any:
    if isinstance(a, list) or isinstance(b, list):
        return [*_as_parts(a), *_as_parts(b)]
    return f"{a}\n\n{b}"


def _repair_messages(messages: Any) -> list[Any] | None:
    """Return a repaired copy of a chat ``messages`` list, or None if nothing changed."""
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        return None
    out: list[dict[str, Any]] = []
    for m in copy.deepcopy(messages):
        role = m.get("role")
        if isinstance(role, str):
            m["role"] = _ROLE_ALIASES.get(role.strip().lower(), role.strip().lower())
        if not message_text(m.get("content")).strip():
            continue
        if m["role"] == "assistant" and not any(o["role"] == "user" for o in out):
            continue  # a conversation can't open with the assistant
        if out and out[-1]["role"] == m["role"] and m["role"] in ("user", "assistant"):
            out[-1]["content"] = _merge_content(out[-1]["content"], m["content"])
            continue
        out.append(m)
    return None if out == messages else out


def _repair_roles(record: Any, technique: Technique) -> Record | None:
    if not isinstance(record, dict) or technique in (Technique.CPT, Technique.DPO):
        return None
    fixed = _repair_messages(record.get("messages"))
    return None if fixed is None else {**record, "messages": fixed}


def _truncate(
    record: Record, technique: Technique, limit: int, count: TokenCounter
) -> Record | None:
    """Drop the oldest exchanges until ``record`` fits ``limit``; None if it can't."""
    if technique in (Technique.CPT, Technique.DPO):
        return None
    msgs = record["messages"]
    head = msgs[:1] if msgs[0].get("role") == "system" else []
    turns = msgs[len(head) :]
    while len(turns) > 2:
        turns = turns[2:]  # one user + assistant pair; keeps the user/assistant alternation
        candidate = {**record, "messages": [*head, *turns]}
        if count(_record_text(candidate, technique)) <= limit:
            return candidate
    return None


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
        fixed: Record | None = None
        if problems:
            repaired = _repair_roles(record, technique)
            if repaired is None or _check_record(repaired, technique):
                report.issues.append(
                    Issue(line, Severity.ERROR, "; ".join(dict.fromkeys(problems)), Fix.DROP_ROW)
                )
                continue
            record = fixed = repaired
            report.issues.append(
                Issue(
                    line,
                    Severity.ERROR,
                    f"{'; '.join(dict.fromkeys(problems))} (repairable: merge/rename roles)",
                    Fix.REPAIR_ROLES,
                    replacement=fixed,
                )
            )
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
            message = (
                f"{approx}{tokens:,} tokens exceeds model limit {limits.max_tokens_per_sample:,}"
            )
            truncated = _truncate(record, technique, limits.max_tokens_per_sample, count)
            if truncated is None:
                report.issues.append(Issue(line, Severity.ERROR, message, Fix.DROP_ROW))
                continue
            dropped = len(record["messages"]) - len(truncated["messages"])
            record = fixed = truncated
            tokens = count(_record_text(record, technique))
            report.issues.append(
                Issue(
                    line,
                    Severity.ERROR,
                    f"{message} (truncatable: drop {dropped} oldest messages)",
                    Fix.TRUNCATE,
                    replacement=fixed,
                )
            )
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
    """Apply the report's row fixes (drop, repair, truncate); return the records to upload."""
    drop = {i.line for i in report.issues if i.fix is Fix.DROP_ROW}
    # A row can be repaired and then truncated; the last replacement is the final row.
    replace = {i.line: i.replacement for i in report.issues if i.replacement is not None}
    report.applied_fixes = [i for i in report.issues if i.line is not None and i.fix]
    return [replace.get(line, r) for line, r in rows if line not in drop]
