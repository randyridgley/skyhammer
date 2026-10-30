"""Reward functions for RFT / RLVR / RLAIF.

Three ways to define one:

    @grader                                  # Python function -> SDK-managed Lambda
    def exact_match(sample, output) -> float: ...

    Grader.from_lambda("arn:aws:lambda:...") # bring your own Lambda
    Grader.llm_judge(model="...", rubric="") # model-as-a-judge

Each backend adapts a Grader to its own contract (Bedrock Nova RFT vs SageMaker RLVR).
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import textwrap
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

GradeFn = Callable[[dict[str, Any], str], float]


class GraderKind(StrEnum):
    PYTHON = "python"
    LAMBDA = "lambda"
    LLM_JUDGE = "llm_judge"


@dataclass(frozen=True)
class Grader:
    kind: GraderKind
    name: str
    fn: GradeFn | None = None
    lambda_arn: str | None = None
    judge_model: str | None = None
    rubric: str | None = None
    timeout_s: int = 10
    requirements: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_lambda(cls, arn: str, name: str | None = None) -> Grader:
        return cls(GraderKind.LAMBDA, name or arn.rsplit(":", 1)[-1], lambda_arn=arn)

    @classmethod
    def llm_judge(cls, model: str, rubric: str, name: str = "llm-judge") -> Grader:
        return cls(GraderKind.LLM_JUDGE, name, judge_model=model, rubric=rubric)

    @property
    def source(self) -> str | None:
        return textwrap.dedent(inspect.getsource(self.fn)) if self.fn else None

    def fingerprint(self) -> str:
        parts = [
            self.kind.value,
            self.name,
            self.lambda_arn or "",
            self.judge_model or "",
            self.rubric or "",
            self.source or "",
            ",".join(self.requirements),
        ]
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()

    @classmethod
    def builtin(cls, name: str) -> Grader:
        """Ready-made graders: exact_match, contains, json_valid."""
        try:
            fn = BUILTINS[name]
        except KeyError:
            raise ValueError(
                f"Unknown builtin grader {name!r}; choose from {sorted(BUILTINS)}"
            ) from None
        return cls(GraderKind.PYTHON, name.replace("_", "-"), fn=fn)

    def __call__(self, sample: dict[str, Any], output: str) -> float:
        if self.kind is GraderKind.LAMBDA and self.lambda_arn:
            return _invoke_lambda(self.lambda_arn, sample, output)
        if self.kind is GraderKind.LLM_JUDGE:
            return _judge(self, sample, output)
        if self.fn is None:
            raise TypeError(f"{self.kind.value} graders can't be called locally")
        return self.fn(sample, output)

    def test(self, samples: Iterable[dict[str, Any]], outputs: Iterable[str]) -> list[float]:
        """Run a Python grader locally and check every score is a float in [0, 1]."""
        scores = []
        for s, o in zip(samples, outputs, strict=True):
            score = float(self(s, o))
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"grader {self.name!r} returned {score}, expected [0, 1]")
            scores.append(score)
        return scores

    @property
    def function_name(self) -> str:
        return self.function_name_for("sagemaker")

    def function_name_for(self, contract: str) -> str:
        """Lambda name for a deployed Python grader, versioned by the deployed module (the
        grader's code plus the reward-contract adapter, so adapter fixes redeploy)."""
        code = (lambda_source(self) if self.fn else self.fingerprint()) + contract
        digest = hashlib.sha256(code.encode()).hexdigest()
        suffix = "-br" if contract == "bedrock" else ""
        return f"skyhammer-grader-{self.name}"[:52] + f"{suffix}-{digest[:8]}"

    def deploy(
        self, session: Any = None, *, region: str | None = None, contract: str = "sagemaker"
    ) -> Grader:
        """Deploy a Python grader as a Lambda (idempotent); returns a Lambda grader.

        The function speaks every contract: SageMaker RLVR reward calls (a list of samples
        with ``reference_answer.text``), Bedrock RFT reward calls (a list of samples with
        ``metadata.reference_answer``) and eval calls (``{"sample", "output"}``). SageMaker
        and Bedrock both send a plain list but expect different replies, so ``contract``
        ("sagemaker" or "bedrock") picks the reply shape and gets its own function. Graders
        must be self-contained (standard library only; ``requirements`` aren't packaged yet).
        """
        from skyhammer import tags as tagging
        from skyhammer.session import get_session

        if self.kind is GraderKind.LAMBDA:
            return self
        if self.kind is not GraderKind.PYTHON or self.fn is None:
            raise TypeError("Only Python graders can be deployed; LLM judges run in Bedrock")
        if self.requirements:
            raise NotImplementedError("Packaging grader requirements isn't supported yet")
        if contract not in ("sagemaker", "bedrock"):
            raise ValueError("contract must be 'sagemaker' or 'bedrock'")
        session = session or get_session()
        res = session.resources
        lam = session.client("lambda", region=region)  # same region as the training job
        name = self.function_name_for(contract)
        try:
            arn = lam.get_function(FunctionName=name)["Configuration"]["FunctionArn"]
            return Grader.from_lambda(arn, name=self.name)
        except lam.exceptions.ResourceNotFoundException:
            pass
        if not res.grader_role_arn:
            raise RuntimeError("No grader Lambda role; run `skyhammer bootstrap`")
        arn = lam.create_function(
            FunctionName=name,
            Runtime="python3.12",
            Role=res.grader_role_arn,
            Handler="skyhammer_grader.lambda_handler",
            Code={"ZipFile": _lambda_zip(self)},
            Timeout=max(self.timeout_s, 60),  # RL reward calls send batches of samples
            MemorySize=256,
            Description=f"Skyhammer grader {self.name} ({contract} reward contract)",
            Environment={"Variables": {"SKYHAMMER_REWARD_CONTRACT": contract}},
            Tags=tagging.build_tags({"name": name, "kind": "grader"}, session.settings.tags),
        )["FunctionArn"]
        lam.get_waiter("function_active_v2").wait(FunctionName=name)
        return Grader.from_lambda(arn, name=self.name)

    def delete(self, session: Any = None, *, region: str | None = None) -> None:
        """Delete this grader's Lambdas (deployed Python graders only) in ``region``: the
        training region they were deployed to (default: the session's).

        Jobs with the same grader code share one Lambda, so only delete it once every RL job
        using it has finished; a job still training loses its reward function."""
        from skyhammer.session import get_session

        lam = (session or get_session()).client("lambda", region=region)
        for contract in ("sagemaker", "bedrock"):
            with contextlib.suppress(lam.exceptions.ResourceNotFoundException):
                lam.delete_function(FunctionName=self.function_name_for(contract))


def grader(
    fn: GradeFn | None = None,
    *,
    name: str | None = None,
    timeout_s: int = 10,
    requirements: Iterable[str] = (),
) -> Any:
    def wrap(f: GradeFn) -> Grader:
        return Grader(
            GraderKind.PYTHON,
            name or f.__name__.replace("_", "-"),
            fn=f,
            timeout_s=timeout_s,
            requirements=tuple(requirements),
        )

    return wrap(fn) if fn is not None else wrap


def reference_answer(sample: dict[str, Any]) -> str:
    """The expected answer: an explicit ``answer``/``reference`` field, or the record's
    final assistant message (so SFT-format eval sets work unchanged)."""
    for key in ("answer", "reference", "expected"):
        if isinstance(sample.get(key), str):
            return str(sample[key])
    messages = sample.get("messages") or []
    if messages and messages[-1].get("role") == "assistant":
        content = messages[-1].get("content")
        return content if isinstance(content, str) else ""
    return ""


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _exact_match(sample: dict[str, Any], output: str) -> float:
    return float(_normalize(output) == _normalize(reference_answer(sample)))


def _contains(sample: dict[str, Any], output: str) -> float:
    ref = _normalize(reference_answer(sample))
    return float(bool(ref) and ref in _normalize(output))


def _json_valid(sample: dict[str, Any], output: str) -> float:
    try:
        json.loads(output)
    except ValueError:
        return 0.0
    return 1.0


BUILTINS: dict[str, GradeFn] = {
    "exact_match": _exact_match,
    "contains": _contains,
    "json_valid": _json_valid,
}


def _invoke_lambda(arn: str, sample: dict[str, Any], output: str) -> float:
    """Eval-time contract: payload {"sample", "output"} -> a number or {"score": number}."""
    import boto3

    region = arn.split(":")[3]
    resp = boto3.client("lambda", region_name=region).invoke(
        FunctionName=arn, Payload=json.dumps({"sample": sample, "output": output}).encode()
    )
    result = json.loads(resp["Payload"].read() or b"null")
    if resp.get("FunctionError"):
        raise RuntimeError(f"grader {arn} failed: {result}")
    score = result.get("score") if isinstance(result, dict) else result
    if not isinstance(score, int | float):
        raise RuntimeError(f"grader {arn} returned {result!r}; expected a number or {{'score': n}}")
    return float(score)


# ---- LLM as a judge (EV-5) ----

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "One or two sentences"},
        "score": {"type": "integer", "description": "1 (worst) to 5 (best)"},
    },
    "required": ["reasoning", "score"],
    "additionalProperties": False,
}


def judge_prompt(rubric: str, sample: dict[str, Any], output: str) -> str:
    prompt = [m for m in sample.get("messages") or [] if m.get("role") != "assistant"]
    question = "\n".join(f"{m['role']}: {m.get('content')}" for m in prompt)
    reference = reference_answer(sample)
    parts = [
        "You are grading an AI assistant's answer. Score it from 1 (worst) to 5 (best).",
        f"Rubric:\n{rubric}",
        f"Conversation:\n{question}",
        *([f"Reference answer:\n{reference}"] if reference else []),
        f"Answer to grade:\n{output}",
    ]
    return "\n\n".join(parts)


def _judge(g: Grader, sample: dict[str, Any], output: str) -> float:
    from skyhammer.llm import LLM

    resp = LLM(g.judge_model or "nova-pro").chat.completions.create(
        messages=[{"role": "user", "content": judge_prompt(g.rubric or "", sample, output)}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "grade", "schema": _JUDGE_SCHEMA},
        },
        temperature=0,
        max_tokens=300,
    )
    parsed = resp.choices[0].message.parsed() or {}
    score = min(5, max(1, int(parsed.get("score", 1))))
    return (score - 1) / 4


# ---- Lambda packaging (EV-4) ----

_LAMBDA_ADAPTER = """

import json as _json
import os as _os
import uuid as _uuid


def _skyhammer_output(sample):
    for key in ("model_response", "response", "completion"):
        if isinstance(sample.get(key), str) and sample[key]:
            return sample[key]
    for m in reversed(sample.get("messages") or []):
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return ""


def _skyhammer_reference(sample):
    for ref in (
        sample.get("reference_answer"),
        (sample.get("extra_info") or {}).get("reference_answer"),
        (sample.get("metadata") or {}).get("reference_answer"),  # Bedrock RFT
    ):
        if isinstance(ref, dict) and isinstance(ref.get("text"), str):
            return ref["text"]
        if isinstance(ref, str):
            return ref
        if isinstance(ref, dict):
            return _json.dumps(ref)
    return (sample.get("reward_model") or {}).get("ground_truth") or ""


def _skyhammer_score(sample):
    view = {**(sample.get("metadata") or {}), **sample}  # Bedrock puts custom fields here
    ref = _skyhammer_reference(sample)
    if ref and "answer" not in view:
        view["answer"] = ref
    score = float(__FN__(view, _skyhammer_output(sample)))
    return {
        "id": sample.get("id") if isinstance(sample.get("id"), str) else str(_uuid.uuid4()),
        "aggregate_reward_score": score,
        "metrics_list": [{"name": "__NAME__", "value": score, "type": "Reward"}],
    }


def lambda_handler(event, context):
    if isinstance(event, dict) and "sample" in event and "output" in event:  # skyhammer eval
        return {"score": float(__FN__(event["sample"], event["output"]))}
    if _os.environ.get("SKYHAMMER_REWARD_CONTRACT") == "bedrock":
        if isinstance(event, dict):
            event = event.get("trajectories", [event])
        return [_skyhammer_score(s) for s in event]  # Bedrock RFT: a plain list back
    raw = event.get("body", event) if isinstance(event, dict) else event
    body = _json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(body, dict):
        body = body.get("batch", [body])
    results = [_skyhammer_score(s) for s in body]
    return {"statusCode": 200, "body": _json.dumps(results)}
"""


def lambda_source(g: Grader) -> str:
    """The deployed module: helpers, the grader's own source, and the reward-contract adapter.

    ``reference_answer`` is included so graders can use it without importing skyhammer.
    """
    assert g.fn is not None
    helpers = "".join(inspect.getsource(h) + "\n\n" for h in (reference_answer, _normalize))
    src = textwrap.dedent(inspect.getsource(g.fn))
    src = "\n".join(line for line in src.splitlines() if not line.startswith("@"))
    head = "from typing import Any\nimport json\n\n\n"
    adapter = _LAMBDA_ADAPTER.replace("__FN__", g.fn.__name__).replace("__NAME__", g.name)
    return head + helpers + src + adapter


def _lambda_zip(g: Grader) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("skyhammer_grader.py", date_time=(2020, 1, 1, 0, 0, 0))
        info.external_attr = 0o644 << 16
        z.writestr(info, lambda_source(g))
    return buf.getvalue()
