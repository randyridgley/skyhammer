"""End-to-end training runs (30-60 min each, cents of spend). Resources are always deleted."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skyhammer import LLM, Dataset, FineTuningJob
from skyhammer.session import Session

pytestmark = pytest.mark.slow

DATA = Path(__file__).resolve().parents[2] / "examples" / "data" / "train.jsonl"
PROMPT = [
    {"role": "system", "content": "You are a helpful support agent."},
    {"role": "user", "content": "How do I reset password #7?"},
]


def _train_reattach_deploy_chat(session: Session, base: str, name: str) -> None:
    ds = Dataset.from_file(DATA, name=name)
    job = LLM(base, session=session).create_sft_job(ds, output_model=name, epochs=1)
    # Reattach from scratch, as a different process would.
    again = FineTuningJob.get(name, session=session)
    assert again.name == job.name
    again.wait(poll_seconds=60, timeout_seconds=3 * 3600)
    assert any("step" in line or "TRAINING" in line for line in again.logs(200))
    ft = LLM.get(name, session=session).deploy(wait=True)
    resp = ft.chat.completions.create(messages=PROMPT, max_tokens=40, temperature=0)
    assert resp.choices[0].message.content.strip()
    # Idempotent re-deploy reuses the deployment.
    assert LLM.get(name, session=session).deploy(wait=True).model_id == ft.model_id


def test_nova_sft_end_to_end(live_session: Session, unique_name: str, cleanup: list[str]) -> None:
    cleanup.append(unique_name)
    _train_reattach_deploy_chat(live_session, "nova-micro", unique_name)


def test_qwen_sft_import_end_to_end(
    live_session: Session, unique_name: str, cleanup: list[str]
) -> None:
    cleanup.append(unique_name)
    _train_reattach_deploy_chat(live_session, "qwen3-0.6b", unique_name)


def test_example_data_is_valid() -> None:
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    assert len(rows) >= 50
