"""DX-12: training metrics, progress lines and notebook views."""

import io
from pathlib import Path
from typing import Any

from skyhammer.backends.bedrock import BedrockNativeBackend
from skyhammer.cost import CostEstimate
from skyhammer.jobs import JobRecord
from skyhammer.progress import html_table, loss_series, sparkline, svg_sparkline
from skyhammer.session import Session

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_sparklines() -> None:
    assert sparkline([4.0, 2.0, 1.0, 0.5]) == "█▄▂▁"
    assert sparkline([1.0, 1.0]) == "▁▁"  # flat line, no division by zero
    assert sparkline([]) == ""
    assert svg_sparkline([1.0]) == ""  # need two points for a line
    assert "<polyline" in svg_sparkline([3.0, 1.0, 2.0])


def test_loss_series_reads_each_backends_key() -> None:
    rows = [{"step": 1, "training_loss": 2.0}, {"step": 2, "gpt_loss": 1.0}, {"step": 3, "lr": 0.1}]
    assert loss_series(rows) == [2.0, 1.0]  # Bedrock, SageMaker; rows without loss skipped


def test_html_table_escapes_values_but_not_trusted_svg() -> None:
    out = html_table([("job", "<script>"), ("curve", "<svg></svg>")], raw={"curve"})
    assert "&lt;script&gt;" in out and "<svg></svg>" in out


def test_cost_estimate_html() -> None:
    assert "$80.00/hour" in CostEstimate(1, None, usd_per_hour=80.0)._repr_html_()


class FakeS3:
    """Serves the real step_wise_training_metrics.csv from a Nova Micro job (2026-10-30)."""

    KEY = (
        "jobs/j/output/model-customization-job-x/training_artifacts/step_wise_training_metrics.csv"
    )

    def get_paginator(self, op: str) -> Any:
        key = self.KEY

        class P:
            def paginate(self, **_: Any) -> list[dict[str, Any]]:
                return [{"Contents": [{"Key": key}, {"Key": "jobs/j/output/model.tar"}]}]

        return P()

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        assert Key == self.KEY
        return {"Body": io.BytesIO((FIXTURES / "bedrock_step_metrics.csv").read_bytes())}


def test_bedrock_metrics_from_recorded_csv(session: Session, monkeypatch: Any) -> None:
    real = session.client
    monkeypatch.setattr(
        session, "client", lambda svc, region=None: FakeS3() if svc == "s3" else real(svc, region)
    )
    record = JobRecord(
        job_name="j",
        job_id="arn",
        backend="bedrock",
        region="us-east-1",
        technique="sft",
        base_model="nova-micro",
        backend_model_id="m",
        serving="bedrock",
        output_model="o",
        training_type="lora",
        fingerprint="f",
    )
    rows = BedrockNativeBackend(session).metrics(record)
    assert [r["step"] for r in rows] == [1, 2, 3, 4, 5, 6]
    losses = loss_series(rows)
    assert round(losses[0], 3) == 3.963 and round(losses[-1], 3) == 0.539
    assert sparkline(losses) == "█▅▃▂▁▁"
