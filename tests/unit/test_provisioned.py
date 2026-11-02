"""SV-10: Provisioned Throughput prices, budget checks, purchase/swap/delete, deploy."""

import datetime as dt
from typing import Any

import pytest
from typer.testing import CliRunner

from skyhammer import LLM, CostLimitExceededError, SkyhammerError
from skyhammer.cli.main import app
from skyhammer.inference import Usage
from skyhammer.jobs import ModelRef
from skyhammer.provisioned import (
    Provisioned,
    ProvisionedThroughput,
    check_budget,
    commitment_term,
    effective_status,
    price,
)
from skyhammer.session import Session

CUSTOM = "arn:aws:bedrock:us-east-1:123456789012:custom-model/amazon.nova-micro-v1:0:128k/abc"
CUSTOM_V2 = CUSTOM.replace("abc", "def")
TAGS = {"skyhammer:name": "support-nova"}
PT_ARN = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/pt1"


# ---- prices (real Price List rows, us-east-1) ----


def test_prices_per_term_from_the_price_list(session: Session) -> None:
    assert price(session, "Nova Micro").usd_per_hour == 60.5
    assert price(session, "Nova Micro", commitment="1_month").usd_per_hour == 55.0
    six = price(session, "Nova Micro", model_units=2, commitment="6m")
    assert six.usd_per_hour == 60.5 and six.term_usd == pytest.approx(60.5 * 4380)
    # Nova 2 Lite and Nova Lite are different price rows; a version never matches loosely.
    assert price(session, "Nova 2 Lite").usd_per_hour == 60.5
    assert price(session, "Claude Sonnet 4.6").usd_per_hour is None
    # Fine-tunes cost what their base costs unless the Price List has a Custom row.
    assert price(session, "Nova Pro", custom=True).usd_per_hour == 60.5
    premier = price(session, "Titan Text G1 - Premier", custom=True)
    assert premier.usd_per_hour == 32.25
    assert "$60.50/hour" in str(price(session, "Nova Lite"))
    assert "not cancellable" in str(price(session, "Nova Lite", commitment="1_month"))


def test_commitment_names() -> None:
    assert commitment_term(None) is None and commitment_term("none") is None
    assert commitment_term("1_month") == commitment_term("OneMonth") == "OneMonth"
    assert commitment_term("6_months") == "SixMonths"
    with pytest.raises(SkyhammerError, match="commitment must be"):
        commitment_term("forever")


def test_budget_must_be_explicit(session: Session) -> None:
    hourly = price(session, "Nova Micro")
    with pytest.raises(SkyhammerError, match=r"\$60.50/hour.*max_hourly_usd"):
        check_budget(hourly, None, accept_commitment=False)
    with pytest.raises(CostLimitExceededError, match="Raise max_hourly_usd"):
        check_budget(hourly, 50, accept_commitment=False)
    check_budget(hourly, 61, accept_commitment=False)
    monthly = price(session, "Nova Micro", commitment="1_month")
    with pytest.raises(SkyhammerError, match="accept_commitment=True"):
        check_budget(monthly, 100, accept_commitment=False)
    check_budget(monthly, 100, accept_commitment=True)
    with pytest.raises(SkyhammerError, match="can't price"):
        check_budget(price(session, "Claude Sonnet 4.6"), 100, accept_commitment=False)


# ---- the resource ----


class FakeBedrock:
    def __init__(self, existing: list[dict[str, Any]] | None = None) -> None:
        self.pts = {p["provisionedModelArn"]: p for p in existing or []}
        self.tags: dict[str, dict[str, str]] = {
            arn: {"skyhammer:name": "x"} for arn in self.pts if "theirs" not in arn
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _pt(self, ident: str) -> dict[str, Any]:
        for p in self.pts.values():
            if ident in (p["provisionedModelArn"], p["provisionedModelName"]):
                return p
        from botocore.exceptions import ClientError

        raise ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "Get")

    def list_provisioned_model_throughputs(self, **kw: Any) -> dict[str, Any]:
        name = kw.get("nameContains", "")
        return {
            "provisionedModelSummaries": [
                p for p in self.pts.values() if name in p["provisionedModelName"]
            ]
        }

    def get_provisioned_model_throughput(self, provisionedModelId: str) -> dict[str, Any]:
        return self._pt(provisionedModelId)

    def list_tags_for_resource(self, resourceARN: str) -> dict[str, Any]:
        return {"tags": [{"key": k, "value": v} for k, v in self.tags.get(resourceARN, {}).items()]}

    def create_provisioned_model_throughput(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("create", kw))
        arn = f"{PT_ARN[:-3]}{kw['provisionedModelName']}"
        self.pts[arn] = pt(kw["provisionedModelName"], arn, kw["modelId"], "Creating")
        self.tags[arn] = {t["key"]: t["value"] for t in kw["tags"]}
        return {"provisionedModelArn": arn}

    def update_provisioned_model_throughput(self, **kw: Any) -> None:
        self.calls.append(("update", kw))
        self._pt(kw["provisionedModelId"])["desiredModelArn"] = kw["desiredModelId"]

    def delete_provisioned_model_throughput(self, provisionedModelId: str) -> None:
        self.calls.append(("delete", {"id": provisionedModelId}))
        self.pts.pop(self._pt(provisionedModelId)["provisionedModelArn"])

    def list_foundation_models(self, byInferenceType: str) -> dict[str, Any]:
        ids = [
            "amazon.nova-micro-v1:0:24k",
            "amazon.nova-micro-v1:0:128k",
            "amazon.nova-lite-v1:0:300k",
        ]
        return {"modelSummaries": [{"modelId": i} for i in ids]}


def pt(name: str, arn: str, model: str, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "provisionedModelName": name,
        "provisionedModelArn": arn,
        "modelArn": model,
        "status": status,
        "modelUnits": 1,
        **extra,
    }


@pytest.fixture
def fake(session: Session, monkeypatch: pytest.MonkeyPatch) -> FakeBedrock:
    br = FakeBedrock()
    monkeypatch.setattr("skyhammer.registry.update_model", lambda *a, **kw: None)  # no S3
    real = session.client
    monkeypatch.setattr(
        session,
        "client",
        lambda service, region=None: br if service == "bedrock" else real(service, region),
    )
    return br


def test_purchase_is_idempotent_and_swaps_models(session: Session, fake: FakeBedrock) -> None:
    p = Provisioned(session)
    bought = p.purchase(
        "support-pt", CUSTOM, model_units=1, commitment=None, tags={"skyhammer:name": "v1"}
    )
    assert bought.status == "Creating"
    assert fake.calls[0][1]["modelUnits"] == 1 and "commitmentDuration" not in fake.calls[0][1]
    fake.pts[bought.arn]["status"] = "InService"

    assert (
        p.purchase("support-pt", CUSTOM, model_units=1, commitment=None, tags=TAGS).arn
        == bought.arn
    )
    assert len(fake.calls) == 1  # reused, nothing bought
    with pytest.raises(SkyhammerError, match="can't change that in place"):
        p.purchase("support-pt", CUSTOM, model_units=2, commitment=None, tags=TAGS)

    swapped = p.purchase("support-pt", CUSTOM_V2, model_units=1, commitment=None, tags=TAGS)
    assert fake.calls[-1] == (
        "update",
        {"provisionedModelId": bought.arn, "desiredModelId": CUSTOM_V2},
    )
    assert effective_status(swapped) == "Updating"  # until the new model is actually served
    fake.pts[bought.arn]["modelArn"] = CUSTOM_V2
    assert p.wait(bought.arn, poll_seconds=0).model_arn == CUSTOM_V2


def test_failed_purchases_are_replaced_and_foreign_names_refused(
    session: Session, fake: FakeBedrock
) -> None:
    fake.pts["arn:failed"] = pt("support-pt", "arn:failed", CUSTOM, "Failed")
    fake.pts["arn:theirs"] = pt("their-pt", "arn:theirs", CUSTOM, "InService")
    p = Provisioned(session)
    p.purchase("support-pt", CUSTOM, model_units=2, commitment="OneMonth", tags=TAGS)
    assert [c[0] for c in fake.calls] == ["delete", "create"]
    assert fake.calls[1][1]["commitmentDuration"] == "OneMonth"
    with pytest.raises(SkyhammerError, match="wasn't created by Skyhammer"):
        p.purchase("their-pt", CUSTOM, model_units=1, commitment=None, tags=TAGS)
    with pytest.raises(SkyhammerError, match="wasn't created by Skyhammer"):
        p.delete("their-pt")
    with pytest.raises(SkyhammerError, match="at least 1"):
        p.purchase("x", CUSTOM, model_units=0, commitment=None, tags=TAGS)
    with pytest.raises(SkyhammerError, match="skyhammer:name"):
        p.purchase("x", CUSTOM, model_units=1, commitment=None, tags={})
    assert [n.name for n, _ in p.tagged()] == ["support-pt"]


def test_delete_respects_commitments(session: Session, fake: FakeBedrock) -> None:
    later = dt.datetime.now(dt.UTC) + dt.timedelta(days=20)
    fake.pts[PT_ARN] = pt("held", PT_ARN, CUSTOM, "InService", commitmentExpirationTime=later)
    fake.tags[PT_ARN] = {"skyhammer:name": "x"}
    with pytest.raises(SkyhammerError, match="committed until"):
        Provisioned(session).delete("held")
    fake.pts[PT_ARN]["commitmentExpirationTime"] = later - dt.timedelta(days=30)
    Provisioned(session).delete("held")
    assert fake.calls[-1] == ("delete", {"id": PT_ARN})


def test_failures_surface_with_bedrocks_message(session: Session, fake: FakeBedrock) -> None:
    fake.pts[PT_ARN] = pt("bad", PT_ARN, CUSTOM, "Failed", failureMessage="Insufficient quota")
    with pytest.raises(Exception, match="Insufficient quota"):
        Provisioned(session).wait(PT_ARN, poll_seconds=0)
    with pytest.raises(SkyhammerError, match="No Provisioned Throughput"):
        Provisioned(session).get("missing")


# ---- LLM.deploy(type="provisioned") ----


def nova_ft(session: Session, api: str = "converse", backend: str = "bedrock") -> LLM:
    return LLM.from_model_ref(
        ModelRef("support-nova", CUSTOM, backend, "nova-micro", "us-east-1", api=api), session
    )


def test_deploy_fine_tune_checks_budget_before_buying(session: Session, fake: FakeBedrock) -> None:
    llm = nova_ft(session)
    with pytest.raises(SkyhammerError, match=r"\$60.50/hour"):
        llm.deploy(type="provisioned")
    with pytest.raises(CostLimitExceededError):
        llm.deploy(type="provisioned", model_units=2, max_hourly_usd=100)
    assert fake.calls == []  # nothing bought

    llm.deploy(type="provisioned", max_hourly_usd=61, wait=False)
    _, kw = fake.calls[0]
    assert kw["modelId"] == CUSTOM and kw["provisionedModelName"] == "support-nova"
    tags = {t["key"]: t["value"] for t in kw["tags"]}
    assert tags["skyhammer:name"] == "support-nova" and tags["cost-center"] == "1234"
    assert llm.is_provisioned and llm.model_id.endswith("provisioned-model/support-nova")

    fake.pts[llm.model_id]["status"] = "InService"
    assert llm.deployment is not None
    assert llm.deployment.wait(poll_seconds=0).ref.status == "InService"

    usage = Usage(1000, 1000, 2000)
    llm._record_usage(llm.model_id, usage, None)
    assert usage.cost_usd is None  # billed per model-unit hour, not per token
    assert session.spend.total_usd == 0


def test_deploy_base_model_picks_the_longest_context(session: Session, fake: FakeBedrock) -> None:
    llm = LLM("nova-micro", session=session)
    llm.deploy(type="provisioned", max_hourly_usd=61, wait=False, name="micro-pt")
    _, kw = fake.calls[0]
    assert kw["modelId"] == "amazon.nova-micro-v1:0:128k"
    assert {t["key"]: t["value"] for t in kw["tags"]}["skyhammer:base-model"] == "nova-micro"

    again = LLM.provisioned("micro-pt", session=session)
    assert again.name == "nova-micro" and again.model_id == llm.model_id
    with pytest.raises(SkyhammerError, match="No Skyhammer Provisioned Throughput"):
        LLM.provisioned("other", session=session)


def test_only_bedrock_custom_models_can_be_provisioned(session: Session, fake: FakeBedrock) -> None:
    for llm in (nova_ft(session, api="mantle"), nova_ft(session, api="openai_chat")):
        with pytest.raises(SkyhammerError, match="isn't a Bedrock custom model"):
            llm.deploy(type="provisioned", max_hourly_usd=100)
    with pytest.raises(SkyhammerError, match="accept_commitment"):
        nova_ft(session).deploy(type="provisioned", commitment="6_months", max_hourly_usd=100)


def test_price_cli(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("skyhammer.cli.main.Session", lambda: session)
    monkeypatch.setattr("skyhammer.cli.main._resolve_llm", lambda name: LLM(name, session=session))
    out = CliRunner().invoke(app, ["provisioned", "price", "nova-micro", "--units", "2"])
    assert out.exit_code == 0, out.output
    assert "$121.00" in out.output and "$110.00" in out.output and "$60.50" in out.output
    assert "$264,990" in out.output  # 6 months: 2 x $30.25 x 4380 h


def test_provisioned_throughput_from_api() -> None:
    p = ProvisionedThroughput.from_api(
        pt(
            "a",
            PT_ARN,
            CUSTOM,
            "Updating",
            desiredModelArn=CUSTOM_V2,
            commitmentDuration="OneMonth",
        )
    )
    assert (p.commitment, p.desired_model_arn, effective_status(p)) == (
        "OneMonth",
        CUSTOM_V2,
        "Updating",
    )
