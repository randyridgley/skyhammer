import json
from typing import Any

import pytest

from skyhammer.errors import SpendLimitExceededError
from skyhammer.llm import LLM
from skyhammer.pricing import InferencePrices, normalize_model_name, parse_price_row
from skyhammer.session import Session
from tests.conftest import FIXTURES

PRODUCTS = json.loads((FIXTURES / "price_products_sample.json").read_text())


def index() -> InferencePrices:
    return InferencePrices([r for p in PRODUCTS if (r := parse_price_row(p))])


def test_parse_skips_mantle_customization_and_provisioned() -> None:
    usages = {p["product"]["attributes"]["usagetype"]: parse_price_row(p) for p in PRODUCTS}
    assert usages["USE1-qwen.qwen3-32b-mantle-input-tokens-standard"] is None
    assert usages["USE1-NovaMicro-input-tokens-custom-model"] is None
    assert usages["USE1-MP:USE1_Reserved_1Month_InputTPM_Geo-Units"] is None
    flex = usages["USE1-Qwen3-32B-input-tokens-flex"]
    assert flex and flex["tier"] == "flex" and flex["kind"] == "input"
    assert usages["USE1-NovaMicro-input-tokens-batch"]["tier"] == "batch"  # type: ignore[index]


def test_real_prices_from_fixture() -> None:
    p = index()
    nova = p.lookup("Nova Micro")
    assert nova and nova.input == pytest.approx(0.035e-6) and nova.output == pytest.approx(0.14e-6)
    regional = p.lookup("Claude Haiku 4.5")
    glob = p.lookup("Claude Haiku 4.5", routing="global")
    assert regional and glob and regional.input == pytest.approx(1.1e-6)
    assert glob.input == pytest.approx(1.0e-6)  # global is ~10% cheaper than geo/regional
    assert p.lookup("Qwen3 32B (dense)", tier="flex").input < p.lookup("Qwen3 32B (dense)").input  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "model, price, ok",
    [
        ("Nova 2 Lite", "Nova Lite", False),  # regression: wrong-version match
        ("Nova 2 Lite", "Nova 2.0 Lite", True),
        ("Ministral 3 8B", "Ministral 8B 3.0", True),
        ("Pegasus v1.5", "TwelveLabs Pegasus 1.5", True),
        ("GPT-6 Astra", "OpenAI GPT-6 Astra", True),
        ("DeepSeek-R1", "R1", True),
        ("Qwen3 32B (dense)", "Qwen3 32B", True),
        ("NVIDIA Nemotron Nano 9B v2", "NVIDIA Nemotron Nano 2", False),  # size differs
        ("Llama 3.3 70B Instruct", "Llama 3.1 70B", False),
    ],
)
def test_name_matching_never_crosses_versions(model: str, price: str, ok: bool) -> None:
    p = InferencePrices(
        [
            {
                "model": normalize_model_name(price),
                "kind": k,
                "tier": "standard",
                "routing": "regional",
                "usd_per_token": 1e-6,
            }
            for k in ("input", "output")
        ]
    )
    assert (p.match(model) is not None) is ok


def test_ambiguous_match_is_unknown() -> None:
    rows = [
        {
            "model": normalize_model_name(n),
            "kind": "input",
            "tier": "standard",
            "routing": "regional",
            "usd_per_token": 1e-6,
        }
        for n in ("Acme X", "Acme X Fast")
    ]
    assert InferencePrices(rows).match("Acme X Turbo") is None


class FakeRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def converse(self, **req: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1000, "outputTokens": 500, "totalTokens": 1500},
        }


def _llm(
    session: Session, name: str, rt: FakeRuntime, monkeypatch: pytest.MonkeyPatch, **kw: Any
) -> LLM:
    llm = LLM(name, session=session, **kw)
    monkeypatch.setattr(session, "client", lambda service, region=None: rt)
    return llm


def test_cost_per_call_and_session_spend(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    rt = FakeRuntime()
    resp = _llm(session, "nova-micro", rt, monkeypatch).chat.completions.create(
        messages=[{"role": "user", "content": "hi"}]
    )
    assert resp.usage.cost_usd == pytest.approx(1000 * 0.035e-6 + 500 * 0.14e-6)
    glob = _llm(session, "claude-haiku-4-5", rt, monkeypatch, routing="global")
    r2 = glob.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
    assert r2.usage.cost_usd == pytest.approx(1000 * 1.0e-6 + 500 * 5.0e-6)  # global prices
    assert session.spend.total_usd == pytest.approx(resp.usage.cost_usd + r2.usage.cost_usd)
    assert set(session.spend.by_model) == {
        "us.amazon.nova-micro-v1:0",
        "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    }


def test_unpriced_model_is_counted_not_guessed(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    resp = _llm(
        session, "llama3-3-70b-instruct", FakeRuntime(), monkeypatch
    ).chat.completions.create(
        messages=[{"role": "user", "content": "hi"}]
    )  # not in the price fixture
    assert resp.usage.cost_usd is None and session.spend.unpriced_calls == 1


def test_spend_limit_blocks_next_call(settings: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    s = Session(settings.model_copy(update={"max_spend_usd": 0.0001}))
    rt = FakeRuntime()
    llm = _llm(s, "nova-micro", rt, monkeypatch)
    llm.chat.completions.create(messages=[{"role": "user", "content": "hi"}])  # spends $0.000105
    with pytest.raises(SpendLimitExceededError, match="reached the limit"):
        llm.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
    assert rt.calls == 1  # the second call never reached Bedrock
