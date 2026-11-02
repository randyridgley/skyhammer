"""DA-12: mixing datasets by weight, with the parts kept for lineage."""

from collections import Counter
from typing import Any

import pytest

from skyhammer import Dataset, registry
from skyhammer.errors import SkyhammerError
from skyhammer.jobs import JobRecord
from skyhammer.llm import LLM
from skyhammer.session import Session


def rows(tag: str, n: int) -> Dataset:
    return Dataset.from_records(
        [
            {
                "messages": [
                    {"role": "user", "content": f"{tag}{i}"},
                    {"role": "assistant", "content": "a"},
                ]
            }
            for i in range(n)
        ],
        name=tag,
    )


def origin(ds: Dataset) -> Counter[str]:
    return Counter(r["messages"][0]["content"].rstrip("0123456789") for r in ds.records())


def test_default_size_uses_each_row_at_most_once() -> None:
    support, chat = rows("support", 70), rows("chat", 500)
    mixed = Dataset.mix({support: 0.7, chat: 0.3})
    # support runs out first: 70 rows at 70% -> a 100-row mix.
    assert len(mixed) == 100
    assert origin(mixed) == {"support": 70, "chat": 30}
    assert len({json_row(r) for r in mixed.records()}) == 100  # no repeats
    assert mixed.name == "support+chat"
    assert mixed.mixture == [
        {"name": "support", "sha256": support.sha256, "rows": 70, "share": 0.7},
        {"name": "chat", "sha256": chat.sha256, "rows": 30, "share": 0.3},
    ]


def json_row(r: dict[str, Any]) -> str:
    return r["messages"][0]["content"]


def test_weights_are_relative_and_rounding_adds_up() -> None:
    a, b, c = rows("a", 100), rows("b", 100), rows("c", 100)
    mixed = Dataset.mix([(a, 1), (b, 1), (c, 1)], n=100, name="thirds")
    assert len(mixed) == 100 and mixed.name == "thirds"
    assert sorted(origin(mixed).values()) == [33, 33, 34]
    assert Dataset.mix({a: 7, b: 3}, n=50).mixture == Dataset.mix({a: 0.7, b: 0.3}, n=50).mixture


def test_same_inputs_same_version_and_seed_changes_it() -> None:
    a, b = rows("a", 40), rows("b", 40)
    assert Dataset.mix({a: 1, b: 1}).sha256 == Dataset.mix({a: 1, b: 1}).sha256
    assert Dataset.mix({a: 1, b: 1}).sha256 != Dataset.mix({a: 1, b: 1}, seed=1).sha256
    # Rows are interleaved, not one part after the other.
    first_half = origin(Dataset.from_records(Dataset.mix({a: 1, b: 1}).records()[:40], "x"))
    assert first_half["a"] and first_half["b"]


def test_upsampling_needs_repeat() -> None:
    small, big = rows("small", 10), rows("big", 1000)
    with pytest.raises(SkyhammerError, match=r"needs 50 rows from 'small', which has 10.*repeat"):
        Dataset.mix({small: 0.5, big: 0.5}, n=100)
    mixed = Dataset.mix({small: 0.5, big: 0.5}, n=100, repeat=True)
    assert origin(mixed) == {"small": 50, "big": 50}
    # Every small row used 5 times: whole passes before any row repeats again.
    assert set(
        Counter(json_row(r) for r in mixed.records() if "small" in json_row(r)).values()
    ) == {5}


def test_bad_inputs() -> None:
    a = rows("a", 5)
    with pytest.raises(SkyhammerError, match="at least one"):
        Dataset.mix({})
    with pytest.raises(SkyhammerError, match="non-negative"):
        Dataset.mix({a: -1})
    with pytest.raises(SkyhammerError, match="positive"):
        Dataset.mix({a: 0})
    with pytest.raises(SkyhammerError, match="no rows"):
        Dataset.mix({a: 1, Dataset.from_records([], "empty"): 1})
    # A zero weight leaves a part out entirely.
    assert Dataset.mix({a: 1, rows("b", 5): 0}).mixture == [
        {"name": "a", "sha256": a.sha256, "rows": 5, "share": 1.0}
    ]


def test_jobs_record_the_mixture_and_lineage_follows_it(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    support, chat = rows("support", 160), rows("chat", 400)
    mixed = Dataset.mix({support: 0.8, chat: 0.2}, name="support-mix")
    request, estimate = LLM("qwen3-0.6b", session=session).plan(
        "sft", mixed, output_model="support-qwen-v2", validation_split=0.1
    )
    record = JobRecord.from_request(request, _handle(), estimate)
    assert record.dataset["name"] == "support-mix-train"
    assert record.dataset["mixture"] == mixed.mixture  # survives the validation split

    monkeypatch.setattr(Dataset, "list", staticmethod(lambda session=None: []))
    monkeypatch.setattr(registry, "list_jobs", lambda session, include_deleted: [record])
    (job,) = Dataset.lineage("support", session=object())["jobs"]
    assert (job["model"], job["via"], job["rows"]) == ("support-qwen-v2", "support-mix-train", 160)
    assert job["dataset_sha256"] == support.sha256
    with pytest.raises(SkyhammerError, match="No dataset named"):
        Dataset.lineage("unrelated", session=object())


def _handle() -> Any:
    from skyhammer.backends.base import JobHandle

    return JobHandle("sagemaker", "support-qwen-v2-abc", "arn:job", "us-east-1")
