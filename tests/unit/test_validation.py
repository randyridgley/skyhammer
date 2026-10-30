import json

from skyhammer.catalog import DatasetLimits, Technique
from skyhammer.validation import Fix, apply_fixes, parse_jsonl, validate
from tests.conftest import chat

LIMITS = DatasetLimits(min_samples=2, max_samples=100, max_tokens_per_sample=50)


def rows(*records: object) -> list[tuple[int, object]]:
    return parse_jsonl(json.dumps(r) if not isinstance(r, str) else r for r in records)


def test_valid_sft() -> None:
    report = validate(rows(chat(), chat(system="s")), Technique.SFT, LIMITS)
    assert report.ok, report.summary()
    assert report.estimated_tokens > 0


def test_invalid_json_and_roles_are_droppable() -> None:
    data = rows(
        chat(),
        chat(),
        "{not json",
        {"messages": [{"role": "assistant", "content": "x"}]},
        {"messages": [{"role": "user", "content": "q"}]},
    )  # no final assistant
    report = validate(data, Technique.SFT, LIMITS)
    assert [i.line for i in report.errors] == [3, 4, 5]
    assert all(i.fix is Fix.DROP_ROW for i in report.errors)
    assert report.fixable
    fixed = apply_fixes(data, report)
    assert len(fixed) == 2
    assert len(report.applied_fixes) == 3


def test_too_few_rows_is_not_fixable() -> None:
    report = validate(rows(chat(), "{bad"), Technique.SFT, LIMITS)
    assert not report.ok
    assert not report.fixable
    assert any(i.line is None for i in report.errors)


def test_token_limit() -> None:
    report = validate(rows(chat(), chat(), chat("x" * 1000)), Technique.SFT, LIMITS)
    assert report.errors[0].line == 3
    assert "exceeds model limit" in report.errors[0].message


def test_system_only_first_and_empty_content() -> None:
    bad = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": ""},
        ]
    }
    report = validate(rows(chat(), chat(), bad), Technique.SFT, LIMITS)
    assert "system message is only allowed first" in report.errors[0].message


def test_rft_rows_end_with_user() -> None:
    prompt = {**chat(assistant=None), "answer": "4"}
    assert validate(rows(prompt, prompt), Technique.RFT, LIMITS).ok
    assert not validate(rows(chat(), chat()), Technique.RFT, LIMITS).ok


def test_distillation_accepts_prompt_only_or_full() -> None:
    assert validate(rows(chat(assistant=None), chat()), Technique.DISTILLATION, LIMITS).ok


def test_dpo() -> None:
    good = {
        "input": {"messages": [{"role": "user", "content": "q"}]},
        "preferred_output": [{"role": "assistant", "content": "a"}],
        "non_preferred_output": [{"role": "assistant", "content": "b"}],
    }
    bad = {**good, "non_preferred_output": []}
    report = validate(rows(good, good, bad), Technique.DPO, LIMITS)
    assert [i.line for i in report.errors] == [3]


def test_cpt() -> None:
    assert validate(rows({"text": "doc one"}, {"text": "doc two"}), Technique.CPT, LIMITS).ok
    assert not validate(rows({"text": ""}, {"text": "x"}), Technique.CPT, LIMITS).ok


def test_nova_reserved_terms_are_errors_and_droppable() -> None:
    from skyhammer.catalog import NOVA_LIMITS

    limits = NOVA_LIMITS.model_copy(update={"min_samples": 1})
    bad = chat("Summarize this transcript. User: hi Assistant: hello")
    report = validate(rows(chat(), bad), Technique.SFT, limits)
    assert [i.line for i in report.errors] == [2]
    assert "'User:'" in report.errors[0].message and "'Assistant:'" in report.errors[0].message
    assert report.fixable
    assert report.limits is limits


def test_reserved_terms_only_apply_when_configured() -> None:
    assert validate(rows(chat("User: hi"), chat()), Technique.SFT, LIMITS).ok


def test_dataset_from_s3_and_pandas() -> None:
    import boto3
    from moto import mock_aws

    from skyhammer.config import Settings
    from skyhammer.dataset import Dataset
    from skyhammer.session import Session

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="skyhammer-data")
        s3.put_object(
            Bucket="skyhammer-data",
            Key="sets/train.jsonl",
            Body="\n".join(json.dumps(chat(f"q{i}")) for i in range(3)).encode(),
        )
        ds = Dataset.from_s3(
            "s3://skyhammer-data/sets/train.jsonl", session=Session(Settings(region="us-east-1"))
        )
    assert (ds.name, len(ds), ds.source) == ("train", 3, "s3://skyhammer-data/sets/train.jsonl")

    class FakeDF:
        def to_dict(self, orient: str) -> list[dict[str, object]]:
            return [{"conv": chat()["messages"]}]

    assert Dataset.from_pandas(FakeDF(), "df", messages_column="conv")._rows[0][1]["messages"]


def test_exact_token_counts_from_the_model_tokenizer() -> None:
    def words(text: str) -> int:
        return len(text.split())

    report = validate(rows(chat("a " * 40, "b"), chat("c", "d")), Technique.SFT, LIMITS, words)
    assert report.exact_tokens and report.ok and report.max_sample_tokens == 41
    assert not report.summary().split(" rows, ")[1].startswith("~")
    over = validate(rows(chat("a " * 80, "b"), chat()), Technique.SFT, LIMITS, words)
    assert "81 tokens exceeds model limit 50" in over.errors[0].message
    estimated = validate(rows(chat("a " * 80, "b"), chat()), Technique.SFT, LIMITS)
    assert estimated.errors == [] and "~" in estimated.summary()  # 4 chars/token estimate


def test_dataset_lineage_links_versions_to_models(monkeypatch: object) -> None:
    import pytest

    from skyhammer import Dataset, registry
    from skyhammer.errors import SkyhammerError
    from skyhammer.jobs import JobRecord

    mp: pytest.MonkeyPatch = monkeypatch  # type: ignore[assignment]
    mp.setattr(
        Dataset, "list", staticmethod(lambda session=None: [{"name": "support", "sha256": "abc"}])
    )

    def job(name: str, ds: str, at: float, tags: dict[str, str]) -> JobRecord:
        return JobRecord(
            name,
            "id",
            "sagemaker",
            "us-east-1",
            "sft",
            "qwen3-0.6b",
            "m",
            "bedrock",
            f"model-{name}",
            "lora",
            "fp",
            tags=tags,
            dataset={"name": ds, "sha256": "s", "rows": 10},
            created_at=at,
        )

    mp.setattr(
        registry,
        "list_jobs",
        lambda session, include_deleted: [
            job("a", "support", 1, {}),
            job("b", "other", 2, {}),
            job("c", "support", 3, {"distilled-from": "us.anthropic.claude-haiku-4-5"}),
        ],
    )
    info = Dataset.lineage("support", session=object())
    assert [j["job"] for j in info["jobs"]] == ["c", "a"]
    assert info["jobs"][0]["distilled_from"] == "us.anthropic.claude-haiku-4-5"
    with pytest.raises(SkyhammerError, match="No dataset named"):
        Dataset.lineage("missing", session=object())
