"""SE-6: assume-role sessions per account, and promoting models between accounts."""

import json
from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.validate import validate_parameters
from moto import mock_aws

from skyhammer import accounts, registry
from skyhammer.bootstrap import _account_parameters, template_body
from skyhammer.config import Account, Settings
from skyhammer.errors import ConfigError, JobFailedError, SkyhammerError
from skyhammer.iam import cross_account_policy
from skyhammer.jobs import ModelRef
from skyhammer.session import Session

SRC, DST = "111111111111", "222222222222"
NOVA = ModelRef(
    "support-nova-v1",
    f"arn:aws:bedrock:us-east-1:{SRC}:custom-model/amazon.nova-micro-v1:0:128k/abc123",
    "bedrock",
    "nova-micro",
    "us-east-1",
)
QWEN = ModelRef(
    "support-qwen-v1",
    f"arn:aws:bedrock:us-east-1:{SRC}:imported-model/xyz",
    "sagemaker",
    "qwen3-0.6b",
    "us-east-1",
    api="openai_chat",
    artifacts_uri=f"s3://skyhammer-{SRC}-us-east-1/jobs/j/output/model/",
)


def valid(service: str, op: str, kwargs: dict[str, Any]) -> None:
    model = boto3.client(service, region_name="us-east-1").meta.service_model
    validate_parameters(kwargs, model.operation_model(op).input_shape)


def error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "Op")


# ---- sessions ----


def test_assumed_session_uses_the_role_and_refreshes(settings: Settings) -> None:
    with mock_aws():
        role = f"arn:aws:iam::{DST}:role/skyhammer-prod"
        prod = Session(settings).assume(role, region="us-west-2")
        assert prod.region == "us-west-2"
        assert prod.settings.role_arn == role
        # This account's explicit bucket and roles don't carry over to the other account.
        assert prod.settings.bucket is None and prod.settings.import_role_arn is None
        creds = prod.boto.get_credentials()
        assert creds.method == "assume-role"
        assert prod.account_id == DST  # moto answers as the assumed role's account
        assert "role=" in repr(prod)


def test_for_account_reads_settings(settings: Settings) -> None:
    accts = {"prod": Account(role_arn=f"arn:aws:iam::{DST}:role/x", external_id="e-1")}
    session = Session(settings.model_copy(update={"accounts": accts}))
    prod = session.for_account("prod")
    assert prod.settings.role_arn == accts["prod"].role_arn
    assert prod.settings.external_id == "e-1" and prod.region == "us-east-1"
    with pytest.raises(ConfigError, match="known: prod"):
        session.for_account("staging")


# ---- routes and grants ----


def test_route_by_model_kind() -> None:
    assert accounts.route(NOVA) == "copy"
    assert accounts.route(QWEN) == "import"
    mantle = ModelRef(
        "m",
        "arn:aws:bedrock:us-west-2:1:custom-model/x",
        "mantle",
        "qwen3-32b",
        "us-west-2",
        api="mantle",
    )
    with pytest.raises(SkyhammerError, match="can't be promoted"):
        accounts.route(mantle)


class FakeSession:
    def __init__(self, account: str, params: dict[str, str], clients: dict[str, Any]) -> None:
        self.account_id = account
        self.region = "us-east-1"
        self.settings = Settings(tags={"team": "ml"})
        self.resources = SimpleNamespace(kms_key_arn=f"arn:aws:kms:us-east-1:{account}:key/k")
        cfn = SimpleNamespace(
            describe_stacks=lambda StackName: {
                "Stacks": [
                    {
                        "Parameters": [
                            {"ParameterKey": k, "ParameterValue": v} for k, v in params.items()
                        ]
                    }
                ]
            }
        )
        self.clients = {"cloudformation": cfn, **clients}

    def client(self, name: str, region: str | None = None) -> Any:
        return self.clients[name]


def test_missing_grants_fail_early_with_the_fix() -> None:
    src = FakeSession(SRC, {"ShareWithAccounts": ""}, {})
    dst = FakeSession(DST, {"TrustedSourceAccounts": ""}, {})
    with pytest.raises(ConfigError) as e:
        accounts.check_grants(src, dst, QWEN, "import")  # type: ignore[arg-type]
    assert f'share_with: ["{DST}"]' in str(e.value)
    assert f'trust_accounts: ["{SRC}"]' in str(e.value)
    # A copy only needs the source's grant (key policy); the target trusts via RAM.
    src = FakeSession(SRC, {"ShareWithAccounts": f"333333333333,{DST}"}, {})
    accounts.check_grants(src, dst, NOVA, "copy")  # type: ignore[arg-type]


# ---- promotion ----


class FakeRam:
    def __init__(self, existing: bool = False, pending: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.existing = existing
        self.pending = pending

    def get_resource_shares(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("get", kw))
        return {"resourceShares": [{"resourceShareArn": "arn:share/1"}] if self.existing else []}

    def create_resource_share(self, **kw: Any) -> dict[str, Any]:
        valid("ram", "CreateResourceShare", kw)
        self.calls.append(("create", kw))
        return {"resourceShare": {"resourceShareArn": "arn:share/1"}}

    def associate_resource_share(self, **kw: Any) -> None:
        self.calls.append(("associate", kw))

    def get_resource_share_invitations(self, **kw: Any) -> dict[str, Any]:
        status = "PENDING" if self.pending else "ACCEPTED"
        return {"resourceShareInvitations": [{"status": status, "resourceShareInvitationArn": "i"}]}

    def accept_resource_share_invitation(self, **kw: Any) -> None:
        self.calls.append(("accept", kw))


class FakeBedrock:
    """Target-account Bedrock: the shared model shows up after one poll; the copy job
    completes on its second describe (or fails)."""

    def __init__(self, fail: bool = False) -> None:
        self.share_polls = 0
        self.job_polls = 0
        self.fail = fail
        self.created: dict[str, Any] = {}

    def get_custom_model(self, modelIdentifier: str) -> dict[str, Any]:
        if modelIdentifier == NOVA.model_arn:
            self.share_polls += 1
            if self.share_polls < 2:
                raise error("ResourceNotFoundException")
            return {"modelArn": modelIdentifier}
        raise error("ResourceNotFoundException")  # no copy yet

    def create_model_copy_job(self, **kw: Any) -> dict[str, str]:
        valid("bedrock", "CreateModelCopyJob", kw)
        self.created = kw
        return {"jobArn": f"arn:aws:bedrock:us-east-1:{DST}:model-copy-job/j1"}

    def get_model_copy_job(self, jobArn: str) -> dict[str, Any]:
        self.job_polls += 1
        status = "InProgress" if self.job_polls < 2 else ("Failed" if self.fail else "Completed")
        return {
            "status": status,
            "failureMessage": "KMS access denied",
            "targetModelArn": f"arn:aws:bedrock:us-east-1:{DST}:custom-model/support-nova-v1",
        }


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[tuple[ModelRef, dict[str, Any]]]:
    out: list[tuple[ModelRef, dict[str, Any]]] = []
    monkeypatch.setattr(registry, "save_model_ref", lambda s, ref, **kw: out.append((ref, kw)))
    monkeypatch.setattr(accounts.time, "sleep", lambda s: None)
    return out


def test_promote_bedrock_model_shares_and_copies(saved: list[Any]) -> None:
    ram_src, ram_dst, bedrock = FakeRam(), FakeRam(pending=True), FakeBedrock()
    src = FakeSession(SRC, {"ShareWithAccounts": DST}, {"ram": ram_src})
    dst = FakeSession(DST, {}, {"ram": ram_dst, "bedrock": bedrock})
    llm = accounts.promote(SimpleNamespace(model_ref=NOVA, session=src), to=dst, session=src)  # type: ignore[arg-type]
    create = dict(ram_src.calls)["create"]
    assert create["principals"] == [DST] and create["resourceArns"] == [NOVA.model_arn]
    assert create["allowExternalPrincipals"] is False
    assert {"key": "skyhammer:name", "value": "support-nova-v1"} in create["tags"]
    assert ("accept", {"resourceShareInvitationArn": "i"}) in ram_dst.calls
    assert bedrock.created["sourceModelArn"] == NOVA.model_arn
    assert bedrock.created["modelKmsKeyId"].startswith(f"arn:aws:kms:us-east-1:{DST}:")
    assert llm.model_ref.model_arn.endswith(":custom-model/support-nova-v1")
    assert llm.model_ref.backend == "bedrock" and llm.session is dst
    ref, extra = saved[0]
    assert ref == llm.model_ref and extra["promoted_from"]["account"] == SRC


def test_promote_reuses_an_existing_share_and_reports_copy_failures(saved: list[Any]) -> None:
    ram_src = FakeRam(existing=True)
    src = FakeSession(SRC, {"ShareWithAccounts": DST}, {"ram": ram_src})
    dst = FakeSession(DST, {}, {"ram": FakeRam(), "bedrock": FakeBedrock(fail=True)})
    with pytest.raises(JobFailedError, match="KMS access denied"):
        accounts.promote(SimpleNamespace(model_ref=NOVA, session=src), to=dst, session=src)  # type: ignore[arg-type]
    kinds = [k for k, _ in ram_src.calls]
    assert "create" not in kinds and "associate" in kinds
    assert saved == []


def test_promote_open_weight_model_reimports(
    saved: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    imported: dict[str, Any] = {}

    def fake_import(session: Any, region: str, uri: str, name: str, **kw: Any) -> str:
        imported.update(session=session, uri=uri, name=name, tags=kw["tags"])
        return f"arn:aws:bedrock:us-east-1:{DST}:imported-model/new"

    monkeypatch.setattr(accounts, "import_to_bedrock", fake_import)
    src = FakeSession(SRC, {"ShareWithAccounts": DST}, {})
    dst = FakeSession(DST, {"TrustedSourceAccounts": SRC}, {})
    llm = accounts.promote(
        SimpleNamespace(model_ref=QWEN, session=src),
        to=dst,
        name="qwen-prod",
        session=src,  # type: ignore[arg-type]
    )
    assert imported["session"] is dst and imported["uri"] == QWEN.artifacts_uri
    assert imported["tags"]["skyhammer:name"] == "qwen-prod"
    assert (llm.model_ref.name, llm.model_ref.api) == ("qwen-prod", "openai_chat")


def test_promote_refuses_same_account_and_base_models() -> None:
    src = FakeSession(SRC, {}, {})
    with pytest.raises(SkyhammerError, match="already in account"):
        accounts.promote(SimpleNamespace(model_ref=NOVA, session=src), to=src, session=src)  # type: ignore[arg-type]
    with pytest.raises(SkyhammerError, match="base model"):
        accounts.promote(
            SimpleNamespace(model_ref=None, name="nova-micro", session=src), to=src, session=src
        )  # type: ignore[arg-type]


def test_promoted_model_resolves_by_name_in_the_target(session: Session) -> None:
    from skyhammer.llm import LLM

    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="skyhammer-test-bucket")
        registry.save_model_ref(session, QWEN, promoted_from={"account": SRC})
        llm = LLM.get("support-qwen-v1", session=session)
        assert llm.model_ref == QWEN
        assert registry.load_model(session, "support-qwen-v1")["promoted_from"] == {"account": SRC}


# ---- bootstrap and IAM ----


def test_bootstrap_grants_follow_settings(settings: Settings) -> None:
    s = Session(
        settings.model_copy(update={"share_with": [DST, "333333333333"], "trust_accounts": [SRC]})
    )
    params = {p["ParameterKey"]: p["ParameterValue"] for p in _account_parameters(s)}
    assert params == {"ShareWithAccounts": f"{DST},333333333333", "TrustedSourceAccounts": SRC}


def test_template_grants_are_conditional() -> None:
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def tag(loader: Any, suffix: str, node: Any) -> Any:  # !If, !Ref, !Sub ... as data
        value = (
            loader.construct_sequence(node)
            if isinstance(node, yaml.SequenceNode)
            else loader.construct_mapping(node)
            if isinstance(node, yaml.MappingNode)
            else loader.construct_scalar(node)
        )
        return {suffix: value}

    Loader.add_multi_constructor("!", tag)
    t = yaml.load(template_body(), Loader=Loader)
    assert {"ShareWithAccounts", "TrustedSourceAccounts"} <= set(t["Parameters"])
    granted = {
        "Key": t["Resources"]["Key"]["Properties"]["KeyPolicy"]["Statement"],
        "BucketPolicy": t["Resources"]["BucketPolicy"]["Properties"]["PolicyDocument"]["Statement"],
        "DataAccessPolicy": t["Resources"]["DataAccessPolicy"]["Properties"]["PolicyDocument"][
            "Statement"
        ],
    }
    conditions = {
        name: [s["If"][0] for s in statements if isinstance(s, dict) and "If" in s]
        for name, statements in granted.items()
    }
    assert conditions == {
        "Key": ["ShareModels"],
        "BucketPolicy": ["ShareModels", "ShareModels"],
        "DataAccessPolicy": ["TrustSources", "TrustSources"],
    }
    trust = granted["DataAccessPolicy"][-1]["If"][1]
    assert trust["Condition"] == {
        "StringEquals": {"aws:ResourceAccount": {"Ref": "TrustedSourceAccounts"}}
    }


def test_cross_account_policy_is_small_and_scoped() -> None:
    policy = cross_account_policy(DST)
    assert len(json.dumps(policy, separators=(",", ":"))) < 2000
    copy = next(s for s in policy["Statement"] if s["Sid"] == "CopySharedModels")
    assert f"arn:aws:bedrock:*:{DST}:model-copy-job/*" in copy["Resource"]
