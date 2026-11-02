"""Cross-account work (SE-6): one account per stage, models promoted between them.

Configure the accounts once (``skyhammer.yaml``), each reached by assuming a role from
your base credentials:

    accounts:
      prod: {role_arn: arn:aws:iam::222222222222:role/skyhammer}
    share_with: ["222222222222"]          # in the training account's settings
    # trust_accounts: ["111111111111"]    # in the prod account's settings

then ``skyhammer bootstrap`` in both. ``Session.for_account("prod")`` works in that
account; ``promote(model, to="prod")`` makes a fine-tuned model available there:

- Bedrock-trained models (Nova and other Bedrock customizations) are shared through AWS
  RAM and copied into the target account. Both accounts must be in one AWS Organization
  with RAM sharing enabled (Bedrock's rule).
- Open-weight models trained on SageMaker (or imported) are imported again in the target
  account from the source account's weights in S3 (bucket and key policies from
  ``share_with``; the target's import role from ``trust_accounts``). No Organization needed.

Datasets in a data account need no promotion: read them with that account's session,
``Dataset.from_s3(uri, session=Session.for_account("data"))``, and train as usual.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError

from skyhammer import registry
from skyhammer import tags as tagging
from skyhammer.errors import ConfigError, JobFailedError, SkyhammerError
from skyhammer.imports import import_to_bedrock
from skyhammer.jobs import ModelRef

if TYPE_CHECKING:
    from skyhammer.llm import LLM
    from skyhammer.session import Session

log = logging.getLogger("skyhammer")


def _target(source: Session, to: Session | str) -> Session:
    return source.for_account(to) if isinstance(to, str) else to


def _source_ref(model: str | LLM, session: Session) -> ModelRef:
    from skyhammer.llm import LLM

    llm = LLM.get(model, session=session) if isinstance(model, str) else model
    if llm.model_ref is None:
        raise SkyhammerError(
            f"{llm.name} is a base model; every account can call it, so there is nothing to promote"
        )
    return llm.model_ref


def route(ref: ModelRef) -> str:
    """How ``ref`` reaches another account: "copy" (RAM share + model copy) or "import"."""
    if ref.api == "converse" and ":custom-model/" in ref.model_arn:
        return "copy"
    if ref.artifacts_uri:
        return "import"
    raise SkyhammerError(
        f"{ref.name} can't be promoted: it is neither a Bedrock custom model nor backed by "
        "weights in S3 (bedrock-mantle fine-tunes stay in their account)"
    )


def stack_parameters(session: Session, region: str | None = None) -> dict[str, str]:
    cfn = session.client("cloudformation", region=region)
    try:
        stack = cfn.describe_stacks(StackName=session.settings.bootstrap_stack)["Stacks"][0]
    except ClientError as e:
        if "does not exist" in str(e):
            return {}
        raise
    return {p["ParameterKey"]: p.get("ParameterValue", "") for p in stack.get("Parameters", [])}


def _listed(params: dict[str, str], key: str, account: str) -> bool:
    return account in [a.strip() for a in params.get(key, "").split(",")]


def check_grants(source: Session, target: Session, ref: ModelRef, how: str) -> None:
    """Fail early, with the fix, when the bootstrap grants a promotion needs are missing."""
    src, dst = source.account_id, target.account_id
    problems = []
    if not _listed(stack_parameters(source, ref.region), "ShareWithAccounts", dst):
        problems.append(
            f'in account {src}: add share_with: ["{dst}"] to its settings and run '
            f"`skyhammer bootstrap` in {ref.region}"
        )
    if how == "import" and not _listed(stack_parameters(target), "TrustedSourceAccounts", src):
        problems.append(
            f'in account {dst}: add trust_accounts: ["{src}"] to its settings and run '
            f"`skyhammer bootstrap` in {target.region}"
        )
    if problems:
        raise ConfigError(
            f"Promoting {ref.name} from {src} to {dst} needs bootstrap grants: "
            + "; ".join(problems)
        )


# ---- Bedrock custom models: RAM share, then a model copy job in the target ----


def share(source: Session, ref: ModelRef, account_id: str) -> str:
    """Share a Bedrock custom model with ``account_id`` through AWS RAM (idempotent);
    returns the resource share ARN."""
    ram = source.client("ram", region=ref.region)
    name = f"skyhammer-{ref.name}"
    shares = ram.get_resource_shares(
        resourceOwner="SELF", name=name, resourceShareStatus="ACTIVE"
    ).get("resourceShares", [])
    if shares:
        arn = str(shares[0]["resourceShareArn"])
        ram.associate_resource_share(
            resourceShareArn=arn, resourceArns=[ref.model_arn], principals=[account_id]
        )
        return arn
    tags = tagging.build_tags({"name": ref.name, "kind": "share"}, source.settings.tags)
    resp = ram.create_resource_share(
        name=name,
        resourceArns=[ref.model_arn],
        principals=[account_id],
        allowExternalPrincipals=False,  # Bedrock shares models within the Organization only
        tags=[{"key": k, "value": v} for k, v in tags.items()],
    )
    return str(resp["resourceShare"]["resourceShareArn"])


def accept(target: Session, share_arn: str, region: str) -> None:
    """Accept a pending RAM invitation (none exists when the Organization auto-accepts)."""
    ram = target.client("ram", region=region)
    invitations = ram.get_resource_share_invitations(resourceShareArns=[share_arn]).get(
        "resourceShareInvitations", []
    )
    for inv in invitations:
        if inv.get("status") == "PENDING":
            ram.accept_resource_share_invitation(
                resourceShareInvitationArn=inv["resourceShareInvitationArn"]
            )


def _wait(
    what: str,
    check: Callable[[], str | None],
    *,
    timeout_s: float,
    poll_s: float,
    sleep: Callable[[float], None],
) -> str:
    deadline = time.monotonic() + timeout_s
    while True:
        result = check()
        if result:
            return result
        if time.monotonic() > deadline:
            raise TimeoutError(f"{what} not done after {timeout_s / 60:.0f} min")
        sleep(poll_s)


def copy(
    target: Session,
    ref: ModelRef,
    name: str,
    *,
    wait: bool = True,
    poll_s: float = 30,
    sleep: Callable[[float], None] | None = None,
) -> str:
    """Copy a shared custom model into the target account (idempotent); returns its ARN."""
    sleep = sleep or time.sleep
    bedrock = target.client("bedrock")
    try:
        return str(bedrock.get_custom_model(modelIdentifier=name)["modelArn"])
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("ResourceNotFoundException", "ValidationException"):
            raise

    def visible() -> str | None:  # RAM shares take a moment to reach Bedrock
        try:
            bedrock.get_custom_model(modelIdentifier=ref.model_arn)
        except ClientError as e:
            if e.response["Error"]["Code"] in (
                "ResourceNotFoundException",
                "AccessDeniedException",
            ):
                return None
            raise
        return ref.model_arn

    _wait(f"RAM share of {ref.name}", visible, timeout_s=600, poll_s=10, sleep=sleep)
    res = target.resources
    tags = tagging.build_tags({"name": name, "kind": "promoted"}, target.settings.tags)
    kwargs: dict[str, Any] = {
        "sourceModelArn": ref.model_arn,
        "targetModelName": name,
        "targetModelTags": tagging.as_bedrock(tags),
        # Same inputs, same job: a retry after a crash doesn't start a second copy.
        "clientRequestToken": hashlib.sha256(f"{ref.model_arn}|{name}".encode()).hexdigest()[:64],
    }
    if res.kms_key_arn:
        kwargs["modelKmsKeyId"] = res.kms_key_arn
    job_arn = bedrock.create_model_copy_job(**kwargs)["jobArn"]
    log.info("copying %s into account %s as %s", ref.name, target.account_id, name)

    def done() -> str | None:
        job = bedrock.get_model_copy_job(jobArn=job_arn)
        if job["status"] == "Failed":
            raise JobFailedError(name, job.get("failureMessage") or "model copy failed", None)
        return str(job["targetModelArn"]) if job["status"] == "Completed" else None

    if not wait:
        return str(bedrock.get_model_copy_job(jobArn=job_arn)["targetModelArn"])
    return _wait(f"Model copy {name}", done, timeout_s=4 * 3600, poll_s=poll_s, sleep=sleep)


# ---- promotion ----


def promote(
    model: str | LLM,
    *,
    to: Session | str,
    name: str | None = None,
    session: Session | None = None,
    wait: bool = True,
) -> LLM:
    """Make a fine-tuned model available in another account; returns it as an ``LLM`` in
    that account (``deploy()`` it there as usual). ``to`` is a session or the name of an
    account in settings ``accounts``; ``name`` defaults to the model's name. Idempotent:
    promoting again returns the existing copy."""
    from skyhammer.llm import LLM
    from skyhammer.session import get_session

    source = session or (model.session if isinstance(model, LLM) else get_session())
    target = _target(source, to)
    ref = _source_ref(model, source)
    name = name or ref.name
    how = route(ref)
    if source.account_id == target.account_id:
        raise SkyhammerError(f"{ref.name} is already in account {target.account_id}")
    check_grants(source, target, ref, how)
    origin = {"account": source.account_id, "model": ref.name, "arn": ref.model_arn}
    if how == "copy":
        share_arn = share(source, ref, target.account_id)
        accept(target, share_arn, ref.region)
        arn = copy(target, ref, name, wait=wait)
        promoted = ModelRef(name, arn, "bedrock", ref.base_alias, target.region)
    else:
        assert ref.artifacts_uri is not None  # route() checked
        tags = tagging.build_tags({"name": name, "kind": "promoted"}, target.settings.tags)
        arn = import_to_bedrock(
            target, target.region, ref.artifacts_uri, name, tags=tags, wait=wait, log=log.info
        )
        promoted = ModelRef(
            name,
            arn,
            "sagemaker",
            ref.base_alias,
            target.region,
            api="openai_chat",
            serving="bedrock",
            artifacts_uri=ref.artifacts_uri,
        )
    registry.save_model_ref(target, promoted, promoted_from=origin)
    return LLM.from_model_ref(promoted, session=target)
