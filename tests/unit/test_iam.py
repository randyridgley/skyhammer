"""SE-4: the least-privilege user policy stays scoped and fits IAM's limits."""

import json

from skyhammer.iam import user_policy

ACCOUNT = "123456789012"


def statements() -> dict[str, dict]:
    return {s["Sid"]: s for s in user_policy(ACCOUNT)["Statement"]}


def test_fits_a_managed_policy() -> None:
    assert len(json.dumps(user_policy(ACCOUNT), separators=(",", ":"))) <= 6144


def test_writes_are_scoped_to_skyhammer_resources() -> None:
    s = statements()
    assert s["Data"]["Resource"] == [
        f"arn:aws:s3:::skyhammer-{ACCOUNT}-*",
        f"arn:aws:s3:::skyhammer-{ACCOUNT}-*/*",
    ]
    assert s["Graders"]["Resource"].endswith(":function:skyhammer-grader-*")
    assert (
        s["PassBootstrapRoles"]["Resource"] == f"arn:aws:iam::{ACCOUNT}:role/skyhammer-bootstrap-*"
    )
    assert s["Bootstrap"]["Resource"].endswith(":stack/skyhammer-bootstrap/*")
    assert s["Encryption"]["Condition"] == {
        "ForAnyValue:StringEquals": {"kms:ResourceAliases": "alias/skyhammer"}
    }
    # Only read-only or account-scoped actions use "*".
    for st in s.values():
        if st["Resource"] == "*":
            actions = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
            assert all(
                a.split(":")[1].startswith(("Get", "List", "Describe", "Search"))
                or a
                in {
                    "bedrock:CallWithBearerToken",
                    "bedrock-mantle:CallWithBearerToken",
                    "application-autoscaling:RegisterScalableTarget",
                    "application-autoscaling:DeregisterScalableTarget",
                    "application-autoscaling:PutScalingPolicy",
                    "cloudwatch:PutMetricAlarm",
                    "cloudwatch:DeleteAlarms",
                }
                for a in actions
            ), st["Sid"]


def test_without_bootstrap() -> None:
    assert "Bootstrap" not in {s["Sid"] for s in user_policy(ACCOUNT, bootstrap=False)["Statement"]}
