"""Cross-account: train in one account, serve in another (dev -> prod).

Each account is reached by assuming a role from your base credentials. In skyhammer.yaml:

    accounts:
      prod: {role_arn: "arn:aws:iam::222222222222:role/skyhammer-prod"}
    share_with: ["222222222222"]   # this (training) account's models may go to prod

and in prod's settings `trust_accounts: ["111111111111"]`; then `skyhammer bootstrap` in
both accounts (the grants are bootstrap stack parameters). `skyhammer auth policy
--cross-account` prints the extra IAM permissions for the people or CI that promote.

Bedrock-trained models (Nova) are shared through AWS RAM and copied into prod, which needs
both accounts in one AWS Organization; open-weight models are imported again in prod from
the training account's S3 weights. Promotion is free apart from the import or copy itself
(model storage is billed monthly in each account); serving costs as usual.
Run: uv run python examples/39_cross_account.py support-qwen-v1
CLI: skyhammer models promote support-qwen-v1 --to prod
"""

import sys

from skyhammer import LLM, Session

dev = Session()
prod = dev.for_account("prod")
print("training in", dev.account_id, "| serving in", prod.account_id)

# A data account works the same way: read with its session, train with yours.
#   data = Dataset.from_s3("s3://corp-data/support.jsonl", session=dev.for_account("data"))

tuned = LLM.get(sys.argv[1] if len(sys.argv) > 1 else "support-qwen-v1", session=dev)
in_prod = tuned.promote(to="prod")  # idempotent: re-running returns the existing copy
print("promoted:", in_prod.model_ref)

# From now on prod can find it by name, like any of its own models.
served = LLM.get(in_prod.model_ref.name, session=prod).deploy()
print(served.chat.completions.create(messages=[{"role": "user", "content": "Hi"}]).choices[0])
served.delete()  # the deployment and prod's copy; the dev model is untouched
