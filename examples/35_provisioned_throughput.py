"""Provisioned Throughput: reserved Bedrock capacity for a fine-tune or base model.

On-demand (pay per token) suits almost all traffic. Provisioned Throughput reserves model
units billed by the hour whether you use them or not, for steady high volume that
on-demand quotas can't carry. One Nova model unit is $60.50/hour in us-east-1 (about
$44,000/month); 1- and 6-month commitments cut the rate but can't be cancelled.

Skyhammer prices the purchase from the AWS Price List first and won't buy without
``max_hourly_usd`` covering it. Calls then go to the capacity with no code changes, and
deploying another fine-tune of the same base under the same name swaps it in place.

New accounts have 0 model units: request them in Service Quotas ("Model units
no-commitment Provisioned Throughputs across base models", or the per-model quota for
custom models) before buying.

Cost: free as written (prices only). With ``--buy``: one no-commitment Nova Micro model
unit, about $61 for the hour or less it runs (Bedrock bills whole hours), deleted at the
end.
Run: uv run python examples/35_provisioned_throughput.py [--buy]
"""

import sys

from skyhammer import LLM
from skyhammer.provisioned import price

llm = LLM("nova-micro")  # or LLM.get("support-nova") for a fine-tune
for term in (None, "1_month", "6_months"):
    print(f"{term or 'no commitment':14} {price(llm.session, llm.info.name, commitment=term)}")

if "--buy" in sys.argv:
    llm.deploy(type="provisioned", model_units=1, max_hourly_usd=65, name="skyhammer-example-pt")
    try:
        reply = llm.chat.completions.create(
            messages=[{"role": "user", "content": "Say hi in five words."}], max_tokens=20
        )
        print(llm.model_id, "->", reply.choices[0].message.content)
        # Later, from any process: LLM.provisioned("skyhammer-example-pt")
    finally:
        assert llm.deployment is not None
        llm.deployment.delete()  # stops the hourly billing
