"""Guardrails: block topics, harmful content or PII on any call, including fine-tuned models.

Pass ``guardrail={"id", "version"}`` to any chat call. When the guardrail blocks the
prompt or the answer, ``finish_reason`` is ``"content_filter"`` and the content is the
guardrail's blocked message. It works the same for every model: serverless models apply
it on the call; fine-tuned models imported into Bedrock and SageMaker endpoints check the
prompt and the answer with the ApplyGuardrail API (a blocked prompt never reaches the model).

This example creates a temporary guardrail that denies investment advice, then deletes it.

Cost: a fraction of a cent (guardrail checks are billed per 1,000 characters).
Run: uv run python examples/29_guardrails.py
"""

import time

import boto3

from skyhammer import LLM

bedrock = boto3.client("bedrock")
guardrail_id = bedrock.create_guardrail(
    name="skyhammer-example-guardrail",
    topicPolicyConfig={
        "topicsConfig": [
            {
                "name": "Investment advice",
                "type": "DENY",
                "definition": "Recommendations about which stocks, funds or crypto to buy or sell.",
                "examples": ["Which stocks should I buy?"],
            }
        ]
    },
    blockedInputMessaging="Sorry, I can't help with investment advice.",
    blockedOutputsMessaging="Sorry, I can't share that.",
    tags=[{"key": "skyhammer:name", "value": "skyhammer-example-guardrail"}],
)["guardrailId"]
try:
    while bedrock.get_guardrail(guardrailIdentifier=guardrail_id)["status"] != "READY":
        time.sleep(2)

    llm = LLM("nova-micro")
    guardrail = {"id": guardrail_id, "version": "DRAFT"}  # or a published version number
    for question in [
        "How do I open a savings account?",
        "Which tech stocks should I buy this week?",
    ]:
        resp = llm.chat.completions.create(
            messages=[{"role": "user", "content": question}], max_tokens=80, guardrail=guardrail
        )
        choice = resp.choices[0]
        print(f"{question}\n  [{choice.finish_reason}] {choice.message.content}\n")

    # trace=True returns what the guardrail assessed (here, the denied topic it matched).
    resp = llm.chat.completions.create(
        messages=[{"role": "user", "content": "Should I buy bitcoin?"}],
        guardrail={**guardrail, "trace": True},
    )
    assessment = resp.raw["trace"]["guardrail"]["inputAssessment"][guardrail_id]
    print("matched:", assessment["topicPolicy"]["topics"][0]["name"])

    # Fine-tuned models and SageMaker endpoints take the same parameter:
    #   LLM.get("support-qwen").chat.completions.create(messages=..., guardrail=guardrail)
finally:
    bedrock.delete_guardrail(guardrailIdentifier=guardrail_id)
