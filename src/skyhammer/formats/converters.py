"""Record-level converters. Input records are assumed to have passed validation."""

from __future__ import annotations

from typing import Any

Record = dict[str, Any]

BEDROCK_CONVERSATION_SCHEMA = "bedrock-conversation-2024"


def message_text(content: Any) -> str:
    """OpenAI content may be a string or a list of ``{"type": "text", "text": ...}`` parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type", "text") == "text"
        )
    return ""


def to_bedrock_conversation(record: Record) -> Record:
    """OpenAI chat record -> Bedrock/Nova ``bedrock-conversation-2024`` record."""
    out: Record = {"schemaVersion": BEDROCK_CONVERSATION_SCHEMA}
    messages = record["messages"]
    system = [m for m in messages if m["role"] == "system"]
    if system:
        out["system"] = [{"text": message_text(m["content"])} for m in system]
    out["messages"] = [
        {"role": m["role"], "content": [{"text": message_text(m["content"])}]}
        for m in messages
        if m["role"] != "system"
    ]
    return out


def to_sagemaker_sft(record: Record) -> Record:
    """OpenAI chat record -> SageMaker serverless SFT record.

    TODO(M1): confirm the exact serverless recipe schema; recipes are expected to accept
    OpenAI-style ``messages`` so this normalizes content to plain strings only.
    """
    return {
        "messages": [
            {"role": m["role"], "content": message_text(m["content"])} for m in record["messages"]
        ]
    }


def to_sagemaker_dpo(record: Record) -> Record:
    """OpenAI preference record -> SageMaker DPO "messages" format: full ``chosen`` and
    ``rejected`` conversations sharing the same prompt (and system message, if any)."""

    def msgs(ms: list[Record]) -> list[Record]:
        return [{"role": m["role"], "content": message_text(m["content"])} for m in ms]

    prompt = msgs(record["input"]["messages"])
    return {
        "chosen": [*prompt, *msgs(record["preferred_output"][-1:])],
        "rejected": [*prompt, *msgs(record["non_preferred_output"][-1:])],
    }


def to_sagemaker_rlvr(record: Record) -> Record:
    """Prompt + reference answer -> SageMaker RLVR (verl) record. The reward Lambda gets the
    reference as ``reference_answer.text``; the trailing assistant turn, if any, is the
    reference rather than part of the prompt."""
    from skyhammer.graders import reference_answer

    messages = record["messages"]
    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    return {
        "prompt": [{"role": m["role"], "content": message_text(m["content"])} for m in messages],
        "reward_model": {"ground_truth": reference_answer(record), "style": "rule"},
    }
