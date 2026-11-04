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


def _bedrock_content(content: Any) -> list[Record]:
    """Message content -> Converse blocks: text, plus image/video blocks for media parts
    (already staged to S3 by ``Dataset.upload``; see ``skyhammer.media``)."""
    if not isinstance(content, list) or not any(
        isinstance(p, dict) and p.get("type") in ("image_url", "video_url") for p in content
    ):
        return [{"text": message_text(content)}]
    blocks: list[Record] = []
    text: list[str] = []
    for part in content:
        kind = part.get("type", "text")
        if kind not in ("image_url", "video_url"):
            text.append(part.get("text", "") if kind == "text" else "")
            continue
        if "".join(text):
            blocks.append({"text": "".join(text)})
        text = []
        ref = part[kind]
        if not isinstance(ref, dict) or not str(ref.get("url", "")).startswith("s3://"):
            raise ValueError("media must be staged to S3 first (Dataset.upload does this)")
        location: Record = {"uri": ref["url"]}
        if ref.get("bucket_owner"):
            location["bucketOwner"] = ref["bucket_owner"]
        blocks.append(
            {
                kind.removesuffix("_url"): {
                    "format": ref["format"],
                    "source": {"s3Location": location},
                }
            }
        )
    if "".join(text):
        blocks.append({"text": "".join(text)})
    return blocks


def to_bedrock_conversation(record: Record) -> Record:
    """OpenAI chat record -> Bedrock/Nova ``bedrock-conversation-2024`` record."""
    out: Record = {"schemaVersion": BEDROCK_CONVERSATION_SCHEMA}
    messages = record["messages"]
    system = [m for m in messages if m["role"] == "system"]
    if system:
        out["system"] = [{"text": message_text(m["content"])} for m in system]
    out["messages"] = [
        {"role": m["role"], "content": _bedrock_content(m["content"])}
        for m in messages
        if m["role"] != "system"
    ]
    return out


def fold_system(record: Record) -> Record:
    """Move system messages into the first user turn, for chat templates that reject a
    conversation prefix with no user message (Qwen 3.6 under verl, which renders each
    turn on its own: "No user query found in messages", live 2026-11-04). Records without
    ``messages`` or system messages are returned unchanged."""
    messages = record.get("messages")
    if not isinstance(messages, list) or not any(m.get("role") == "system" for m in messages):
        return record
    system = "\n\n".join(message_text(m["content"]) for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    for m in rest:
        if m["role"] == "user":
            content = m["content"]
            if isinstance(content, list):
                m["content"] = [{"type": "text", "text": system + "\n\n"}, *content]
            else:
                m["content"] = f"{system}\n\n{content}"
            break
    return {**record, "messages": rest}


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
