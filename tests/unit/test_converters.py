from skyhammer.formats import (
    message_text,
    to_bedrock_conversation,
    to_sagemaker_dpo,
    to_sagemaker_sft,
)
from tests.conftest import chat


def test_message_text_handles_parts() -> None:
    assert message_text("hi") == "hi"
    assert (
        message_text(
            [{"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}]
        )
        == "ab"
    )
    assert message_text(None) == ""


def test_bedrock_conversation_moves_system_out_of_messages() -> None:
    out = to_bedrock_conversation(chat("hi", "hello", system="be nice"))
    assert out == {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{"text": "be nice"}],
        "messages": [
            {"role": "user", "content": [{"text": "hi"}]},
            {"role": "assistant", "content": [{"text": "hello"}]},
        ],
    }


def test_bedrock_conversation_without_system() -> None:
    assert "system" not in to_bedrock_conversation(chat())


def test_sagemaker_sft_flattens_content() -> None:
    rec = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": "a"},
        ]
    }
    assert to_sagemaker_sft(rec) == {
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    }


def test_sagemaker_dpo() -> None:
    """SageMaker DPO "messages" format (AWS docs, customizing-models-dpo, 2026-10-29)."""
    rec = {
        "input": {
            "messages": [
                {"role": "system", "content": "Be kind."},
                {"role": "user", "content": [{"type": "text", "text": "q"}]},
            ]
        },
        "preferred_output": [{"role": "assistant", "content": "good"}],
        "non_preferred_output": [{"role": "assistant", "content": "bad"}],
    }
    prompt = [{"role": "system", "content": "Be kind."}, {"role": "user", "content": "q"}]
    assert to_sagemaker_dpo(rec) == {
        "chosen": [*prompt, {"role": "assistant", "content": "good"}],
        "rejected": [*prompt, {"role": "assistant", "content": "bad"}],
    }
