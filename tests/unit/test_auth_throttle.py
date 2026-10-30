import base64
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from botocore.exceptions import ClientError

from skyhammer.auth import generate_api_key, is_api_key, openai_client
from skyhammer.config import load_settings
from skyhammer.errors import SkyhammerError, ThrottledError
from skyhammer.inference import Chat, split_reasoning
from skyhammer.session import Session


def test_short_term_key_is_presigned_bedrock_bearer_token(session: Session) -> None:
    key = generate_api_key(session, hours=1)
    assert is_api_key(key)
    url = base64.b64decode(key.removeprefix("bedrock-api-key-")).decode()
    q = parse_qs(urlparse("https://" + url).query)
    assert url.startswith("bedrock.amazonaws.com/")
    assert q["Action"] == ["CallWithBearerToken"] and q["X-Amz-Expires"] == ["3600"]
    assert q["Version"] == ["1"] and "/us-east-1/bedrock/" in q["X-Amz-Credential"][0]
    with pytest.raises(SkyhammerError, match="at most 12 hours"):
        generate_api_key(session, hours=24)


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key-abc")
    assert load_settings().api_key == "bedrock-api-key-abc"
    monkeypatch.setenv("SKYHAMMER_API_KEY", "bedrock-api-key-xyz")
    assert load_settings().api_key == "bedrock-api-key-xyz"  # skyhammer's own var wins


def test_openai_client_targets_bedrock_endpoint(session: Session) -> None:
    client = openai_client(session=session, api_key="bedrock-api-key-abc")
    assert str(client.base_url) == "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/"
    assert client.api_key == "bedrock-api-key-abc"
    generated = openai_client(session=session, region="us-west-2")
    assert "us-west-2" in str(generated.base_url) and is_api_key(generated.api_key)


class Throttling:
    def converse(self, **_: Any) -> None:
        raise ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Too many tokens, please wait"}},
            "Converse",
        )


def test_throttling_surfaces_actionable_error() -> None:
    with pytest.raises(ThrottledError) as e:
        Chat(Throttling(), "us.anthropic.claude-sonnet-4-6").completions.create(
            messages=[{"role": "user", "content": "hi"}]
        )
    msg = str(e.value)
    assert "Too many tokens" in msg and "routing='global'" in msg and "models quotas" in msg


def test_reasoning_tag_variants() -> None:
    assert split_reasoning("<reasoning>plan</reasoning>ok") == ("ok", "plan")  # GPT-OSS, live
    assert split_reasoning("<think>x</reasoning>ok") == ("<think>x</reasoning>ok", None)


def test_quotas_match_with_provider_prefix(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from skyhammer.llm import LLM

    names = [
        "Cross-region model inference tokens per minute for Anthropic Claude Haiku 4.5",
        "Batch inference job size (in GB) for Claude Haiku 4.5",
        "On-demand model inference tokens per minute for Anthropic Claude Haiku 4.5 V2",
        "Cross-region model inference tokens per minute for Anthropic Claude Haiku 4.55",
        "Cross-region model inference tokens per minute for Anthropic Claude Sonnet 4.6",
    ]

    class FakeSQ:
        def get_paginator(self, _: str) -> Any:
            return self

        def paginate(self, **_: Any) -> list[dict[str, Any]]:
            return [
                {
                    "Quotas": [
                        {"QuotaName": n, "Value": 1.0, "QuotaCode": f"L-{i}"}
                        for i, n in enumerate(names)
                    ]
                }
            ]

    monkeypatch.setattr(session, "client", lambda svc, region=None: FakeSQ())
    got = [q["name"] for q in LLM("claude-haiku-4-5", session=session).quotas()]
    assert got == sorted(names[:3])  # "4.55" and other models excluded
