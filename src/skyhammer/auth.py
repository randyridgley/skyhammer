"""Bedrock API keys and a drop-in OpenAI client.

Short-term Bedrock API keys are SigV4-presigned tokens derived from your current AWS
credentials (valid up to 12 hours, and never longer than those credentials). They work as
bearer tokens for Bedrock and Bedrock Runtime, including Bedrock's OpenAI-compatible
endpoint, so app developers can call models without an AWS profile:

    export SKYHAMMER_API_KEY=$(skyhammer auth api-key)       # or AWS_BEARER_TOKEN_BEDROCK
    client = skyhammer.openai_client()                         # openai.OpenAI for Bedrock
    client.chat.completions.create(model="openai.gpt-oss-20b-1:0", messages=[...])

Long-term keys are IAM service-specific credentials and require an IAM user
(``aws iam create-service-specific-credential --service-name bedrock.amazonaws.com``).
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from botocore.auth import SigV4QueryAuth
from botocore.awsrequest import AWSRequest

from skyhammer.errors import SkyhammerError

if TYPE_CHECKING:
    from skyhammer.session import Session

MAX_HOURS = 12
_PREFIX = "bedrock-api-key-"


def generate_api_key(
    session: Session | None = None, *, region: str | None = None, hours: float = MAX_HOURS
) -> str:
    """A short-term Bedrock API key from the current AWS credentials."""
    from skyhammer.session import get_session

    if not 0 < hours <= MAX_HOURS:
        raise SkyhammerError(f"Short-term keys last at most {MAX_HOURS} hours")
    session = session or get_session()
    creds = session.boto.get_credentials()
    if creds is None:
        raise SkyhammerError("No AWS credentials found to sign a Bedrock API key")
    region = region or session.region
    req = AWSRequest(
        method="POST",
        url="https://bedrock.amazonaws.com/",
        params={"Action": "CallWithBearerToken"},
    )
    SigV4QueryAuth(
        creds.get_frozen_credentials(), "bedrock", region, expires=int(hours * 3600)
    ).add_auth(req)
    token = str(req.url).removeprefix("https://") + "&Version=1"
    return _PREFIX + base64.b64encode(token.encode()).decode()


def is_api_key(value: str) -> bool:
    return value.startswith(_PREFIX)


def openai_client(
    *,
    region: str | None = None,
    api_key: str | None = None,
    session: Session | None = None,
    **kwargs: Any,
) -> Any:
    """An ``openai.OpenAI`` client for Bedrock's OpenAI-compatible endpoint. Uses
    ``api_key``, then the configured key, then a fresh short-term key."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SkyhammerError("Install the OpenAI SDK: pip install 'skyhammer[openai]'") from e
    from skyhammer.session import get_session

    session = session or get_session()
    region = region or session.region
    key = api_key or session.settings.api_key or generate_api_key(session, region=region)
    return OpenAI(
        base_url=f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1", api_key=key, **kwargs
    )
