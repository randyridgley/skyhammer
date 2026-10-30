"""Client for Bedrock's OpenAI-compatible endpoint (``bedrock-mantle``).

Some Bedrock features exist only there, notably reinforcement fine-tuning of open-weight
models (Qwen3-32B, GPT-OSS-20B) and calling the resulting fine-tuned models. Requests are
signed with SigV4 from the session's AWS credentials (service ``bedrock-mantle``), or
carry a short-term Bedrock API key derived from them (``auth="api_key"``), which is what
AWS documents for fine-tuning jobs. Neither needs the OpenAI SDK.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from typing import TYPE_CHECKING, Any

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from skyhammer.errors import SkyhammerError

if TYPE_CHECKING:
    from skyhammer.session import Session

SERVICE = "bedrock-mantle"


class MantleError(SkyhammerError):
    def __init__(self, status: int, method: str, path: str, body: str) -> None:
        self.status = status
        super().__init__(f"bedrock-mantle {method} {path} failed ({status}): {body[:500]}")


class MantleClient:
    def __init__(
        self, session: Session, region: str, *, auth: str = "sigv4", timeout_s: float = 120
    ) -> None:
        if auth not in ("sigv4", "api_key"):
            raise ValueError("auth must be 'sigv4' or 'api_key'")
        self.session = session
        self.region = region
        self.auth = auth
        self.base_url = f"https://{SERVICE}.{region}.api.aws"
        self.timeout_s = timeout_s

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        files: dict[str, tuple[str, bytes]] | None = None,
        fields: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send a signed request; ``files``/``fields`` make it multipart/form-data."""
        headers: dict[str, str] = {}
        data: bytes | None = None
        if files is not None:
            data, content_type = _multipart(fields or {}, files)
            headers["Content-Type"] = content_type
        elif json_body is not None:
            data = json.dumps(json_body).encode()
            headers["Content-Type"] = "application/json"
        url = self.base_url + path
        if self.auth == "api_key":
            from skyhammer.auth import generate_api_key

            headers["Authorization"] = (
                f"Bearer {generate_api_key(self.session, region=self.region)}"
            )
        else:
            creds = self.session.boto.get_credentials()
            if creds is None:
                raise SkyhammerError("No AWS credentials found to sign bedrock-mantle requests")
            signed = AWSRequest(method=method, url=url, data=data, headers=headers)
            SigV4Auth(creds.get_frozen_credentials(), SERVICE, self.region).add_auth(signed)
            headers = dict(signed.headers)
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read()
        except urllib.error.HTTPError as e:
            raise MantleError(e.code, method, path, e.read().decode(errors="replace")) from None
        result: dict[str, Any] = json.loads(body) if body else {}
        return result


def _multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes]]) -> tuple[bytes, str]:
    boundary = f"skyhammer-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        header = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
        parts.append(f"{header}{value}\r\n".encode())
    for name, (filename, content) in files.items():
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'
            ).encode()
            + content
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
