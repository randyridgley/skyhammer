"""Images and video in training data (DA-10).

Rows use OpenAI content parts, the same ones ``chat`` takes:

    {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "What part is damaged?"},
            {"type": "image_url", "image_url": {"url": "photos/0412.jpg"}}]},
        {"role": "assistant", "content": "The left hinge."}]}

The URL is a local path (relative to the JSONL file, or to the working directory for
in-memory datasets), a ``file://`` URL, a base64 ``data:`` URL, or an ``s3://`` URI.
Validation checks every reference against the model's limits (formats, count per row,
file size, one kind of media per dataset) before anything is uploaded. On upload, local
and inline media go to the Skyhammer bucket under ``media/<sha256>.<ext>`` (content
addressed, so each file uploads once) and S3 media from other buckets is copied there,
because Bedrock reads training media only from the dataset's own bucket.

Every file read is checked to really be the format it claims (by its first bytes), so a
row can't get another local file uploaded by calling it an image. Rows from a remote
source (``Dataset.from_s3``, ``from_hf``, or a mix including one) were written by someone
else: they may use inline media and, for S3 datasets, objects in the dataset's own
bucket, but not local files or other buckets your credentials can read.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from skyhammer.errors import SkyhammerError

if TYPE_CHECKING:
    from skyhammer.catalog import DatasetLimits
    from skyhammer.session import Session

Record = dict[str, Any]

KINDS = ("image", "video")
# File extension or MIME subtype -> the format name Bedrock uses.
_FORMATS = {
    "image": {"png": "png", "jpg": "jpeg", "jpeg": "jpeg", "gif": "gif", "webp": "webp"},
    "video": {
        "mp4": "mp4",
        "mov": "mov",
        "quicktime": "mov",
        "mkv": "mkv",
        "x-matroska": "mkv",
        "webm": "webm",
    },
}
_EXT = {"jpeg": "jpg"}
_DATA_URL = re.compile(r"data:(image|video)/([\w.+-]+);base64,(.*)", re.S)


@dataclass(frozen=True)
class MediaRef:
    """One image or video reference in a row."""

    kind: str  # "image" or "video"
    url: str
    format: str | None  # Bedrock format name, when it can be told

    @property
    def is_s3(self) -> bool:
        return self.url.startswith("s3://")

    @property
    def is_inline(self) -> bool:
        return self.url.startswith("data:")

    @property
    def is_remote(self) -> bool:
        return urlparse(self.url).scheme in ("http", "https")

    def local_path(self, base_dir: Path | None) -> Path | None:
        if self.is_s3 or self.is_inline or self.is_remote:
            return None
        if self.url.startswith("file://"):
            return Path(unquote(urlparse(self.url).path))
        path = Path(self.url).expanduser()
        return path if path.is_absolute() or base_dir is None else base_dir / path

    def data(self, base_dir: Path | None) -> bytes | None:
        """The bytes of a local or inline reference (None for S3)."""
        if self.is_inline:
            m = _DATA_URL.match(self.url)
            if not m:
                raise SkyhammerError(f"Malformed {self.kind} data URL")
            try:
                return base64.b64decode(m.group(3), validate=True)
            except binascii.Error as e:
                raise SkyhammerError(f"Malformed base64 in {self.kind} data URL") from e
        path = self.local_path(base_dir)
        return path.read_bytes() if path is not None else None


def sniff(head: bytes) -> set[str]:
    """Formats the first bytes of a file are consistent with (empty if no media format).
    Rows say what a file is; this checks it, so a row can't pass off another local file
    (a credentials file, say) as an image and get it uploaded."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return {"png"}
    if head.startswith(b"\xff\xd8\xff"):
        return {"jpeg"}
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return {"gif"}
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return {"webp"}
    if head[4:8] == b"ftyp":  # ISO media: MP4 and QuickTime share the container
        return {"mp4", "mov"}
    if head.startswith(b"\x1a\x45\xdf\xa3"):  # EBML: Matroska and WebM
        return {"mkv", "webm"}
    return set()


def _content_problem(ref: MediaRef, head: bytes) -> str | None:
    if ref.format in sniff(head):
        return None
    return f"{_short(ref.url)}: contents aren't a {ref.format} {ref.kind}"


def _format(kind: str, url: str, explicit: str | None) -> str | None:
    formats = _FORMATS[kind]
    if explicit:
        e = explicit.lower()
        return formats.get(e) or (e if e in formats.values() else None)
    m = _DATA_URL.match(url)
    if m:
        return formats.get(m.group(2).lower())
    name = urlparse(url).path or url
    ext = name.rsplit(".", 1)[-1].lower() if "." in name.rsplit("/", 1)[-1] else ""
    return formats.get(ext)


def _message_lists(record: Any) -> Iterator[tuple[str, list[Any]]]:
    """(where, messages) for every message list in a chat, DPO or prompt row."""
    if not isinstance(record, dict):
        return
    for where, value in (
        ("messages", record.get("messages")),
        (
            "input",
            (record.get("input") or {}).get("messages")
            if isinstance(record.get("input"), dict)
            else None,
        ),
        ("preferred_output", record.get("preferred_output")),
        ("non_preferred_output", record.get("non_preferred_output")),
    ):
        if isinstance(value, list):
            yield where, value


def media_parts(record: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """(role, part) for every ``image_url``/``video_url`` part in a row."""
    for _, messages in _message_lists(record):
        for m in messages:
            if not isinstance(m, dict) or not isinstance(m.get("content"), list):
                continue
            for part in m["content"]:
                if isinstance(part, dict) and part.get("type") in ("image_url", "video_url"):
                    yield str(m.get("role")), part


def ref_of(part: dict[str, Any]) -> MediaRef:
    kind = str(part["type"]).removesuffix("_url")
    value = part.get(part["type"])
    url = value.get("url") if isinstance(value, dict) else value
    if not isinstance(url, str) or not url:
        raise SkyhammerError(f"{part['type']} part needs a url")
    explicit = value.get("format") if isinstance(value, dict) else None
    return MediaRef(kind, url, _format(kind, url, explicit))


def has_media(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") in ("image_url", "video_url") for p in content
    )


def kinds(record: Any) -> set[str]:
    out = set()
    for _, part in media_parts(record):
        out.add(str(part["type"]).removesuffix("_url"))
    return out


def remote_bucket(source: str | None) -> str | None:
    """For a dataset read from S3, its bucket; "" for other remote sources (Hugging Face);
    None for local files and datasets built in memory."""
    if not source or "://" not in source:
        return None
    return source.removeprefix("s3://").split("/", 1)[0] if source.startswith("s3://") else ""


def check(
    record: Any,
    limits: DatasetLimits,
    *,
    sft: bool,
    base_dir: Path | None,
    source: str | None = None,
) -> list[str]:
    """Problems with a row's media against ``limits`` (empty when it has none).

    Rows from a remote ``source`` (S3, Hugging Face) were written by someone else, so they
    may not point at files on this machine or at other S3 buckets your credentials can
    read: only inline media and, for S3 datasets, objects in the dataset's own bucket."""
    found = list(media_parts(record))
    if not found:
        return []
    if not limits.media:
        return ["has images or video, but this model's fine-tuning takes text only"]
    if not sft:
        return ["images and video are supported for supervised fine-tuning (sft) only"]
    problems: list[str] = []
    counts = dict.fromkeys(KINDS, 0)
    for role, part in found:
        try:
            ref = ref_of(part)
        except SkyhammerError as e:
            problems.append(str(e))
            continue
        allowed = limits.media.get(ref.kind)
        if allowed is None:
            problems.append(f"{ref.kind}s aren't supported for this model")
            continue
        counts[ref.kind] += 1
        if role != "user":
            problems.append(f"{ref.kind} in a {role!r} message; media goes in user messages")
        if ref.is_remote:
            problems.append(f"{ref.url}: download it first, or use a local path or s3:// URI")
            continue
        remote = remote_bucket(source)
        if remote is not None:
            if ref.is_s3 and ref.url.removeprefix("s3://").split("/", 1)[0] != remote:
                problems.append(
                    f"{_short(ref.url)}: a dataset read from {source} can only use media in "
                    "its own bucket"
                )
                continue
            if not ref.is_s3 and not ref.is_inline:
                problems.append(
                    f"{_short(ref.url)}: a dataset read from {source} can't use local files"
                )
                continue
        if ref.format not in allowed.formats:
            problems.append(
                f"{_short(ref.url)}: {ref.kind} format {ref.format or 'unknown'} isn't "
                f"supported ({', '.join(allowed.formats)}); pass {{'format': ...}} if the "
                "extension is misleading"
            )
        if ref.is_s3:
            continue  # checked by the service; reading it here would download it
        path = ref.local_path(base_dir)
        if path is not None and not path.is_file():
            problems.append(
                f"{ref.url}: file not found" + (f" (in {base_dir})" if base_dir else "")
            )
            continue
        try:
            if path is not None:
                size = path.stat().st_size
                with path.open("rb") as f:
                    head = f.read(16)
            else:
                data = ref.data(base_dir) or b""
                size, head = len(data), data[:16]
        except (SkyhammerError, OSError) as e:
            problems.append(str(e))
            continue
        if ref.format in allowed.formats and (bad := _content_problem(ref, head)):
            problems.append(bad)
        if size > allowed.max_mb * 2**20:
            problems.append(
                f"{_short(ref.url)}: {size / 2**20:.1f} MB is over the {allowed.max_mb:g} MB "
                f"{ref.kind} limit"
            )
    for kind, n in counts.items():
        allowed = limits.media.get(kind)
        if allowed and n > allowed.per_sample:
            problems.append(f"{n} {kind}s in one row; at most {allowed.per_sample}")
    return problems


def _short(url: str) -> str:
    return url if len(url) <= 60 else url[:40] + "..." + url[-12:]


def absolutize(record: Any, base_dir: Path | None) -> Any:
    """``record`` with relative local media paths made absolute against ``base_dir`` (for
    rows moved into another dataset, such as a mix)."""
    if base_dir is None or not any(True for _ in media_parts(record)):
        return record
    out = copy.deepcopy(record)
    for _, part in media_parts(out):
        ref = ref_of(part)
        path = ref.local_path(base_dir)
        if path is not None and not ref.url.startswith("file://"):
            value = part[part["type"]]
            if isinstance(value, dict):
                value["url"] = str(path)
            else:
                part[part["type"]] = str(path)
    return out


def digest(rows: list[Any], base_dir: Path | None, source: str | None = None) -> str | None:
    """Hash of the local media the rows point to, so changing an image's contents makes a
    new dataset version (and a new job). None when there is none (or for remote datasets,
    which can't use local files)."""
    if remote_bucket(source) is not None:
        return None
    h = hashlib.sha256()
    seen = False
    for record in rows:
        for _, part in media_parts(record):
            try:
                ref = ref_of(part)
                data = ref.data(base_dir) if not ref.is_inline else None  # inline: in the row
            except (SkyhammerError, OSError):
                continue  # reported by validation
            if data is not None:
                h.update(ref.url.encode() + b"\0" + hashlib.sha256(data).digest())
                seen = True
    return h.hexdigest() if seen else None


# ---- upload ----


def stage(
    records: list[Record],
    session: Session,
    region: str,
    bucket: str,
    kms_key_arn: str | None,
    base_dir: Path | None,
    source: str | None = None,
) -> list[Record]:
    """Copies of ``records`` with every media reference uploaded to (or copied into)
    ``bucket`` and rewritten to ``{"url": "s3://...", "format": ..., "bucket_owner": ...}``.
    Idempotent: keys are content hashes, and existing objects aren't uploaded again."""
    s3 = session.client("s3", region=region)
    extra = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key_arn} if kms_key_arn else {}
    owner = session.account_id
    done: dict[str, str] = {}

    def exists(key: str) -> bool:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def put(ref: MediaRef) -> str:
        assert ref.format is not None  # validated
        ext = _EXT.get(ref.format, ref.format)
        remote = remote_bucket(source)
        if ref.is_s3:
            src_bucket, src_key = ref.url.removeprefix("s3://").split("/", 1)
            if src_bucket == bucket:
                return ref.url
            if remote is not None and src_bucket != remote:
                raise SkyhammerError(f"{ref.url}: not in {source}'s bucket")
            etag = s3.head_object(Bucket=src_bucket, Key=src_key)["ETag"]
            name = hashlib.sha256(f"{ref.url}|{etag}".encode()).hexdigest()
            key = f"media/s3/{name}.{ext}"
            if not exists(key):
                s3.copy_object(
                    Bucket=bucket,
                    Key=key,
                    CopySource={"Bucket": src_bucket, "Key": src_key},
                    **extra,
                )
            return f"s3://{bucket}/{key}"
        if remote is not None and not ref.is_inline:
            raise SkyhammerError(f"{ref.url}: a dataset read from {source} can't use local files")
        data = ref.data(base_dir)
        assert data is not None
        if bad := _content_problem(ref, data[:16]):  # re-checked: files can change
            raise SkyhammerError(bad)
        key = f"media/{hashlib.sha256(data).hexdigest()}.{ext}"
        if not exists(key):
            s3.put_object(Bucket=bucket, Key=key, Body=data, **extra)
        return f"s3://{bucket}/{key}"

    out = []
    for record in records:
        if not any(True for _ in media_parts(record)):
            out.append(record)
            continue
        staged: Record = copy.deepcopy(record)
        for _, part in media_parts(staged):
            ref = ref_of(part)
            if ref.url not in done:
                done[ref.url] = put(ref)
            part[part["type"]] = {"url": done[ref.url], "format": ref.format, "bucket_owner": owner}
        out.append(staged)
    return out
