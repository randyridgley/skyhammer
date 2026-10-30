"""``Dataset`` resource: local JSONL -> validated -> backend format -> content-addressed S3."""

from __future__ import annotations

import builtins
import hashlib
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skyhammer.catalog import DatasetLimits, ModelEntry, Technique
from skyhammer.errors import DatasetValidationError, SkyhammerError
from skyhammer.validation import ValidationReport, apply_fixes, parse_jsonl, validate

if TYPE_CHECKING:
    from skyhammer.backends.base import Backend

Record = dict[str, Any]


class Dataset:
    """An OpenAI-format JSONL dataset.

    Content-addressed: the S3 location is derived from the SHA-256 of the source rows, so
    uploading identical data twice is a no-op and changed data becomes a new version.
    """

    def __init__(
        self, name: str, rows: builtins.list[tuple[int, Any]], source: str | None = None
    ) -> None:
        self.name = name
        self.source = source
        self._rows = rows
        self._prepared: builtins.list[Record] | None = None
        self.report: ValidationReport | None = None

    # ---- constructors ----

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None) -> Dataset:
        p = Path(path)
        with p.open(encoding="utf-8") as f:
            rows = parse_jsonl(f)
        return cls(name or p.stem, rows, source=str(p))

    @staticmethod
    def list(session: Any = None) -> builtins.list[dict[str, Any]]:
        """Uploaded datasets: name, content versions (sha256), and formats in the bucket."""
        from skyhammer.session import get_session

        session = session or get_session()
        res = session.resources
        out: dict[tuple[str, str], dict[str, Any]] = {}
        for page in (
            session.client("s3")
            .get_paginator("list_objects_v2")
            .paginate(Bucket=res.bucket, Prefix="datasets/")
        ):
            for obj in page.get("Contents", []):
                parts = obj["Key"].split("/")
                if len(parts) != 4:
                    continue
                _, name, sha, fname = parts
                entry = out.setdefault(
                    (name, sha),
                    {
                        "name": name,
                        "sha256": sha,
                        "formats": [],
                        "bytes": 0,
                        "uploaded": obj["LastModified"],
                    },
                )
                entry["formats"].append(fname.removesuffix(".jsonl"))
                entry["bytes"] += obj["Size"]
                entry["uploaded"] = max(entry["uploaded"], obj["LastModified"])
        return sorted(out.values(), key=lambda e: e["uploaded"], reverse=True)

    @staticmethod
    def lineage(name: str, session: Any = None) -> dict[str, Any]:
        """A dataset's uploaded versions and the jobs/models trained on it (newest first)."""
        from skyhammer import registry
        from skyhammer.session import get_session

        session = session or get_session()
        versions = [d for d in Dataset.list(session) if d["name"] == name]
        jobs = [
            {
                "job": r.job_name,
                "model": r.output_model,
                "base_model": r.base_model,
                "technique": r.technique,
                "dataset_sha256": r.dataset.get("sha256"),
                "rows": r.dataset.get("rows"),
                "distilled_from": r.tags.get("skyhammer:distilled-from")
                or r.tags.get("distilled-from"),
                "created_at": r.created_at,
            }
            for r in registry.list_jobs(session, include_deleted=True)
            if r.dataset.get("name") == name
        ]
        if not versions and not jobs:
            raise SkyhammerError(f"No dataset named {name!r}. Run `skyhammer datasets list`.")
        jobs.sort(key=lambda j: float(j["created_at"] or 0), reverse=True)
        return {"name": name, "versions": versions, "jobs": jobs}

    @classmethod
    def from_s3(cls, uri: str, name: str | None = None, *, session: Any = None) -> Dataset:
        """Read a JSONL dataset already in S3 (no local copy needed)."""
        from skyhammer.session import get_session

        bucket, key = uri.removeprefix("s3://").split("/", 1)
        body = (session or get_session()).client("s3").get_object(Bucket=bucket, Key=key)["Body"]
        rows = parse_jsonl(body.read().decode("utf-8").splitlines())
        return cls(name or Path(key).stem, rows, source=uri)

    @classmethod
    def from_pandas(cls, df: Any, name: str, *, messages_column: str = "messages") -> Dataset:
        """A DataFrame with a ``messages`` column (or any columns matching the technique)."""
        records = df.to_dict(orient="records")
        if messages_column != "messages":
            for r in records:
                r["messages"] = r.pop(messages_column)
        return cls.from_records(records, name)

    @classmethod
    def from_hf(
        cls,
        repo: str,
        *,
        split: str = "train",
        name: str | None = None,
        messages_column: str = "messages",
        limit: int | None = None,
    ) -> Dataset:
        """A Hugging Face dataset (needs ``pip install 'skyhammer[hf]'``)."""
        try:
            from datasets import load_dataset
        except ImportError as e:
            from skyhammer.errors import SkyhammerError

            raise SkyhammerError(
                "Install Hugging Face datasets: pip install 'skyhammer[hf]'"
            ) from e
        ds = load_dataset(repo, split=split)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        records = [dict(r) for r in ds]
        if messages_column != "messages":
            for r in records:
                r["messages"] = r.pop(messages_column)
        return cls.from_records(records, name or repo.replace("/", "-"))

    @classmethod
    def from_records(cls, records: builtins.list[Record], name: str) -> Dataset:
        return cls(name, list(enumerate(records, start=1)))

    # ---- identity ----

    @property
    def sha256(self) -> str:
        h = hashlib.sha256()
        for _, record in self._rows:
            h.update(json.dumps(record, sort_keys=True, default=repr).encode())
            h.update(b"\n")
        return h.hexdigest()

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def num_records(self) -> int:
        """Rows that will be uploaded (after auto-fix), or all rows if not yet prepared."""
        return len(self._prepared) if self._prepared is not None else len(self._rows)

    def __repr__(self) -> str:
        return f"Dataset(name={self.name!r}, rows={len(self)}, sha256={self.sha256[:12]!r})"

    # ---- validation ----

    def validate(
        self,
        model: ModelEntry,
        technique: Technique,
        *,
        is_validation_set: bool = False,
        limits: DatasetLimits | None = None,
    ) -> ValidationReport:
        """Validate against ``limits`` (backend-specific) or the catalog's model limits."""
        limits = limits or model.limits
        if is_validation_set:
            limits = limits.model_copy(update={"min_samples": 1})
        from skyhammer.tokenization import tokenizer_counter

        self.report = validate(self._rows, technique, limits, tokenizer_counter(model.hf_repo))
        return self.report

    def prepare(
        self,
        model: ModelEntry,
        technique: Technique,
        *,
        auto_fix: bool = False,
        is_validation_set: bool = False,
        limits: DatasetLimits | None = None,
    ) -> None:
        """Validate for (model, technique); raise unless valid (or fixable with ``auto_fix``)."""
        report = self.validate(model, technique, is_validation_set=is_validation_set, limits=limits)
        if report.ok:
            self._prepared = [r for _, r in self._rows]
            return
        if auto_fix and report.fixable:
            self._prepared = apply_fixes(self._rows, report)
            return
        raise DatasetValidationError(report)

    def split(self, validation_fraction: float, *, seed: int = 0) -> tuple[Dataset, Dataset]:
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        rows = list(self._rows)
        random.Random(seed).shuffle(rows)
        n_val = max(1, int(len(rows) * validation_fraction))
        return (
            Dataset(f"{self.name}-train", rows[n_val:], self.source),
            Dataset(f"{self.name}-val", rows[:n_val], self.source),
        )

    # ---- upload ----

    def _encode(self, backend: Backend, technique: Technique) -> bytes:
        if self._prepared is None:
            raise RuntimeError("Dataset must be prepared (validated) before upload")
        return b"".join(
            json.dumps(backend.format_record(technique, r)).encode() + b"\n" for r in self._prepared
        )

    def upload(self, backend: Backend, technique: Technique) -> str:
        """Upload in the backend's format; returns the S3 URI. Idempotent.

        Keyed by the hash of the *encoded* bytes: different model limits or auto-fixes
        produce different content and therefore a different object.
        """
        body = self._encode(backend, technique)
        digest = hashlib.sha256(body).hexdigest()
        res = backend.session.resources_for(backend.region)
        s3 = backend.session.client("s3", region=backend.region)
        key = f"datasets/{self.name}/{digest}/{backend.name}-{technique.value}.jsonl"
        uri = f"s3://{res.bucket}/{key}"
        try:
            s3.head_object(Bucket=res.bucket, Key=key)
            return uri
        except s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
        extra = (
            {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": res.kms_key_arn}
            if res.kms_key_arn
            else {}
        )
        s3.put_object(Bucket=res.bucket, Key=key, Body=body, **extra)
        return uri
