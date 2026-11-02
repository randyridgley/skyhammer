"""``Dataset`` resource: local JSONL -> validated -> backend format -> content-addressed S3."""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skyhammer import media
from skyhammer.catalog import DatasetLimits, ModelEntry, Technique
from skyhammer.errors import DatasetValidationError, SkyhammerError
from skyhammer.validation import ValidationReport, apply_fixes, parse_jsonl, validate

if TYPE_CHECKING:
    from skyhammer.backends.base import Backend
    from skyhammer.llm import LLM
    from skyhammer.session import Session

Record = dict[str, Any]
_UNSET = object()


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
        # Set by ``mix``: the parts (name, sha256, rows taken, share), kept in job records.
        self.mixture: builtins.list[dict[str, Any]] | None = None
        self._media_digest: object = _UNSET

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
        """A dataset's uploaded versions and the jobs/models trained on it (newest first),
        including jobs trained on a ``mix`` that drew from it (``via`` names the mix)."""
        from skyhammer import registry
        from skyhammer.session import get_session

        session = session or get_session()
        versions = [d for d in Dataset.list(session) if d["name"] == name]
        jobs = []
        for r in registry.list_jobs(session, include_deleted=True):
            direct = r.dataset.get("name") == name
            part: dict[str, Any] = next(
                (p for p in r.dataset.get("mixture") or [] if p["name"] == name), {}
            )
            if not direct and not part:
                continue
            jobs.append(
                {
                    "job": r.job_name,
                    "model": r.output_model,
                    "base_model": r.base_model,
                    "technique": r.technique,
                    "dataset_sha256": r.dataset.get("sha256") if direct else part["sha256"],
                    "rows": r.dataset.get("rows") if direct else part["rows"],
                    "via": None if direct else r.dataset.get("name"),
                    "distilled_from": r.tags.get("skyhammer:distilled-from")
                    or r.tags.get("distilled-from"),
                    "created_at": r.created_at,
                }
            )
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
        ds = cls.from_records(records, name or repo.replace("/", "-"))
        ds.source = f"hf://{repo}"
        return ds

    @classmethod
    def from_records(cls, records: builtins.list[Record], name: str) -> Dataset:
        return cls(name, list(enumerate(records, start=1)))

    @classmethod
    def synthesize(
        cls,
        model: str | LLM,
        *,
        n: int,
        task: str | None = None,
        documents: Dataset | builtins.list[str] | str | None = None,
        examples: Dataset | None = None,
        name: str = "synthetic",
        session: Session | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """About ``n`` SFT conversations written by a Bedrock ``model``: questions grounded
        in ``documents`` (contextual), or following a ``task`` description and the style of
        a few seed ``examples``. Cached in S3; ``max_cost_usd`` bounds the spend up front.
        Other options (``system``, ``per_call``, ``max_tokens``, ``temperature``) are in
        ``skyhammer.synthetic.synthesize``."""
        from skyhammer.llm import LLM
        from skyhammer.synthetic import synthesize

        generator = LLM(model, session=session) if isinstance(model, str) else model
        return synthesize(
            generator, n=n, task=task, documents=documents, examples=examples, name=name, **kwargs
        )

    @classmethod
    def mix(
        cls,
        parts: Mapping[Dataset, float] | Iterable[tuple[Dataset, float]],
        *,
        n: int | None = None,
        name: str | None = None,
        seed: int = 0,
        repeat: bool = False,
    ) -> Dataset:
        """One dataset drawn from several by weight, for example 70% your domain data and
        30% general chat so the model keeps its general skills while specializing.

        Weights are relative (``{support: 7, chat: 3}`` equals ``{support: 0.7, chat: 0.3}``).
        ``n`` defaults to the largest mix that uses each row at most once; a larger ``n``
        needs ``repeat=True`` (rows of a small part are then used more than once). Rows are
        drawn and shuffled with ``seed``, so the same inputs give the same dataset (and
        version). The parts are recorded in ``mixture`` and in the records of jobs trained
        on it, so ``Dataset.lineage()`` of a part lists them too."""
        pairs = builtins.list(parts.items() if isinstance(parts, Mapping) else parts)
        if not pairs:
            raise SkyhammerError("Dataset.mix needs at least one (dataset, weight)")
        for ds, w in pairs:
            if w < 0 or not math.isfinite(w):
                raise SkyhammerError(f"Weight for {ds.name!r} must be a non-negative number")
            if w > 0 and len(ds) == 0:
                raise SkyhammerError(f"{ds.name!r} has no rows")
        total = sum(w for _, w in pairs)
        if total <= 0:
            raise SkyhammerError("At least one weight must be positive")
        shares = [(ds, w / total) for ds, w in pairs if w > 0]
        if n is None:
            # Largest mix without repeats: the part that runs out first bounds it.
            n = min(math.floor(len(ds) / share + 1e-9) for ds, share in shares)
        if n < 1:
            raise SkyhammerError("n must be at least 1")
        counts = _allocate(n, [share for _, share in shares])
        rng = random.Random(seed)
        rows: builtins.list[Any] = []
        mixture = []
        for (ds, share), count in zip(shares, counts, strict=True):
            if count > len(ds) and not repeat:
                raise SkyhammerError(
                    f"A {n}-row mix needs {count} rows from {ds.name!r}, which has "
                    f"{len(ds)}. Lower n, or pass repeat=True to use its rows more than once."
                )
            # Relative image paths keep pointing at the part's own files.
            source = [media.absolutize(r, ds.base_dir) for _, r in ds._rows]
            taken: builtins.list[Any] = []
            while len(taken) < count:  # whole shuffled passes, then a partial one
                rng.shuffle(source)
                taken += source[: count - len(taken)]
            rows += taken
            mixture.append(
                {"name": ds.name, "sha256": ds.sha256, "rows": count, "share": round(share, 6)}
            )
        rng.shuffle(rows)
        mixed = cls(
            name or "+".join(ds.name for ds, _ in shares), builtins.list(enumerate(rows, 1))
        )
        mixed.mixture = mixture
        # Rows from a remote source keep its media restrictions (skyhammer.media.check).
        mixed.source = next(
            (ds.source for ds, _ in shares if media.remote_bucket(ds.source) is not None), None
        )
        return mixed

    # ---- identity ----

    @property
    def sha256(self) -> str:
        h = hashlib.sha256()
        for _, record in self._rows:
            h.update(json.dumps(record, sort_keys=True, default=repr).encode())
            h.update(b"\n")
        # Rows reference image/video files by path; their contents are part of the version.
        if self._media_digest is _UNSET:
            self._media_digest = media.digest(
                [r for _, r in self._rows], self.base_dir, self.source
            )
        if self._media_digest:
            h.update(str(self._media_digest).encode())
        return h.hexdigest()

    @property
    def base_dir(self) -> Path | None:
        """Where relative image and video paths resolve: the JSONL file's directory, or
        the working directory for datasets built in memory."""
        if self.source and "://" not in self.source and Path(self.source).is_file():
            return Path(self.source).resolve().parent
        return None

    def __len__(self) -> int:
        return len(self._rows)

    def records(self) -> builtins.list[Record]:
        """The rows as plain dicts (copies; editing them doesn't change the dataset)."""
        return [json.loads(json.dumps(r)) for _, r in self._rows]

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

        self.report = validate(
            self._rows,
            technique,
            limits,
            tokenizer_counter(model.hf_repo),
            base_dir=self.base_dir,
            source=self.source,
        )
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
        train = Dataset(f"{self.name}-train", rows[n_val:], self.source)
        train.mixture = self.mixture  # the parts it was drawn from, for lineage
        return train, Dataset(f"{self.name}-val", rows[:n_val], self.source)

    # ---- upload ----

    def _encode(
        self,
        backend: Backend,
        technique: Technique,
        records: builtins.list[Record] | None = None,
    ) -> bytes:
        if self._prepared is None:
            raise RuntimeError("Dataset must be prepared (validated) before upload")
        return b"".join(
            json.dumps(backend.format_record(technique, r)).encode() + b"\n"
            for r in (self._prepared if records is None else records)
        )

    def upload(self, backend: Backend, technique: Technique) -> str:
        """Upload in the backend's format; returns the S3 URI. Idempotent.

        Keyed by the hash of the *encoded* bytes: different model limits or auto-fixes
        produce different content and therefore a different object.
        """
        if self._prepared is None:
            raise RuntimeError("Dataset must be prepared (validated) before upload")
        res = backend.session.resources_for(backend.region)
        records = self._prepared
        if any(any(True for _ in media.media_parts(r)) for r in records):
            # Images and video go next to the dataset (Bedrock reads them from its bucket).
            records = media.stage(
                records,
                backend.session,
                backend.region,
                res.bucket,
                res.kms_key_arn,
                self.base_dir,
                self.source,
            )
        body = self._encode(backend, technique, records)
        digest = hashlib.sha256(body).hexdigest()
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


def _allocate(n: int, shares: builtins.list[float]) -> builtins.list[int]:
    """Split ``n`` rows by ``shares`` (summing to 1), rounding by largest remainder so the
    counts add up to ``n`` exactly."""
    exact = [n * s for s in shares]
    counts = [math.floor(x) for x in exact]
    by_remainder = sorted(range(len(shares)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in by_remainder[: n - sum(counts)]:
        counts[i] += 1
    return counts
