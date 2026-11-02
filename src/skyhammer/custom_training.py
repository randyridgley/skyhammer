"""Your own training code (TR-17): any loop you like, run as a SageMaker training job and
imported into Bedrock like every other open-weight fine-tune.

Two ways to bring it:

- **A script** (``source_dir`` + ``entry_point``): Skyhammer uploads the directory and
  runs the entry point in the Hugging Face PyTorch training container (PyTorch 2.8,
  transformers 4.56, peft, accelerate); a ``requirements.txt`` in the directory is
  installed first. Hyperparameters arrive as command-line arguments (``--epochs 2``).
  Pass ``image=`` to use another container built on the SageMaker training toolkit.
- **A container** (``image`` only): your image runs with its own entry point, following
  the SageMaker training contract; hyperparameters are in
  ``/opt/ml/input/config/hyperparameters.json``.

Either way the job sees:

- ``SM_CHANNEL_MODEL``: the base model's Hugging Face weights (open JumpStart artifacts)
- ``SM_CHANNEL_TRAIN`` (and ``SM_CHANNEL_VALIDATION``): your dataset as JSONL, rows unchanged
- ``SM_MODEL_DIR`` (``/opt/ml/model``): write the trained model here in Hugging Face format
  (merged safetensors, ``config.json``, tokenizer files); Bedrock imports it from there.
  Optional ``train_metrics.jsonl`` lines like ``{"train/global_step": 10, "train/loss":
  1.2}`` show up in ``job.metrics()``.

Jobs bill per instance-hour and always stop after ``max_hours`` (default 4).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skyhammer import pretrain
from skyhammer.catalog import DatasetLimits
from skyhammer.errors import SkyhammerError

if TYPE_CHECKING:
    from skyhammer.hub import HubModel
    from skyhammer.jobs import TrainingRequest
    from skyhammer.session import Session

MAX_BUNDLE_BYTES = 100 * 2**20
# Rows go to your code unchanged; the bounds only stop accidental uploads.
LIMITS = DatasetLimits(
    min_samples=1,
    max_samples=10_000_000,
    max_tokens_per_sample=10_000_000,
    source="Skyhammer custom training (your code reads the rows)",
)
_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
}
_SKIP_SUFFIXES = (".pyc", ".pyo")


@dataclass(frozen=True)
class TrainingCode:
    """Where your training code lives: a directory with an entry point, a container image,
    or a directory run in your own image. ``environment`` sets extra variables in the job."""

    source_dir: str | None = None
    entry_point: str | None = "train.py"
    image: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    # The code as it was when this object was made: later edits to the directory don't
    # change what a planned job uploads or which job name it gets.
    _bundle: bytes | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.source_dir is None:
            if self.image is None:
                raise SkyhammerError(
                    "Pass source_dir (a directory with your script), image, or both"
                )
            return
        root = Path(self.source_dir)
        if not root.is_dir():
            raise SkyhammerError(f"source_dir {self.source_dir!r} is not a directory")
        if not self.entry_point or not (root / self.entry_point).is_file():
            raise SkyhammerError(f"{self.entry_point!r} not found in {self.source_dir}")
        object.__setattr__(self, "_bundle", self._make_bundle())

    @property
    def script_mode(self) -> bool:
        return self.source_dir is not None

    def files(self) -> list[Path]:
        """The files that go into the bundle, sorted (caches and VCS metadata left out)."""
        assert self.source_dir is not None
        root = Path(self.source_dir)
        out = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            out += [
                Path(dirpath, f)
                for f in sorted(filenames)
                if not f.endswith(_SKIP_SUFFIXES) and f != ".DS_Store"
            ]
        return out

    def bundle(self) -> bytes:
        """sourcedir.tar.gz of ``source_dir`` as of when this object was made; reproducible,
        so its hash identifies the code."""
        assert self._bundle is not None, "container mode has no code bundle"
        return self._bundle

    def _make_bundle(self) -> bytes:
        assert self.source_dir is not None
        root = Path(self.source_dir)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
            for path in self.files():
                data = path.read_bytes()
                info = tarfile.TarInfo(path.relative_to(root).as_posix())
                info.size, info.mtime = len(data), 0
                info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
                tar.addfile(info, io.BytesIO(data))
        data = buf.getvalue()
        if len(data) > MAX_BUNDLE_BYTES:
            raise SkyhammerError(
                f"{self.source_dir} is {len(data) / 2**20:.0f} MB compressed; keep data and "
                "weights out of it (the dataset and base model arrive as channels)"
            )
        return data[:4] + b"\0\0\0\0" + data[8:]  # zero the gzip timestamp

    def fingerprint(self) -> str:
        code = hashlib.sha256(self.bundle()).hexdigest() if self.script_mode else None
        payload: list[Any] = [
            code,
            self.entry_point if self.script_mode else None,
            self.image,
            sorted(self.environment.items()),
        ]
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def _code(request: TrainingRequest) -> TrainingCode:
    if not isinstance(request.code, TrainingCode):
        raise SkyhammerError("Custom training needs code=TrainingCode(source_dir=..., ...)")
    return request.code


def hyperparameters(request: TrainingRequest) -> dict[str, str]:
    """Every hyperparameter you set (named ones and ``hyperparameters_extra``), passed
    through unchecked: your code defines what they mean. Script mode JSON-encodes them
    (the training toolkit decodes them into typed arguments); containers get strings."""
    hp = request.hyperparameters
    values: dict[str, Any] = {
        **hp.set_fields(),
        **{k: v for k, v in hp.extra.items() if k != "instance_type"},
        "training_type": request.training_type.value,
    }
    if _code(request).script_mode:
        return {k: json.dumps(v) for k, v in sorted(values.items())}
    return {k: v if isinstance(v, str) else json.dumps(v) for k, v in sorted(values.items())}


def check(request: TrainingRequest, hub_model: HubModel) -> dict[str, str]:
    """Everything that can fail before spending: code, base weights, instance."""
    _code(request)
    if request.source_model:
        raise SkyhammerError("Custom training starts from a base model, not an earlier fine-tune")
    if hub_model.gated:
        raise SkyhammerError(
            f"{hub_model.name} has gated weights ({hub_model.license}); custom training "
            "supports open-weight models"
        )
    if hub_model.base is None:
        raise SkyhammerError(f"{hub_model.name} has no Hugging Face weights to train from")
    pretrain.instance_type(request)
    return hyperparameters(request)


def upload_code(session: Session, region: str, code: TrainingCode) -> str | None:
    if not code.script_mode:
        return None
    return pretrain.upload_bundle(session, region, code.bundle(), "custom")


def build_request(
    session: Session,
    region: str,
    request: TrainingRequest,
    hub_model: HubModel,
    *,
    training_uri: str,
    validation_uri: str | None,
    output_uri: str,
    code_uri: str | None,
) -> dict[str, Any]:
    """CreateTrainingJob kwargs for your code (no AWS calls)."""
    hp = check(request, hub_model)
    code = _code(request)
    if code.script_mode:
        if code_uri is None:
            raise SkyhammerError("script mode needs the uploaded code bundle")
        hp = {
            **hp,
            "sagemaker_program": json.dumps(code.entry_point),
            "sagemaker_submit_directory": json.dumps(code_uri),
        }
    return pretrain.job_request(
        session,
        region,
        request,
        hub_model,
        image=code.image or pretrain.IMAGE.format(region=region),
        hyperparameters=hp,
        instance=pretrain.instance_type(request),
        training_uri=training_uri,
        validation_uri=validation_uri,
        output_uri=output_uri,
        environment=dict(code.environment),
    )
