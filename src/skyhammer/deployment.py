"""``Deployment`` resource: an invocable endpoint for a custom model."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from skyhammer.backends.base import DeploymentRef
from skyhammer.errors import JobFailedError

# On-demand custom model deployments become Active; Provisioned Throughputs InService.
READY = ("Active", "InService")

if TYPE_CHECKING:
    from skyhammer.backends.base import Backend


class Deployment:
    def __init__(self, ref: DeploymentRef, backend: Backend) -> None:
        self.ref = ref
        self._backend = backend

    @property
    def arn(self) -> str:
        return self.ref.arn

    @property
    def name(self) -> str:
        return self.ref.name

    def refresh(self) -> Deployment:
        self.ref = self._backend.describe_deployment(self.ref.arn)
        return self

    def wait(self, *, poll_seconds: float = 15, timeout_seconds: float = 3600) -> Deployment:
        deadline = time.monotonic() + timeout_seconds
        while self.refresh().ref.status not in READY:
            if self.ref.status == "Failed":
                raise JobFailedError(self.name, "deployment failed")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Deployment {self.name!r} not active after timeout")
            time.sleep(poll_seconds)
        return self

    def delete(self) -> None:
        self._backend.delete_deployment(self.ref.arn)

    def __repr__(self) -> str:
        return f"Deployment(name={self.name!r}, status={self.ref.status!r})"
