from __future__ import annotations

from skyhammer.backends.base import (
    Backend,
    DeploymentRef,
    DeploymentType,
    DeployOptions,
)
from skyhammer.catalog import BackendKind
from skyhammer.session import Session


def get_backend(kind: BackendKind, session: Session, region: str | None = None) -> Backend:
    if kind is BackendKind.BEDROCK:
        from skyhammer.backends.bedrock import BedrockNativeBackend

        return BedrockNativeBackend(session, region)
    if kind is BackendKind.MANTLE:
        from skyhammer.backends.mantle import BedrockMantleBackend

        return BedrockMantleBackend(session, region)
    from skyhammer.backends.sagemaker import SageMakerServerlessBackend

    return SageMakerServerlessBackend(session, region)


__all__ = ["Backend", "DeployOptions", "DeploymentRef", "DeploymentType", "get_backend"]
