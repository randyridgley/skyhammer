"""Live AWS integration tests (account/region from the normal AWS credential chain).

    SKYHAMMER_INTEGRATION=1 uv run pytest tests/integration                 # fast: free/<1¢
    SKYHAMMER_INTEGRATION=1 SKYHAMMER_INTEGRATION_SLOW=1 uv run pytest -m slow  # trains models

Every resource is named ``skyhammer-it-<suite>-<timestamp>`` and deleted by fixtures even
when a test fails (running jobs are cancelled).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator

import pytest

from skyhammer.config import load_settings
from skyhammer.session import Session

log = logging.getLogger(__name__)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    live = os.environ.get("SKYHAMMER_INTEGRATION") == "1"
    slow = os.environ.get("SKYHAMMER_INTEGRATION_SLOW") == "1"
    for item in items:
        if "integration" not in str(item.fspath):
            continue
        item.add_marker(pytest.mark.aws)
        if not live:
            item.add_marker(pytest.mark.skip(reason="set SKYHAMMER_INTEGRATION=1 to hit AWS"))
        elif item.get_closest_marker("slow") and not slow:
            item.add_marker(pytest.mark.skip(reason="set SKYHAMMER_INTEGRATION_SLOW=1 to train"))


@pytest.fixture(scope="session")
def live_session() -> Session:
    return Session(load_settings(region=os.environ.get("SKYHAMMER_IT_REGION", "us-east-1")))


@pytest.fixture
def unique_name(request: pytest.FixtureRequest) -> str:
    suite = request.node.name.removeprefix("test_").replace("_", "-")[:20]
    return f"skyhammer-it-{suite}-{time.strftime('%m%d%H%M%S')}"


@pytest.fixture
def cleanup(live_session: Session) -> Iterator[list[str]]:
    """Append model names; they are cancelled/deleted afterwards whatever happens."""
    from skyhammer.jobs import FineTuningJob, JobState
    from skyhammer.llm import LLM

    names: list[str] = []
    yield names
    for name in names:
        try:
            job = FineTuningJob.get(name, session=live_session)
            if not job.status().state.terminal:
                job.cancel()
                log.warning("cancelled unfinished job %s", job.name)
                continue
            if job.status().state is JobState.COMPLETED:
                LLM.get(name, session=live_session).delete()
        except Exception as e:  # never mask the test failure; report leftovers loudly
            log.error("CLEANUP FAILED for %s: %s - delete it manually", name, e)
