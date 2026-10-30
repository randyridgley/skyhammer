"""skyhammer: fine-tune Amazon Nova and open-weight models on AWS, served by Bedrock."""

from skyhammer.auth import generate_api_key, openai_client
from skyhammer.batch import BatchJob
from skyhammer.catalog import CATALOG, Technique, TrainingType
from skyhammer.config import configure
from skyhammer.dataset import Dataset
from skyhammer.errors import (
    CostLimitExceededError,
    DatasetValidationError,
    JobFailedError,
    SkyhammerError,
    SpendLimitExceededError,
    ThrottledError,
    UnsupportedParameterError,
    UnsupportedRouteError,
)
from skyhammer.evaluation import EvalJob
from skyhammer.graders import Grader, grader, reference_answer
from skyhammer.jobs import FineTuningJob, JobState
from skyhammer.llm import LLM

__version__ = "0.0.1"

__all__ = [
    "CATALOG",
    "LLM",
    "BatchJob",
    "CostLimitExceededError",
    "Dataset",
    "DatasetValidationError",
    "EvalJob",
    "FineTuningJob",
    "Grader",
    "JobFailedError",
    "JobState",
    "SkyhammerError",
    "SpendLimitExceededError",
    "Technique",
    "ThrottledError",
    "TrainingType",
    "UnsupportedParameterError",
    "UnsupportedRouteError",
    "configure",
    "generate_api_key",
    "grader",
    "openai_client",
    "reference_answer",
]
