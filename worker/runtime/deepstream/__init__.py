"""Production-unwired DeepStream child supervision for C5 dark QA."""

from worker.runtime.deepstream.config import ChildConfig
from worker.runtime.deepstream.supervisor import DeepStreamChildSupervisor

__all__ = ["ChildConfig", "DeepStreamChildSupervisor"]
