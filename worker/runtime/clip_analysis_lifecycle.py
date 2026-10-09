from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable

from shared.boundary import Boundary, isolate
from worker.pipeline.output.evidence.clip_analysis_artifact import ClipAnalysisArtifactIdentity
from worker.runtime.clip_analysis_execution import settle_job
from worker.runtime.clip_analysis_process import ClipAnalysisJob
from worker.runtime.clip_analysis_supervisor_status import ClipAnalysisStatus

LOGGER = logging.getLogger(__name__)


def settle(
    job: ClipAnalysisJob,
    status: ClipAnalysisStatus,
    process: subprocess.Popen[bytes] | None,
    *,
    identity: Callable[[ClipAnalysisJob], ClipAnalysisArtifactIdentity],
    terminate: Callable[[subprocess.Popen[bytes]], None],
) -> tuple[ClipAnalysisStatus, bool]:
    return settle_job(job, status, process, identity=identity, terminate=terminate)


def retry_teardown(
    process: subprocess.Popen[bytes],
    terminate: Callable[[subprocess.Popen[bytes]], None],
    *,
    attempts: int = 3,
) -> bool:
    for _ in range(attempts):
        with isolate(
            Boundary.OPTIONAL_FEATURE, stage="clip_analysis_teardown", level=logging.ERROR
        ) as attempt:
            terminate(process)
        if not attempt.failed:
            return True
    return False
