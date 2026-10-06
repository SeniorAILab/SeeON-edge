from __future__ import annotations

import ctypes
import os
import signal
import threading


class ClipAnalysisPdeathsigUnavailable(RuntimeError):
    ...


class ClipAnalysisBootstrapError(RuntimeError):
    ...


def require_pdeathsig() -> None:
    try:
        _ = ctypes.CDLL(None).prctl
    except (AttributeError, OSError) as exc:
        raise ClipAnalysisPdeathsigUnavailable("pdeathsig_unavailable") from exc


def bootstrap_child(*, expected_parent: int, cpu_index: int, control_fd: int) -> None:
    try:
        _arm_lifetime(expected_parent, cpu_index)
    except Exception:  # noqa: BLE001
        os._exit(3)

    def watch_control_pipe() -> None:
        try:
            while os.read(control_fd, 1):
                pass
        except OSError:
            pass
        os._exit(3)

    threading.Thread(target=watch_control_pipe, name="clip-analysis-control", daemon=True).start()


__all__ = [
    "ClipAnalysisBootstrapError",
    "ClipAnalysisPdeathsigUnavailable",
    "bootstrap_child",
    "require_pdeathsig",
]


def _arm_lifetime(expected_parent: int, cpu_index: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise ClipAnalysisBootstrapError("pdeathsig")
    if os.getppid() != expected_parent:
        raise ClipAnalysisBootstrapError("parent")
    os.sched_setaffinity(0, {cpu_index})
    if os.sched_getaffinity(0) != {cpu_index}:
        raise ClipAnalysisBootstrapError("affinity")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
