"""Real OpenCV decode-capability probe for the ``opencv`` decode backend.

This is a faithful port of the ``opencv`` branch of ``default_decode_probe``
in ``edge/runtime/profile/registry.py:151-157`` (pre-migration reference,
read-only -- ``edge/`` is scheduled for deletion once migration completes).
That reference deliberately keeps this check minimal: whether ``cv2`` can be
imported in this process at all. It does *not* additionally probe
``cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG)``, and this port preserves
that same minimal semantics rather than inventing a stricter check -- per the
"port, don't redesign" directive for this stage. (Known limitation inherited
unchanged from the reference: a ``cv2`` build present but missing its FFMPEG
video I/O backend would still probe ``True`` here; flagged upstream rather
than silently fixed, since fixing it would be a redesign, not a port.)

``CpuAvAdapter`` (``worker/adapters/decode/cpu_av/adapter.py``) always opens
an RTSP source via ``cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)``
(``CpuAvAdapter._open_capture``). This probe answers, before any camera is
allowed to start, whether ``cv2`` itself is even importable in this process
-- it is the composition root's real signal for the ``decode_capability``
bootstrap stage (``worker/runtime/bootstrap.py``) on the ``cpu`` and ``mps``
profiles, both of which resolve to the ``opencv`` decode policy
(``worker/runtime/profile/registry.py``: ``PROFILE_REGISTRY``).

``probe_opencv_ffmpeg_capability`` returns ``available=True`` only when
``cv2`` imports without error in this process (never a hypothetical target
install) -- a missing or broken native OpenCV install makes every
subsequent ``cv2.VideoCapture`` call fail regardless of build flags, so
import failure alone is disqualifying and import success is, per the ported
reference, sufficient. Cross-checked against this repo's real, non-mocked
RTSP fixture: on this host, ``cv2`` (4.13.0, FFMPEG-backed) imports cleanly,
and separately-verified real ``cv2.VideoCapture`` calls against a live RTSP
stream on this same host do open and read real frames -- so ``True`` here is
not just import-success, it matches this host's actual decode capability.

A ``True`` result means "this process can import OpenCV"; it does not, and
cannot without a live URL, prove any particular camera stream is reachable
-- that remains a per-camera concern handled when the ingest loop opens its
capture.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(frozen=True, slots=True)
class OpenCvCapability:
    available: bool
    reason: str


# The probed ``cv2`` module, typed loosely: only the import itself is
# exercised, and pinning a narrower Protocol here would not make the real
# ``cv2`` extension module conform to it any more than duck typing already
# does.
Cv2Importer: TypeAlias = Callable[[], Any]


def _import_cv2() -> Any:
    import cv2

    return cv2


def probe_opencv_ffmpeg_capability(*, importer: Cv2Importer = _import_cv2) -> OpenCvCapability:
    """Real signal for whether this process can import OpenCV.

    ``importer`` defaults to the real ``import cv2`` and is injectable only so
    tests can exercise the import-failure branch without needing a broken
    OpenCV install -- mirrors the ``ProbeRunner``/``CaptureFactory``
    injection pattern already used elsewhere in this package
    (``worker/adapters/decode/cpu_av/adapter.py``,
    ``worker/adapters/decode/nvdec_cuvid/probe.py``). Production callers
    never pass ``importer``.

    Catches ``Exception`` broadly (not just ``ImportError``), matching the
    ported reference (``edge/runtime/profile/registry.py:155``) -- native
    extension imports can fail with platform-specific errors (e.g. ``OSError``
    from a broken shared library) beyond plain ``ImportError``, and this
    probe must never itself crash the bootstrap sequence.
    """
    try:
        _ = importer()
    except Exception as exc:  # noqa: BLE001 - decode probe must never break startup
        return OpenCvCapability(False, str(exc))
    return OpenCvCapability(True, "OpenCV is available")


__all__ = ["Cv2Importer", "OpenCvCapability", "probe_opencv_ffmpeg_capability"]
