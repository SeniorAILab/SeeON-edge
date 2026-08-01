from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar, final

from worker.adapters.model.errors import FatalAcceleratorError
from worker.interfaces.bus import FrameBus
from worker.interfaces.decode import DecodeAdapter
from worker.pipeline.ingest.registry import ResolvedSource, SourceRegistry
from worker.pipeline.ingest.rtsp import RTSPSource
from worker.pipeline.ingest.rtsp_url import mask_rtsp_url

_DecodeConfigT = TypeVar("_DecodeConfigT")
_DecodeConfigFactory = Callable[[str, ResolvedSource], _DecodeConfigT]


@dataclass(frozen=True, slots=True)
class CapturePolicy:
    max_failures: int = 30
    reconnect_initial_backoff_sec: float = 0.25
    reconnect_max_backoff_sec: float = 5.0
    max_total_reconnects: int | None = None
    target_fps: float = 5.0


@dataclass(frozen=True, slots=True)
class IngestEvent:
    camera_id: str
    event_type: str
    category: str
    detail: str


class IngestReporter(Protocol):
    def mark_starting(self, camera_id: str) -> None: ...

    def mark_ready(self, camera_id: str) -> None: ...

    def mark_degraded(self, camera_id: str, *, category: str) -> None: ...

    def emit(self, event: IngestEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class CameraIngestSpec(Generic[_DecodeConfigT]):
    camera_id: str
    source_id: str
    make_decode_config: _DecodeConfigFactory[_DecodeConfigT]
    policy: CapturePolicy = field(default_factory=CapturePolicy)


@dataclass(frozen=True, slots=True)
class CameraIngestPorts(Generic[_DecodeConfigT]):
    registry: SourceRegistry
    decoder: DecodeAdapter[_DecodeConfigT]
    bus: FrameBus
    reporter: IngestReporter


@final
class CameraIngestLoop(Generic[_DecodeConfigT]):
    def __init__(
        self,
        spec: CameraIngestSpec[_DecodeConfigT],
        ports: CameraIngestPorts[_DecodeConfigT],
    ) -> None:
        self._spec = spec
        self._ports = ports
        self._stop_event = threading.Event()
        self._ready = False
        self._offline = False

    @property
    def camera_id(self) -> str:
        return self._spec.camera_id

    def run(self) -> None:
        self._ports.reporter.mark_starting(self.camera_id)
        try:
            source = self._build_source()
        except FatalAcceleratorError:
            raise
        except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK - boundary
            self._record_source_failure(error)
            return

        try:
            for packet in source:
                if not self._ready:
                    self._ready = True
                    self._ports.reporter.mark_ready(self.camera_id)
                try:
                    self._ports.bus.publish(packet)
                except FatalAcceleratorError:
                    raise
                except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK - boundary
                    self._record_processing_failure(error)
        except FatalAcceleratorError:
            raise
        except Exception as error:  # noqa: BLE001  # noqa: BROAD_EXCEPT_OK - boundary
            self._record_source_failure(error)

    def stop(self) -> None:
        self._stop_event.set()

    def _build_source(self) -> RTSPSource[_DecodeConfigT]:
        resolved = self._ports.registry.resolve(source_id=self._spec.source_id)
        decode_config = self._spec.make_decode_config(self.camera_id, resolved)
        policy = self._spec.policy
        source = RTSPSource(
            decode_config,
            self._ports.decoder,
            max_failures=policy.max_failures,
            reconnect_initial_backoff_sec=policy.reconnect_initial_backoff_sec,
            reconnect_max_backoff_sec=policy.reconnect_max_backoff_sec,
            max_total_reconnects=policy.max_total_reconnects,
            backoff_wait=self._stop_event.wait,
            stop_requested=self._stop_event.is_set,
            target_fps=policy.target_fps,
            pace_wait=self._stop_event.wait,
        )
        source.set_liveness_callbacks(
            on_reconnecting=self._record_reconnecting,
            on_recovered=self._record_recovered,
        )
        return source

    def _record_reconnecting(self, reason: str) -> None:
        self._ready = False
        category = "rtsp_reconnecting"
        self._ports.reporter.mark_degraded(self.camera_id, category=category)
        self._emit_offline_once(category, reason)

    def _record_recovered(self, reason: str) -> None:
        self._ready = True
        self._ports.reporter.mark_ready(self.camera_id)
        if self._offline:
            self._ports.reporter.emit(
                IngestEvent(
                    camera_id=self.camera_id,
                    event_type="camera.recovered",
                    category="rtsp_recovered",
                    detail=reason,
                )
            )
            self._offline = False

    def _record_source_failure(self, error: Exception) -> None:
        self._ready = False
        category = type(error).__name__
        self._ports.reporter.mark_degraded(self.camera_id, category=category)
        safe_source = mask_rtsp_url(self._spec.source_id)
        self._emit_offline_once(category, f"{category} while ingesting {safe_source}")

    def _record_processing_failure(self, error: Exception) -> None:
        category = type(error).__name__
        self._ports.reporter.emit(
            IngestEvent(
                camera_id=self.camera_id,
                event_type="frame.processing_error",
                category=category,
                detail=f"{category} while publishing a decoded frame",
            )
        )

    def _emit_offline_once(self, category: str, detail: str) -> None:
        if self._offline:
            return
        self._offline = True
        self._ports.reporter.emit(
            IngestEvent(
                camera_id=self.camera_id,
                event_type="camera.offline",
                category=category,
                detail=detail,
            )
        )


class _RunnableIngest(Protocol):
    @property
    def camera_id(self) -> str: ...

    def run(self) -> None: ...

    def stop(self) -> None: ...


@final
class IngestSupervisor:
    def __init__(self, loops: Sequence[_RunnableIngest]) -> None:
        self._loops = tuple(loops)
        self._threads: tuple[threading.Thread, ...] = ()

    def run(self) -> None:
        self.start()
        self.join()

    def start(self) -> None:
        if self._threads:
            return
        self._threads = tuple(
            threading.Thread(
                target=loop.run,
                name=f"worker-ingest-{loop.camera_id}",
                daemon=True,
            )
            for loop in self._loops
        )
        for thread in self._threads:
            thread.start()

    def join(self, *, timeout_sec: float | None = None) -> None:
        for thread in self._threads:
            thread.join(timeout=timeout_sec)

    def stop(self, *, join_timeout_sec: float = 1.0) -> None:
        for loop in self._loops:
            loop.stop()
        self.join(timeout_sec=join_timeout_sec)


__all__ = [
    "CameraIngestLoop",
    "CameraIngestPorts",
    "CameraIngestSpec",
    "CapturePolicy",
    "IngestEvent",
    "IngestReporter",
    "IngestSupervisor",
]
