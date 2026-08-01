from __future__ import annotations

import threading
from typing import final

from worker.adapters.encode.adapter_errors import (
    EncoderPolicyError,
    EncoderStartError,
    EncoderWriteError,
)
from worker.adapters.encode.csv_segment_index import CsvSegmentIndex, Segment
from worker.adapters.encode.encoder_setup import (
    SessionRequest,
    camera_directory_name,
    parse_encode_policy,
)
from worker.adapters.encode.models import (
    EncodePolicy,
    EncoderGeometry,
    EncoderMetrics,
    SegmentEncoderConfig,
)
from worker.adapters.encode.segment_process import (
    ProcessSpawner,
    SegmentProcessSpec,
    segment_process_args,
    spawn_encoder_process,
)
from worker.adapters.encode.session import FFmpegEncoderSession, SessionGeneration
from worker.types import FramePacket


@final
class FFmpegSegmentEncoder:
    def __init__(
        self,
        config: SegmentEncoderConfig,
        *,
        spawner: ProcessSpawner = spawn_encoder_process,
    ) -> None:
        self.config = config
        self.metrics = EncoderMetrics()
        self._spawner = spawner
        self._lock = threading.RLock()
        self._sessions: dict[str, FFmpegEncoderSession] = {}
        self._generation_by_camera: dict[str, int] = {}
        self._unavailable_cameras: set[str] = set()

    @property
    def unavailable_cameras(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._unavailable_cameras)

    def open(
        self,
        camera: str,
        profile: str,
        geometry: EncoderGeometry,
    ) -> FFmpegEncoderSession:
        if not camera.strip():
            raise EncoderPolicyError("camera id must not be blank")
        encoder = parse_encode_policy(profile)
        with self._lock:
            existing = self._sessions.get(camera)
            if existing is not None and existing.matches(encoder, geometry):
                return existing
            if existing is not None:
                self.metrics.recreates += 1
                self.close_session(existing)
            return self._create_session(camera, encoder, geometry)

    def _create_session(
        self,
        camera: str,
        encoder: EncodePolicy,
        geometry: EncoderGeometry,
    ) -> FFmpegEncoderSession:
        generation = self._generation_by_camera.get(camera, 0) + 1
        state = self._spawn(
            SessionRequest(
                camera=camera,
                encoder=encoder,
                geometry=geometry,
                generation=generation,
            )
        )
        session = FFmpegEncoderSession(self, state)
        self._generation_by_camera[camera] = generation
        self._sessions[camera] = session
        self.metrics.active_sessions += 1
        return session

    def _spawn(self, request: SessionRequest) -> SessionGeneration:
        output_dir = (
            self.config.store_dir
            / camera_directory_name(request.camera)
            / f"gen-{request.generation:05d}"
        )
        spec = SegmentProcessSpec(
            self.config,
            request.geometry,
            request.encoder,
            output_dir,
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            process = self._spawner(segment_process_args(spec))
        except EncoderStartError:
            self._record_start_failure(request.camera)
            raise
        except OSError as exc:
            self._record_start_failure(request.camera)
            raise EncoderStartError(
                f"failed to prepare camera encoder ({type(exc).__name__})"
            ) from exc
        self.metrics.process_starts += 1
        self._unavailable_cameras.discard(request.camera)
        return SessionGeneration(
            camera=request.camera,
            encoder=request.encoder,
            geometry=request.geometry,
            number=request.generation,
            process=process,
            index=CsvSegmentIndex(spec.segment_list_path, request.generation),
        )

    def _record_start_failure(self, camera: str) -> None:
        self.metrics.failures += 1
        self._unavailable_cameras.add(camera)

    def write_session(self, session: FFmpegEncoderSession, packet: FramePacket) -> None:
        with self._lock:
            if session.closed:
                raise EncoderWriteError("encoder session is closed")
            geometry = session.geometry
            packet_geometry = EncoderGeometry(packet.width, packet.height, geometry.fps)
            if packet_geometry != geometry:
                self._recreate_session_process(session, packet_geometry)
            image = packet.frame.image
            if image.ndim != 3 or image.shape != (packet.height, packet.width, 3):
                raise EncoderWriteError("frame image does not match packet geometry")
            try:
                session.process.write(image.tobytes(order="C"))
            except EncoderWriteError:
                self._fail_session(session)
                raise
            self._refresh_index(session)

    def _recreate_session_process(
        self,
        session: FFmpegEncoderSession,
        geometry: EncoderGeometry,
    ) -> None:
        self.metrics.recreates += 1
        returncode = session.process.reap()
        self._refresh_index(session)
        if returncode not in (0, None):
            self.metrics.failures += 1
        generation = session.generation + 1
        try:
            state = self._spawn(
                SessionRequest(
                    camera=session.camera,
                    encoder=session.encoder,
                    geometry=geometry,
                    generation=generation,
                )
            )
        except EncoderStartError:
            self._deactivate_session(session)
            raise
        self._generation_by_camera[session.camera] = generation
        session.replace(state)

    def _refresh_index(self, session: FFmpegEncoderSession) -> None:
        self.metrics.finalized_segments += len(session.index.refresh())

    def close_session(self, session: FFmpegEncoderSession) -> None:
        with self._lock:
            if session.closed:
                return
            returncode = session.process.reap()
            self._refresh_index(session)
            if returncode not in (0, None):
                self.metrics.failures += 1
                self._unavailable_cameras.add(session.camera)
            self._deactivate_session(session)

    def _fail_session(self, session: FFmpegEncoderSession) -> None:
        self.metrics.failures += 1
        self._unavailable_cameras.add(session.camera)
        _ = session.process.reap()
        self._refresh_index(session)
        self._deactivate_session(session)

    def _deactivate_session(self, session: FFmpegEncoderSession) -> None:
        if self._sessions.get(session.camera) is session:
            _ = self._sessions.pop(session.camera)
            self.metrics.active_sessions -= 1
        session.mark_closed()

    def segments_for(self, session: FFmpegEncoderSession) -> tuple[Segment, ...]:
        with self._lock:
            if not session.closed:
                self._refresh_index(session)
            return session.index.completed()

    def select_for(
        self,
        session: FFmpegEncoderSession,
        *,
        start_time_sec: float,
        end_time_sec: float,
    ) -> tuple[Segment, ...]:
        with self._lock:
            if not session.closed:
                self._refresh_index(session)
            return session.index.select(
                start_time_sec=start_time_sec,
                end_time_sec=end_time_sec,
            )
__all__ = ["FFmpegSegmentEncoder"]
