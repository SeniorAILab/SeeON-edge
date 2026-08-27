from __future__ import annotations

import hashlib
import logging
import queue
import socket
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, final

from worker.adapters.decode.native_au_codec import (
    AuEnvelope,
    AuFrameError,
    AuKind,
    receive_envelope,
    stream_configuration,
)
from worker.adapters.decode.native_au_mux_template import (
    distinct_parameter_sets,
    native_configuration_signature,
)
from worker.adapters.decode.native_au_progress import NativeAuProgress
from worker.types.source_packet import SourcePacket, SourceStreamConfiguration, StreamEpoch


def _components_of(envelope: AuEnvelope) -> tuple[object, ...]:
    """Signature inputs in the same order native_configuration_signature hashes."""
    parameter_sets = distinct_parameter_sets(envelope.codec_data)
    return (
        envelope.codec,
        envelope.framing,
        envelope.parser_caps,
        len(parameter_sets),
        hashlib.sha256(parameter_sets).hexdigest()[:8],
        envelope.width,
        envelope.height,
        str(envelope.time_base),
    )


_COMPONENT_NAMES = (
    "codec",
    "framing",
    "parser_caps",
    "codec_data_len",
    "codec_data_hash",
    "width",
    "height",
    "time_base",
)

LOGGER = logging.getLogger(__name__)


class NativeAuSink(Protocol):
    def register_camera(self, camera_id: str) -> None: ...
    def append(self, packet: SourcePacket) -> bool: ...
    def roll_epoch(self, epoch: StreamEpoch) -> None: ...


class NativeAuGapHandler(Protocol):
    def __call__(self, camera_id: str, category: str) -> None: ...


class NativeAuStreamDeathHandler(Protocol):
    def __call__(self, category: str) -> None: ...


@dataclass(frozen=True, slots=True)
class _Gap:
    camera_id: str


_Work = AuEnvelope | _Gap


@final
class NativeAuReceiver:
    """Drain AU bytes independently from slower configuration and ring work."""

    def __init__(
        self,
        endpoint: socket.socket,
        worker_boot_id: str,
        sink: NativeAuSink,
        gap_handler: NativeAuGapHandler,
        stream_death_handler: NativeAuStreamDeathHandler | None = None,
        accept_handler: Callable[[str, int, int, int], None] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._worker_boot_id = worker_boot_id
        self._sink = sink
        self._gap_handler = gap_handler
        self._stream_death_handler = stream_death_handler or (lambda _category: None)
        self._accept_handler = accept_handler
        self._stop = threading.Event()
        self._drain_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None
        self._work: queue.Queue[_Work] = queue.Queue(maxsize=256)
        self._epochs: dict[str, StreamEpoch] = {}
        self._sequences: dict[tuple[str, int, int], int] = {}
        self._configurations: dict[tuple[str, int, int], SourceStreamConfiguration] = {}
        self._configuration_signatures: dict[tuple[str, int, int], str] = {}
        self._timeline: dict[tuple[str, int, int], tuple[int, int]] = {}
        # Cameras with a rebuild request outstanding, keyed by the
        # (generation, epoch) the gap was reported against. A later report on
        # the same identity is a duplicate; one on a newer identity is proof
        # the rebuild landed and something else went wrong, so it is not.
        self._gaps_active: dict[str, tuple[int, int]] = {}
        # Components behind each configuration signature, kept so a signature
        # change can name the field that actually moved instead of reporting an
        # opaque digest mismatch.
        self._configuration_components: dict[
            tuple[str, int, int], tuple[object, ...]
        ] = {}
        self._retired_generations: dict[str, int] = {}
        self._progress = NativeAuProgress()

    def start(self) -> None:
        self._drain_thread = threading.Thread(
            target=self._drain, name="deepstream-au-drain", daemon=True
        )
        self._process_thread = threading.Thread(
            target=self._process, name="deepstream-au-process", daemon=True
        )
        self._process_thread.start()
        self._drain_thread.start()

    def close(self) -> None:
        self._stop.set()
        with suppress(OSError):
            self._endpoint.shutdown(socket.SHUT_RDWR)
        self._endpoint.close()
        for thread in (self._drain_thread, self._process_thread):
            if thread is not None:
                thread.join(timeout=2.0)

    def retire_camera(self, camera_id: str) -> None:
        active = self._epochs.pop(camera_id, None)
        if active is not None:
            self._retired_generations[camera_id] = active.source_generation
        self._gaps_active.pop(camera_id, None)
        self._retire_keys(camera_id)
        remove = getattr(self._sink, "remove_camera", None)
        if callable(remove):
            remove(camera_id)

    def accepted_count(self, camera_id: str) -> int:
        return self._progress.count(camera_id)

    def wait_for_packets(self, camera_id: str, target: int, timeout: float) -> bool:
        return self._progress.wait(camera_id, target, timeout)

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                work: _Work = receive_envelope(self._endpoint)
            except AuFrameError as error:
                if error.camera_id:
                    work = _Gap(error.camera_id)
                else:
                    self._stream_death_handler("au_framing_unknown_camera")
                    return
            except (ConnectionError, OSError):
                if not self._stop.is_set():
                    self._stream_death_handler("au_stream_closed")
                return
            try:
                self._work.put_nowait(work)
            except queue.Full:
                camera = work.camera_id
                with suppress(queue.Empty):
                    _ = self._work.get_nowait()
                with suppress(queue.Full):
                    self._work.put_nowait(_Gap(camera))

    def _process(self) -> None:
        while not self._stop.is_set():
            try:
                work = self._work.get(timeout=0.2)
            except queue.Empty:
                continue
            if isinstance(work, _Gap):
                self._report_gap(work.camera_id, "transport_gap_marker")
                continue
            try:
                self._accept(work)
            except Exception as error:  # noqa: BLE001 - malformed camera data is isolated
                # The bare catch used to discard the exception entirely, so the
                # dominant cause of source rebuilds was invisible: 293 of 308
                # gaps in a five-minute window landed here with nothing to say
                # what had been raised.
                self._report_gap(
                    work.camera_id,
                    f"malformed_envelope:{type(error).__name__}:{error}"[:200],
                )

    def _signature_delta(self, key: tuple[str, int, int], envelope: AuEnvelope) -> str:
        """Name which signature inputs changed, so the cause is not a guess.

        The signature is a digest over seven inputs, so a mismatch alone says
        nothing about which one moved. Cameras retransmit parameter sets
        periodically, which is expected and should not look like a geometry or
        codec change.
        """
        previous = self._configuration_components.get(key)
        current = _components_of(envelope)
        if previous is None:
            return "no_previous_components"
        moved = [
            f"{name}:{old}->{new}"
            for name, old, new in zip(_COMPONENT_NAMES, previous, current, strict=True)
            if old != new
        ]
        return ",".join(moved) if moved else "digest_only"

    def _report_gap(
        self,
        camera_id: str,
        reason: str,
        identity: tuple[int, int] | None = None,
    ) -> None:
        """Ask for a source rebuild, naming the condition that prompted it.

        The wire category stays ``"parser"`` because the child validates it
        against a fixed whitelist, but that label describes none of these
        conditions and sent one investigation into the C++ AU parser, which
        turned out to be emitting nothing at all. The real reason goes to the
        log so the next reader is not misdirected the same way.

        ``identity`` is the (generation, epoch) the condition was observed on;
        transport gaps carry none and fall back to the adopted identity. A
        report is suppressed only while one is outstanding for the same or a
        newer identity. Suppressing by camera alone left rings stranded on the
        live fleet: the rebuild landed, its units were refused for a reason of
        their own, and the refusal could never be reported because the camera
        was still marked pending from the gap that caused the rebuild.
        """
        if not camera_id:
            return
        if identity is None:
            active = self._epochs.get(camera_id)
            identity = (
                (active.source_generation, active.stream_epoch)
                if active is not None
                else (0, 0)
            )
        pending = self._gaps_active.get(camera_id)
        if pending is not None and pending >= identity:
            return
        self._gaps_active[camera_id] = identity
        LOGGER.warning(
            "native au gap: camera_id=%s reason=%s", camera_id, reason
        )
        self._gap_handler(camera_id, "parser")

    def _accept(self, envelope: AuEnvelope) -> None:
        retired = self._retired_generations.get(envelope.camera_id, -1)
        if envelope.generation <= retired:
            return
        self._retired_generations.pop(envelope.camera_id, None)
        active = self._epochs.get(envelope.camera_id)
        if active is not None and (
            envelope.generation < active.source_generation
            or (
                envelope.generation == active.source_generation
                and envelope.epoch < active.stream_epoch
            )
        ):
            # Access units already in flight when the epoch rolled arrive
            # carrying the superseded identity. That is expected debris, not a
            # gap: reporting it asks for another rebuild, which advances the
            # epoch again and strands the next batch of in-flight units, so the
            # loop feeds itself. Measured on the live fleet this produced 301
            # rebuilds in five minutes with zero child-reported failures.
            # Retired generations are already dropped silently a few lines up;
            # a superseded epoch gets the same treatment.
            return
        identity = StreamEpoch(
            self._worker_boot_id, envelope.camera_id, envelope.epoch, envelope.generation
        )
        key = (envelope.camera_id, envelope.generation, envelope.epoch)
        observed = (envelope.generation, envelope.epoch)
        if envelope.kind is AuKind.GAP:
            self._report_gap(envelope.camera_id, "gap_marker", observed)
            return
        expected = self._sequences.get(key, 0) + 1
        if envelope.sequence != expected:
            if key in self._sequences:
                self._report_gap(envelope.camera_id, "sequence_discontinuity", observed)
                return
            # A unit for an identity newer than anything adopted, arriving
            # after sequence 1. The identity itself proves the rebuild landed;
            # only the opening unit was lost, and the sender-side reservation
            # cannot protect it once it is inside this process (the drain
            # queue overflows under a fleet-wide rebuild storm). Refusing it
            # and asking for another rebuild fed that storm and, with the
            # pending-gap suppression, then stranded ten of thirteen rings for
            # half an hour. Adopt it: the ring starts at the next keyframe
            # regardless of which unit opened the epoch.
            LOGGER.warning(
                "native au epoch adopted mid-stream: camera_id=%s generation=%d "
                "epoch=%d first_sequence=%d",
                envelope.camera_id, envelope.generation, envelope.epoch, envelope.sequence,
            )
        if self._timestamp_discontinuous(key, envelope):
            self._report_gap(envelope.camera_id, "timestamp_discontinuity", observed)
            return
        if active != identity:
            if active is not None and envelope.generation > active.source_generation:
                remove = getattr(self._sink, "remove_camera", None)
                if callable(remove):
                    remove(envelope.camera_id)
            self._sink.register_camera(envelope.camera_id)
            self._sink.roll_epoch(identity)
            self._retire_keys(envelope.camera_id, keep=key)
            self._epochs[envelope.camera_id] = identity
        self._sequences[key] = envelope.sequence
        signature = native_configuration_signature(
            envelope.codec, envelope.framing, envelope.parser_caps, envelope.codec_data,
            envelope.width, envelope.height, envelope.time_base,
        )
        configuration = self._configurations.get(key)
        if configuration is None:
            configuration = stream_configuration(envelope)
            self._configurations[key] = configuration
            self._configuration_signatures[key] = signature
            self._configuration_components[key] = _components_of(envelope)
        elif self._configuration_signatures[key] != signature:
            self._report_gap(
                envelope.camera_id,
                f"configuration_signature_changed:{self._signature_delta(key, envelope)}",
                observed,
            )
            return
        packet = SourcePacket(
            identity, configuration, 0, envelope.pts, envelope.dts, envelope.duration,
            envelope.keyframe, envelope.payload, envelope.sequence - 1,
        )
        if not self._sink.append(packet):
            self._report_gap(envelope.camera_id, "ring_append_rejected", observed)
            return
        self._timeline[key] = (envelope.dts, envelope.duration)
        self._gaps_active.pop(envelope.camera_id, None)
        self._progress.accept(envelope.camera_id)
        if self._accept_handler is not None:
            self._accept_handler(
                envelope.camera_id,
                envelope.pts,
                envelope.sequence,
                envelope.generation,
            )

    def _timestamp_discontinuous(self, key: tuple[str, int, int], envelope: AuEnvelope) -> bool:
        previous = self._timeline.get(key)
        if previous is None:
            return False
        previous_dts, previous_duration = previous
        delta = envelope.dts - previous_dts
        maximum = max(previous_duration * 120, envelope.time_base.denominator * 5)
        return delta <= 0 or delta > maximum

    def _retire_keys(self, camera_id: str, keep: tuple[str, int, int] | None = None) -> None:
        for mapping in (
            self._sequences, self._configurations, self._configuration_signatures, self._timeline,
        ):
            for key in tuple(mapping):
                if key[0] == camera_id and key != keep:
                    del mapping[key]


__all__ = [
    "NativeAuGapHandler", "NativeAuReceiver", "NativeAuSink", "NativeAuStreamDeathHandler",
]
