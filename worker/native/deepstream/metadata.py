"""Capacity-one ingestion for nonblocking native metadata wakeups."""

from __future__ import annotations

import os
import socket
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import Final, Literal, Protocol, TypeAlias, final

from worker.native.deepstream.ipc import MetadataCounters, MetadataFrame

_MAX_DATAGRAM: Final = 512
CounterName: TypeAlias = Literal[
    "accepted",
    "overwritten",
    "late",
    "unknown_source",
    "generation_mismatch",
    "epoch_mismatch",
    "boot_mismatch",
    "child_mismatch",
    "transform_mismatch",
    "malformed",
]


@dataclass(frozen=True, slots=True)
class SourceBinding:
    worker_boot_id: str
    child_instance_id: str
    camera_id: str
    source_generation: int
    stream_epoch: int
    transform_id: str


class MetadataPullStopped(Exception):
    """Control session ended while servicing a metadata wakeup."""


class MetadataPuller(Protocol):
    def pull_latest(self, camera_id: str) -> MetadataFrame | None: ...


def _increment(counters: MetadataCounters, name: CounterName) -> MetadataCounters:
    values = {
        "accepted": lambda: replace(counters, accepted=counters.accepted + 1),
        "overwritten": lambda: replace(counters, overwritten=counters.overwritten + 1),
        "late": lambda: replace(counters, late=counters.late + 1),
        "unknown_source": lambda: replace(counters, unknown_source=counters.unknown_source + 1),
        "generation_mismatch": lambda: replace(
            counters,
            generation_mismatch=counters.generation_mismatch + 1,
        ),
        "epoch_mismatch": lambda: replace(counters, epoch_mismatch=counters.epoch_mismatch + 1),
        "boot_mismatch": lambda: replace(counters, boot_mismatch=counters.boot_mismatch + 1),
        "child_mismatch": lambda: replace(counters, child_mismatch=counters.child_mismatch + 1),
        "transform_mismatch": lambda: replace(
            counters,
            transform_mismatch=counters.transform_mismatch + 1,
        ),
        "malformed": lambda: replace(counters, malformed=counters.malformed + 1),
    }
    return values[name]()


@final
class LatestMetadataSlot:
    """Capacity-one validated metadata per source; mutation is intentional."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._expected: dict[str, SourceBinding] = {}
        self._latest: dict[str, MetadataFrame] = {}
        self._high_water: dict[str, tuple[int, int, int]] = {}
        self._counters = MetadataCounters()

    def register_source(self, binding: SourceBinding) -> None:
        with self._lock:
            self._expected[binding.camera_id] = binding
            _ = self._latest.pop(binding.camera_id, None)
            _ = self._high_water.pop(binding.camera_id, None)

    def remove_source(self, camera_id: str) -> None:
        with self._lock:
            _ = self._expected.pop(camera_id, None)
            _ = self._latest.pop(camera_id, None)
            _ = self._high_water.pop(camera_id, None)

    def publish(self, metadata: MetadataFrame) -> bool:
        camera_id = metadata.identity.camera_id
        with self._lock:
            expected = self._expected.get(camera_id)
            counters = self._counters
            if expected is None:
                self._counters = _increment(counters, "unknown_source")
                return False
            if metadata.identity.worker_boot_id != expected.worker_boot_id:
                self._counters = _increment(counters, "boot_mismatch")
                return False
            if str(metadata.child_instance_id) != expected.child_instance_id:
                self._counters = _increment(counters, "child_mismatch")
                return False
            if metadata.source_generation != expected.source_generation:
                self._counters = _increment(counters, "generation_mismatch")
                return False
            if metadata.identity.stream_epoch != expected.stream_epoch:
                self._counters = _increment(counters, "epoch_mismatch")
                return False
            if metadata.transform_id != expected.transform_id:
                self._counters = _increment(counters, "transform_mismatch")
                return False
            identity = (
                metadata.identity.source_pts or 0,
                metadata.identity.seq,
                metadata.native_publish_sequence,
            )
            high_water = self._high_water.get(camera_id)
            if high_water is not None and any(
                current <= previous for current, previous in zip(identity, high_water, strict=True)
            ):
                self._counters = _increment(counters, "late")
                return False
            if camera_id in self._latest:
                counters = _increment(counters, "overwritten")
            self._latest[camera_id] = metadata
            self._high_water[camera_id] = identity
            self._counters = _increment(counters, "accepted")
            return True

    def peek(self, camera_id: str) -> MetadataFrame | None:
        with self._lock:
            return self._latest.get(camera_id)

    def take(self, camera_id: str) -> MetadataFrame | None:
        with self._lock:
            return self._latest.pop(camera_id, None)

    def mark_malformed(self) -> None:
        with self._lock:
            self._counters = _increment(self._counters, "malformed")

    def counters(self) -> MetadataCounters:
        with self._lock:
            return self._counters


@final
class MetadataReceiver:
    """Pull the native capacity-one slot after each nonblocking wakeup."""

    def __init__(self, path: Path, slot: LatestMetadataSlot, puller: MetadataPuller) -> None:
        self._path = path
        self._slot = slot
        self._puller = puller
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._received_count = 0
        self._enabled = threading.Event()
        self._enabled.set()

    def __enter__(self) -> MetadataReceiver:
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._path.parent, 0o700)
        self._path.unlink(missing_ok=True)
        receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        receiver.bind(str(self._path))
        os.chmod(self._path, 0o600)
        self._socket = receiver
        self._thread = threading.Thread(target=self._run, name="deepstream-metadata", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        receiver = self._socket
        if receiver is None:
            return
        while True:
            data = receiver.recv(_MAX_DATAGRAM)
            if data == b"":
                return
            _ = self._enabled.wait()
            try:
                camera_id = data.decode()
            except UnicodeDecodeError:
                self._slot.mark_malformed()
            else:
                try:
                    metadata = self._puller.pull_latest(camera_id)
                except MetadataPullStopped:
                    return
                if metadata is not None:
                    _ = self._slot.publish(metadata)
            with self._condition:
                self._received_count += 1
                self._condition.notify_all()

    def pause(self) -> None:
        self._enabled.clear()

    def resume(self) -> None:
        self._enabled.set()

    def subscription(self) -> int:
        with self._condition:
            return self._received_count

    def wait_received(self, after: int, *, timeout_sec: float) -> None:
        with self._condition:
            received = self._condition.wait_for(
                lambda: self._received_count > after,
                timeout=timeout_sec,
            )
        if not received:
            raise TimeoutError("metadata receive deadline elapsed")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self._enabled.set()
        sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            _ = sender.sendto(b"", str(self._path))
        finally:
            sender.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._socket is not None:
            self._socket.close()
        self._path.unlink(missing_ok=True)


__all__ = [
    "LatestMetadataSlot",
    "MetadataPuller",
    "MetadataPullStopped",
    "MetadataReceiver",
    "SourceBinding",
]
