from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Final

from shared.events.execution_records import ExecutionRecordContractError, WireRecord
from worker.domains.fall.classifier import FALL_WINDOW_FRAMES
from worker.interfaces.execution_records import ExecutionRecordSink

LOGGER = logging.getLogger(__name__)


PRODUCER_SDK = "sdk"
PRODUCER_MODEL = "model"
PRODUCER_POLICY = "policy"
PRODUCER_EVENT = "event"
PRODUCER_BACKEND = "backend"


def try_emit(sink: ExecutionRecordSink | None, record: WireRecord | None) -> bool:
    if sink is None or record is None:
        return False
    try:
        return bool(sink.try_emit(record))
    except Exception:  # noqa: BLE001
        LOGGER.warning(
            "execution-record sink.try_emit failed camera_id=%s record_kind=%s",
            record.camera_id,
            record.record_kind,
        )
        return False


def frame_causal_unit_id(camera_id: str, worker_boot_id: str, stream_epoch: int, seq: int) -> str:
    return f"{camera_id}:{worker_boot_id}:{stream_epoch}:frame:{seq // FALL_WINDOW_FRAMES}"


NO_TRACK: Final = "no-track"
NO_GENERATION: Final = "no-generation"


def fall_causal_unit_id(
    camera_id: str,
    worker_boot_id: str,
    stream_epoch: int,
    track_id: int | None,
    generation: int | None,
) -> str:
    track = NO_TRACK if track_id is None else str(track_id)
    gen = NO_GENERATION if generation is None else str(generation)
    return f"{camera_id}:{worker_boot_id}:{stream_epoch}:{track}:{gen}"


NO_MODULE: Final = "no-module"


def module_causal_unit_id(
    camera_id: str,
    worker_boot_id: str,
    stream_epoch: int,
    module_qualified_id: str | None,
    frame_seq: int,
) -> str:
    module = NO_MODULE if module_qualified_id is None else module_qualified_id
    return f"{camera_id}:{worker_boot_id}:{stream_epoch}:{module}:{frame_seq // FALL_WINDOW_FRAMES}"


def wall_or(observed_at_ns: int | None) -> int:
    return time.time_ns() if observed_at_ns is None else observed_at_ns


WALL: Final = "wall"


def make_record(
    *,
    record_kind: str,
    camera_id: str,
    worker_boot_id: str,
    source_generation: int,
    stream_epoch: int,
    producer: str,
    observed_at_ns: int,
    time_quality: str,
    causal_unit_id: str,
    outcome: str,
    payload: Mapping[str, object],
    frame_seq: int | None = None,
    source_pts_ns: int | None = None,
) -> WireRecord | None:
    try:
        return WireRecord(
            record_kind=record_kind,
            camera_id=camera_id,
            worker_boot_id=worker_boot_id,
            source_generation=source_generation,
            stream_epoch=stream_epoch,
            producer=producer,
            producer_sequence=0,
            observed_at_ns=observed_at_ns,
            time_quality=time_quality,
            causal_unit_id=causal_unit_id,
            outcome=outcome,
            payload=payload,
            frame_seq=frame_seq,
            source_pts_ns=source_pts_ns,
        )
    except ExecutionRecordContractError as error:
        LOGGER.warning(
            "execution-record contract rejected camera_id=%s record_kind=%s %s",
            camera_id,
            record_kind,
            error,
        )
        return None


__all__ = [
    "NO_GENERATION",
    "NO_MODULE",
    "NO_TRACK",
    "PRODUCER_BACKEND",
    "PRODUCER_EVENT",
    "PRODUCER_MODEL",
    "PRODUCER_POLICY",
    "PRODUCER_SDK",
    "WALL",
    "fall_causal_unit_id",
    "frame_causal_unit_id",
    "make_record",
    "module_causal_unit_id",
    "try_emit",
    "wall_or",
]
