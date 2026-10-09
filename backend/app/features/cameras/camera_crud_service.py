from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from backend.app.edge_db import DatabaseConnection
from backend.app.features.audit.catalog import AuditAction
from backend.app.features.cameras.camera_values import CameraStatus, ProbeResult, status_from_probe
from backend.app.features.cameras.update_command import CameraUpdate

AfterWrite = Callable[[DatabaseConnection[Any]], None]
CameraRecord = dict[str, object]
_Result = TypeVar("_Result")


class CameraStore(Protocol):
    def get(self, camera_id: str) -> CameraRecord | None: ...

    def create(
        self,
        *,
        camera_id: str | None = ...,
        label: str,
        rtsp_url: str,
        space_id: str | None,
        status: CameraStatus,
        backend_camera_id: str | None = ...,
        mapping_pending: bool = ...,
        decode_backend: str | None = ...,
        floor: int | None = ...,
        last_probed_at: str | None = ...,
        last_ok_at: str | None = ...,
        never_connected: bool = ...,
        edge_ref: str | None = ...,
        room_edge_ref: str | None = ...,
        after_write: AfterWrite | None = ...,
    ) -> CameraRecord: ...

    def update(
        self, camera_id: str, updates: CameraUpdate, *, after_write: AfterWrite | None = ...
    ) -> CameraRecord | None: ...

    def delete(self, camera_id: str, *, after_write: AfterWrite | None = ...) -> bool: ...


class AuditedWrite(Protocol):
    def __call__(
        self,
        action: AuditAction,
        target_id: str,
        write: Callable[[AfterWrite], _Result],
        expects_audit: Callable[[_Result], bool],
    ) -> _Result: ...


@dataclass(frozen=True, slots=True)
class CameraCrudPorts:
    store: Callable[[], CameraStore]
    audited_write: AuditedWrite
    validate_rtsp_url: Callable[[str], str]
    normalize_decode_backend: Callable[[str | None], str | None]
    normalize_floor: Callable[[int | None], int | None]
    probe: Callable[[str], ProbeResult]
    new_camera_id: Callable[[], str]
    now: Callable[[], str]


@dataclass(frozen=True, slots=True)
class CameraCreateInputs:
    label: str
    rtsp_url: str
    space_id: str | None = None
    decode_backend: str | None = None
    floor: int | None = None
    edge_ref: str | None = None
    room_edge_ref: str | None = None


@dataclass(frozen=True, slots=True)
class CameraPatchInputs:
    camera_id: str
    fields_set: frozenset[str]
    label: str | None = None
    rtsp_url: str | None = None
    space_id: str | None = None
    decode_backend: str | None = None
    floor: int | None = None
    edge_ref: str | None = None
    room_edge_ref: str | None = None


@dataclass(frozen=True, slots=True)
class CameraWriteResult:
    camera: CameraRecord
    changed: bool


class CameraNotFoundError(LookupError):
    def __init__(self, camera_id: str) -> None:
        super().__init__(camera_id)
        self.camera_id = camera_id


def _always(_result: object) -> bool:
    return True


def _found(result: CameraRecord | None) -> bool:
    return result is not None


def create_camera(inputs: CameraCreateInputs, ports: CameraCrudPorts) -> CameraWriteResult:
    rtsp_url = ports.validate_rtsp_url(inputs.rtsp_url)
    decode_backend = ports.normalize_decode_backend(inputs.decode_backend)
    floor = ports.normalize_floor(inputs.floor)
    probe = ports.probe(rtsp_url)
    camera_id = ports.new_camera_id()
    now = ports.now()
    store = ports.store()

    def write(after_write: AfterWrite) -> CameraRecord:
        return store.create(
            camera_id=camera_id,
            label=inputs.label,
            rtsp_url=rtsp_url,
            space_id=inputs.space_id,
            status=status_from_probe(probe),
            backend_camera_id=None,
            mapping_pending=False,
            decode_backend=decode_backend,
            floor=floor,
            last_probed_at=now,
            last_ok_at=now if probe.ok else None,
            never_connected=not probe.ok,
            edge_ref=inputs.edge_ref,
            room_edge_ref=inputs.room_edge_ref,
            after_write=after_write,
        )

    record = ports.audited_write(AuditAction.CAMERA_CREATE, camera_id, write, _always)
    return CameraWriteResult(camera=record, changed=True)


def _patch_updates(inputs: CameraPatchInputs, ports: CameraCrudPorts) -> dict[str, object]:
    fields = inputs.fields_set
    updates: dict[str, object] = {}
    if "label" in fields and inputs.label is not None:
        updates["label"] = inputs.label
    if "rtsp_url" in fields and inputs.rtsp_url is not None:
        rtsp_url = ports.validate_rtsp_url(inputs.rtsp_url)
        probe = ports.probe(rtsp_url)
        updates["rtsp_url"] = rtsp_url
        updates["status"] = status_from_probe(probe)
        now = ports.now()
        updates["last_probed_at"] = now
        if probe.ok:
            updates["last_ok_at"] = now
            updates["never_connected"] = False
    if "space_id" in fields:
        updates["space_id"] = inputs.space_id
    if "decode_backend" in fields:
        updates["decode_backend"] = ports.normalize_decode_backend(inputs.decode_backend)
    if "floor" in fields:
        updates["floor"] = ports.normalize_floor(inputs.floor)
    if "edge_ref" in fields:
        updates["edge_ref"] = inputs.edge_ref
    if "room_edge_ref" in fields:
        updates["room_edge_ref"] = inputs.room_edge_ref
    return updates


def update_camera(inputs: CameraPatchInputs, ports: CameraCrudPorts) -> CameraWriteResult:
    store = ports.store()
    current = store.get(inputs.camera_id)
    if current is None:
        raise CameraNotFoundError(inputs.camera_id)
    if not inputs.fields_set:
        return CameraWriteResult(camera=current, changed=False)
    updates = _patch_updates(inputs, ports)

    def write(after_write: AfterWrite) -> CameraRecord | None:
        return store.update(
            inputs.camera_id, CameraUpdate.model_validate(updates), after_write=after_write
        )

    updated = ports.audited_write(AuditAction.CAMERA_UPDATE, inputs.camera_id, write, _found)
    if updated is None:
        raise CameraNotFoundError(inputs.camera_id)
    return CameraWriteResult(camera=updated, changed=True)


def delete_camera(camera_id: str, ports: CameraCrudPorts) -> None:
    store = ports.store()
    existing = store.get(camera_id)

    def write(after_write: AfterWrite) -> bool:
        return store.delete(camera_id, after_write=after_write)

    if existing is None or not ports.audited_write(
        AuditAction.CAMERA_DELETE, camera_id, write, bool
    ):
        raise CameraNotFoundError(camera_id)


__all__ = [
    "AfterWrite",
    "AuditedWrite",
    "CameraCreateInputs",
    "CameraCrudPorts",
    "CameraNotFoundError",
    "CameraPatchInputs",
    "CameraRecord",
    "CameraStore",
    "CameraWriteResult",
    "create_camera",
    "delete_camera",
    "update_camera",
]
