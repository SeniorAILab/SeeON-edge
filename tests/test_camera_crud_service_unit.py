from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from backend.app.features.audit.catalog import AuditAction
from backend.app.features.cameras.camera_crud_service import (
    AfterWrite,
    CameraCreateInputs,
    CameraCrudPorts,
    CameraNotFoundError,
    CameraPatchInputs,
    CameraRecord,
    CameraWriteResult,
    create_camera,
    delete_camera,
    update_camera,
)
from backend.app.features.cameras.camera_values import DuplicateCameraError, ProbeResult
from backend.app.features.cameras.update_command import CameraUpdate

NOW = "2026-10-07T00:00:00.000Z"


class InvalidField(ValueError):
    pass


class FakeStore:
    def __init__(self, records: dict[str, CameraRecord] | None = None) -> None:
        self.records = dict(records or {})
        self.created: list[dict[str, object]] = []
        self.updates: list[tuple[str, dict[str, object]]] = []
        self.deleted: list[str] = []
        self.update_result_missing = False
        self.delete_result: bool | None = None
        self.create_error: Exception | None = None

    def get(self, camera_id: str) -> CameraRecord | None:
        record = self.records.get(camera_id)
        return None if record is None else dict(record)

    def create(self, **kwargs: Any) -> CameraRecord:
        after_write = kwargs.pop("after_write")
        if self.create_error is not None:
            raise self.create_error
        after_write("connection")
        self.created.append(kwargs)
        record: CameraRecord = {"id": kwargs["camera_id"], **kwargs}
        self.records[str(kwargs["camera_id"])] = record
        return dict(record)

    def update(
        self, camera_id: str, updates: CameraUpdate, *, after_write: AfterWrite | None = None
    ) -> CameraRecord | None:
        fields = updates.model_dump(exclude_unset=True)
        self.updates.append((camera_id, fields))
        if self.update_result_missing:
            return None
        assert after_write is not None
        after_write("connection")
        record = {**self.records[camera_id], **fields}
        self.records[camera_id] = record
        return dict(record)

    def delete(self, camera_id: str, *, after_write: AfterWrite | None = None) -> bool:
        self.deleted.append(camera_id)
        if self.delete_result is not None:
            return self.delete_result
        assert after_write is not None
        after_write("connection")
        return self.records.pop(camera_id, None) is not None


class Recorder:
    def __init__(self, store: FakeStore, probes: list[ProbeResult] | None = None) -> None:
        self.store = store
        self.probes = list(probes or [])
        self.calls: list[tuple[str, object]] = []
        self.audits: list[tuple[AuditAction, str, bool]] = []
        self.ids = iter(["new-camera-1", "new-camera-2"])

    def audited_write(
        self,
        action: AuditAction,
        target_id: str,
        write: Callable[[AfterWrite], Any],
        expects_audit: Callable[[Any], bool],
    ) -> Any:
        appended: list[object] = []
        result = write(appended.append)
        assert expects_audit(result) is bool(appended)
        self.audits.append((action, target_id, expects_audit(result)))
        return result

    def validate(self, rtsp_url: str) -> str:
        self.calls.append(("validate", rtsp_url))
        if "bad" in rtsp_url:
            raise InvalidField("bad url")
        return rtsp_url.strip()

    def decode_backend(self, value: str | None) -> str | None:
        self.calls.append(("decode_backend", value))
        if value == "gpu":
            raise InvalidField("bad decode backend")
        return None if value is None else value.lower()

    def floor(self, value: int | None) -> int | None:
        self.calls.append(("floor", value))
        if value is not None and value > 10:
            raise InvalidField("bad floor")
        return value

    def probe(self, rtsp_url: str) -> ProbeResult:
        self.calls.append(("probe", rtsp_url))
        return self.probes.pop(0) if self.probes else ProbeResult(True, width=1, height=1)

    def ports(self) -> CameraCrudPorts:
        return CameraCrudPorts(
            store=lambda: self.store,
            audited_write=self.audited_write,
            validate_rtsp_url=self.validate,
            normalize_decode_backend=self.decode_backend,
            normalize_floor=self.floor,
            probe=self.probe,
            new_camera_id=lambda: next(self.ids),
            now=lambda: NOW,
        )


def _seeded(**fields: object) -> FakeStore:
    record: CameraRecord = {"id": "cam-1", "label": "Old", "rtsp_url": "rtsp://old", **fields}
    return FakeStore({"cam-1": record})


def test_create_validates_then_probes_then_writes_one_audited_online_record() -> None:
    store = FakeStore()
    recorder = Recorder(store)

    result = create_camera(
        CameraCreateInputs(
            label="Bed",
            rtsp_url=" rtsp://cam ",
            space_id="space-1",
            decode_backend="NVDEC",
            floor=2,
            edge_ref="cam-ref",
            room_edge_ref="room-ref",
        ),
        recorder.ports(),
    )

    assert recorder.calls == [
        ("validate", " rtsp://cam "),
        ("decode_backend", "NVDEC"),
        ("floor", 2),
        ("probe", "rtsp://cam"),
    ]
    assert store.created == [
        {
            "camera_id": "new-camera-1",
            "label": "Bed",
            "rtsp_url": "rtsp://cam",
            "space_id": "space-1",
            "status": "online",
            "backend_camera_id": None,
            "mapping_pending": False,
            "decode_backend": "nvdec",
            "floor": 2,
            "last_probed_at": NOW,
            "last_ok_at": NOW,
            "never_connected": False,
            "edge_ref": "cam-ref",
            "room_edge_ref": "room-ref",
        }
    ]
    assert recorder.audits == [(AuditAction.CAMERA_CREATE, "new-camera-1", True)]
    assert result.changed is True
    assert result.camera["id"] == "new-camera-1"


def test_create_with_failed_probe_registers_offline_never_connected() -> None:
    store = FakeStore()
    recorder = Recorder(store, [ProbeResult(False, "auth")])

    create_camera(CameraCreateInputs(label="A", rtsp_url="rtsp://cam"), recorder.ports())

    created = store.created[0]
    assert (created["status"], created["never_connected"]) == ("offline", True)
    assert (created["last_probed_at"], created["last_ok_at"]) == (NOW, None)


@pytest.mark.parametrize(
    ("inputs", "calls"),
    [
        (
            CameraCreateInputs(label="A", rtsp_url="rtsp://bad"),
            [("validate", "rtsp://bad")],
        ),
        (
            CameraCreateInputs(label="A", rtsp_url="rtsp://cam", decode_backend="gpu"),
            [("validate", "rtsp://cam"), ("decode_backend", "gpu")],
        ),
        (
            CameraCreateInputs(label="A", rtsp_url="rtsp://cam", floor=11),
            [("validate", "rtsp://cam"), ("decode_backend", None), ("floor", 11)],
        ),
    ],
    ids=["url", "decode-backend", "floor"],
)
def test_create_invalid_input_stops_before_probe_and_write(
    inputs: CameraCreateInputs, calls: list[tuple[str, object]]
) -> None:
    store = FakeStore()
    recorder = Recorder(store)

    with pytest.raises(InvalidField):
        create_camera(inputs, recorder.ports())

    assert recorder.calls == calls
    assert store.created == []
    assert recorder.audits == []


def test_create_propagates_store_conflicts_unchanged() -> None:
    store = FakeStore()
    conflict = DuplicateCameraError({"id": "cam-0", "label": "Existing"})
    store.create_error = conflict
    recorder = Recorder(store)

    with pytest.raises(DuplicateCameraError) as raised:
        create_camera(CameraCreateInputs(label="A", rtsp_url="rtsp://cam"), recorder.ports())

    assert raised.value is conflict
    assert ("probe", "rtsp://cam") in recorder.calls


def test_update_missing_camera_raises_not_found_before_any_port_call() -> None:
    recorder = Recorder(FakeStore())

    with pytest.raises(CameraNotFoundError) as raised:
        update_camera(
            CameraPatchInputs(
                camera_id="ghost", fields_set=frozenset({"rtsp_url"}), rtsp_url="rtsp://bad"
            ),
            recorder.ports(),
        )

    assert raised.value.camera_id == "ghost"
    assert recorder.calls == []
    assert recorder.audits == []


def test_update_with_no_fields_returns_current_record_unchanged() -> None:
    store = _seeded(status="offline")
    recorder = Recorder(store)

    result = update_camera(
        CameraPatchInputs(camera_id="cam-1", fields_set=frozenset()), recorder.ports()
    )

    assert result == CameraWriteResult(camera=store.records["cam-1"], changed=False)
    assert store.updates == []
    assert recorder.audits == []


def test_update_label_only_writes_label_without_probe() -> None:
    store = _seeded()
    recorder = Recorder(store)

    result = update_camera(
        CameraPatchInputs(camera_id="cam-1", fields_set=frozenset({"label"}), label="New"),
        recorder.ports(),
    )

    assert store.updates == [("cam-1", {"label": "New"})]
    assert recorder.calls == []
    assert recorder.audits == [(AuditAction.CAMERA_UPDATE, "cam-1", True)]
    assert result.changed is True
    assert result.camera["label"] == "New"


def test_update_rtsp_url_success_records_probe_outcome() -> None:
    store = _seeded()
    recorder = Recorder(store)

    update_camera(
        CameraPatchInputs(
            camera_id="cam-1", fields_set=frozenset({"rtsp_url"}), rtsp_url=" rtsp://new "
        ),
        recorder.ports(),
    )

    assert recorder.calls == [("validate", " rtsp://new "), ("probe", "rtsp://new")]
    assert store.updates == [
        (
            "cam-1",
            {
                "rtsp_url": "rtsp://new",
                "status": "online",
                "last_probed_at": NOW,
                "last_ok_at": NOW,
                "never_connected": False,
            },
        )
    ]


def test_update_rtsp_url_failure_keeps_connection_history_fields_untouched() -> None:
    store = _seeded()
    recorder = Recorder(store, [ProbeResult(False, probe_unavailable=True)])

    update_camera(
        CameraPatchInputs(
            camera_id="cam-1", fields_set=frozenset({"rtsp_url"}), rtsp_url="rtsp://new"
        ),
        recorder.ports(),
    )

    assert store.updates == [
        ("cam-1", {"rtsp_url": "rtsp://new", "status": "offline", "last_probed_at": NOW})
    ]


def test_update_normalizes_optionals_and_passes_explicit_nulls_through() -> None:
    store = _seeded()
    recorder = Recorder(store)

    update_camera(
        CameraPatchInputs(
            camera_id="cam-1",
            fields_set=frozenset(
                {"label", "rtsp_url", "space_id", "decode_backend", "floor", "edge_ref"}
            ),
            label=None,
            rtsp_url=None,
            space_id=None,
            decode_backend="CPU",
            floor=None,
            edge_ref="cam-ref",
        ),
        recorder.ports(),
    )

    assert recorder.calls == [("decode_backend", "CPU"), ("floor", None)]
    assert store.updates == [
        (
            "cam-1",
            {"space_id": None, "decode_backend": "cpu", "floor": None, "edge_ref": "cam-ref"},
        )
    ]


def test_update_with_only_null_fields_still_writes_an_empty_update() -> None:
    store = _seeded()
    recorder = Recorder(store)

    result = update_camera(
        CameraPatchInputs(camera_id="cam-1", fields_set=frozenset({"label"}), label=None),
        recorder.ports(),
    )

    assert store.updates == [("cam-1", {})]
    assert result.changed is True


def test_update_invalid_floor_after_probe_raises_without_write() -> None:
    store = _seeded()
    recorder = Recorder(store)

    with pytest.raises(InvalidField):
        update_camera(
            CameraPatchInputs(
                camera_id="cam-1",
                fields_set=frozenset({"rtsp_url", "floor"}),
                rtsp_url="rtsp://new",
                floor=11,
            ),
            recorder.ports(),
        )

    assert recorder.calls == [("validate", "rtsp://new"), ("probe", "rtsp://new"), ("floor", 11)]
    assert store.updates == []


def test_update_vanishing_camera_raises_not_found_after_unaudited_write() -> None:
    store = _seeded()
    store.update_result_missing = True
    recorder = Recorder(store)

    with pytest.raises(CameraNotFoundError):
        update_camera(
            CameraPatchInputs(camera_id="cam-1", fields_set=frozenset({"label"}), label="X"),
            recorder.ports(),
        )

    assert recorder.audits == [(AuditAction.CAMERA_UPDATE, "cam-1", False)]


def test_delete_existing_camera_is_audited() -> None:
    store = _seeded()
    recorder = Recorder(store)

    delete_camera("cam-1", recorder.ports())

    assert store.deleted == ["cam-1"]
    assert "cam-1" not in store.records
    assert recorder.audits == [(AuditAction.CAMERA_DELETE, "cam-1", True)]


def test_delete_missing_camera_raises_without_write() -> None:
    store = FakeStore()
    recorder = Recorder(store)

    with pytest.raises(CameraNotFoundError):
        delete_camera("ghost", recorder.ports())

    assert store.deleted == []
    assert recorder.audits == []


def test_delete_lost_race_raises_not_found_after_unaudited_write() -> None:
    store = _seeded()
    store.delete_result = False
    recorder = Recorder(store)

    with pytest.raises(CameraNotFoundError):
        delete_camera("cam-1", recorder.ports())

    assert recorder.audits == [(AuditAction.CAMERA_DELETE, "cam-1", False)]


def test_inputs_and_results_are_frozen() -> None:
    inputs = CameraCreateInputs(label="A", rtsp_url="rtsp://cam")
    patch = CameraPatchInputs(camera_id="cam-1", fields_set=frozenset())
    result = CameraWriteResult(camera={}, changed=False)

    for value, name in ((inputs, "label"), (patch, "camera_id"), (result, "changed")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, name, "changed")
