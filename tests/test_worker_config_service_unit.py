from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from backend.app.features.cameras.bed_zone_store import BedZone, BedZoneRegion
from backend.app.features.cameras.worker_config_service import (
    WorkerConfigInputs,
    assemble_worker_config,
    compute_policy_camera_identities,
)
from backend.app.features.detection_settings.store import DomainDetectionSetting
from contracts.worker_config import PulledNightWindow, PulledWorkerConfig


def _snapshot(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {"registry_version": 1, "cameras": list(records)}


def test_build_camera_entries_skips_blank_rtsp_and_threads_bed_zone() -> None:
    bz = BedZone(
        regions=(BedZoneRegion(id="r1", polygon=((0, 0), (1, 0), (1, 1)), origin="manual"),),
        image_width=2,
        image_height=2,
        recognized_at="2024-01-01T00:00:00Z",
    )
    snapshot = _snapshot(
        [
            {"id": "skip-blank", "rtsp_url": "   "},
            {
                "id": "local-1",
                "backend_camera_id": "hub-1",
                "rtsp_url": "rtsp://camera/a",
                "space_id": "space-1",
                "decode_backend": "cpu",
            },
        ]
    )
    inputs = WorkerConfigInputs(
        registry_snapshot=snapshot,
        bed_zones={"local-1": bz},
        pulled=None,
        live_config_version=0,
        live_restart_epoch=0,
        detection_settings={},
        clip_store_subdir=None,
        facility_id=None,
        policy_generation=0,
        policy_bundle=None,
    )
    response = assemble_worker_config(inputs)
    assert len(response["cameras"]) == 1
    camera = response["cameras"][0]
    assert camera["camera_id"] == "hub-1"
    assert camera["space_id"] == "space-1"
    assert camera["rtsp_url"] == "rtsp://camera/a"
    assert camera["decode_backend"] == "cpu"
    assert camera["bed_zone_image_width"] == 2
    assert camera["bed_zone_image_height"] == 2
    assert isinstance(camera["bed_zone_regions"], list) and camera["bed_zone_regions"]


def test_apply_local_detection_overrides_updates_domains_and_windows() -> None:
    pulled = PulledWorkerConfig(
        config_version=3,
        restart_epoch=1,
        night_window=None,
        cameras=(),
        detection_windows={"fall": PulledNightWindow(start="08:00", end="20:00", tz="Asia/Seoul")},
    )
    stored = {
        "fall": DomainDetectionSetting(on=True, mode="window", start="09:00", end="18:00"),
        "bed_exit": DomainDetectionSetting(on=True, mode="always", start=None, end=None),
    }
    inputs = WorkerConfigInputs(
        registry_snapshot=_snapshot([]),
        bed_zones={},
        pulled=pulled,
        live_config_version=3,
        live_restart_epoch=1,
        detection_settings=stored,
        clip_store_subdir=None,
        facility_id=None,
        policy_generation=0,
        policy_bundle=None,
    )
    response = assemble_worker_config(inputs)
    assert response["domains"] == {"fall": {"enabled": True}, "bed_exit": {"enabled": True}}
    assert response["detection_windows"] == {
        "fall": {"start": "09:00", "end": "18:00", "tz": "Asia/Seoul"}
    }
    assert "night_window" not in response  # bed_exit always-on removes alias
    assert isinstance(response["config_version"], int) and response["config_version"] > 3


def test_apply_clip_storage_override_threads_selection() -> None:
    inputs = WorkerConfigInputs(
        registry_snapshot=_snapshot([]),
        bed_zones={},
        pulled=None,
        live_config_version=0,
        live_restart_epoch=0,
        detection_settings={},
        clip_store_subdir="archive",
        facility_id=None,
        policy_generation=0,
        policy_bundle=None,
    )
    response = assemble_worker_config(inputs)
    assert response["clip_store_subdir"] == "archive"


class _FakeBundle:
    def __init__(self, *, digest_prefix: str) -> None:
        self.content_sha256 = digest_prefix + ("0" * 56)

    def as_dict(self) -> dict[str, object]:
        return {"module_id": "fall", "schema_id": "fall.policy", "values": {"t": 0.7}}


def test_numeric_detection_policies_generation_zero_is_noop() -> None:
    pulled = PulledWorkerConfig(
        config_version=7, restart_epoch=1, night_window=None, cameras=(), detection_windows={}
    )
    inputs = WorkerConfigInputs(
        registry_snapshot=_snapshot([]),
        bed_zones={},
        pulled=pulled,
        live_config_version=7,
        live_restart_epoch=1,
        detection_settings={},
        clip_store_subdir=None,
        facility_id=None,
        policy_generation=0,
        policy_bundle=None,
    )
    response = assemble_worker_config(inputs)
    assert "detection_policies" not in response
    assert response["config_version"] == 7


def test_numeric_detection_policies_threads_bundle_and_scales_version() -> None:
    bundle = _FakeBundle(digest_prefix="00000042")
    pulled = PulledWorkerConfig(
        config_version=7, restart_epoch=2, night_window=None, cameras=(), detection_windows={}
    )
    inputs = WorkerConfigInputs(
        registry_snapshot=_snapshot([{"id": "local-1", "backend_camera_id": "hub-1", "rtsp_url": "rtsp://a"}]),
        bed_zones={},
        pulled=pulled,
        live_config_version=7,
        live_restart_epoch=2,
        detection_settings={},
        clip_store_subdir=None,
        facility_id="facility-1",
        policy_generation=3,
        policy_bundle=bundle,  # type: ignore[arg-type]
    )
    response = assemble_worker_config(inputs)
    assert response["detection_policies"]["module_id"] == "fall"
    assert response["cameras"][0]["facility_id"] == "facility-1"
    assert response["config_version"] == 7 * 1_000_000_000 + 0x42
    assert response["restart_epoch"] == 5


def test_compute_policy_camera_identities_filters_and_maps_ids() -> None:
    snapshot = _snapshot(
        [
            {"id": "local-1", "backend_camera_id": None, "rtsp_url": "rtsp://a"},
            {"id": "local-2", "backend_camera_id": "hub-2", "rtsp_url": "rtsp://b"},
            {"id": "skip-blank", "backend_camera_id": "hub-3", "rtsp_url": " "},
        ]
    )
    identities = compute_policy_camera_identities(snapshot)
    assert tuple(i.camera_id for i in identities) == ("local-1", "hub-2")


def test_assemble_worker_config_keeps_live_pulled_versions() -> None:
    inputs = WorkerConfigInputs(
        registry_snapshot=_snapshot(
            [{"id": "local-1", "backend_camera_id": None, "rtsp_url": "rtsp://a"}]
        ),
        bed_zones={},
        pulled=PulledWorkerConfig(
            config_version=7, restart_epoch=2, night_window=None, cameras=(), detection_windows={}
        ),
        live_config_version=7,
        live_restart_epoch=2,
        detection_settings={},
        clip_store_subdir=None,
        facility_id=None,
        policy_generation=0,
        policy_bundle=None,  # generation=0 path ignores bundle
    )
    response = assemble_worker_config(inputs)
    assert response["registry_version"] == 1
    assert response["cameras"][0]["camera_id"] == "local-1"
    assert response["config_version"] == 7
    assert response["restart_epoch"] == 2


def test_worker_config_inputs_is_frozen() -> None:
    inputs = WorkerConfigInputs(
        registry_snapshot=_snapshot([]),
        bed_zones={},
        pulled=None,
        live_config_version=0,
        live_restart_epoch=0,
        detection_settings={},
        clip_store_subdir=None,
        facility_id=None,
        policy_generation=0,
        policy_bundle=None,
    )
    with pytest.raises(FrozenInstanceError):
        inputs.live_config_version = 1  # type: ignore[misc]

