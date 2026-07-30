from __future__ import annotations

from typing import TypedDict


class CameraConfigPayload(TypedDict, total=False):
    camera_id: str
    facility_id: str
    resident_id: str | None
    rtsp_url: str
    heartbeat_interval_sec: int
    frame_stride: int
    label: str


class EdgeConfigPayload(TypedDict):
    relay: dict[str, str]
    cameras: list[CameraConfigPayload]


def edge_config_payload(
    *,
    camera_count: int = 4,
    include_optional_fields: bool = True,
    resident_ids: bool = True,
) -> EdgeConfigPayload:
    cameras = [
        _camera_config_payload(
            index,
            include_optional_fields=include_optional_fields,
            resident_ids=resident_ids,
        )
        for index in range(1, camera_count + 1)
    ]
    return {
        "relay": {
            "url": "http://127.0.0.1:8000",
            "token": "relay-token-1",
        },
        "cameras": cameras,
    }


def _camera_config_payload(
    index: int,
    *,
    include_optional_fields: bool,
    resident_ids: bool,
) -> CameraConfigPayload:
    camera: CameraConfigPayload = {
        "camera_id": f"camera-{index}",
        "facility_id": "facility-1",
        "resident_id": f"resident-{index}" if resident_ids else None,
        "rtsp_url": f"rtsp://camera-{index}/trackID=2",
    }
    if include_optional_fields:
        camera.update(
            {
                "heartbeat_interval_sec": 30,
                "frame_stride": 1,
                "label": f"Room {index}",
            }
        )
    return camera
