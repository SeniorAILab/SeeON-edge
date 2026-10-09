from __future__ import annotations


def camera_filter_ids(registry: object | None, camera_id: str | None) -> tuple[str, ...] | None:
    if camera_id is None:
        return None
    snapshot = getattr(registry, "snapshot", None)
    cameras = snapshot().get("cameras") if callable(snapshot) else None
    for record in cameras if isinstance(cameras, list) else []:
        if not isinstance(record, dict):
            continue
        ids = tuple(
            value
            for value in (record.get("id"), record.get("backend_camera_id"))
            if isinstance(value, str) and value
        )
        if camera_id in ids:
            return ids
    return (camera_id,)


__all__ = ["camera_filter_ids"]
