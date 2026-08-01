from __future__ import annotations

from pathlib import Path

import pytest

from worker.pipeline.ingest.registry import (
    SourceRecord,
    SourceRegistry,
    SourceRegistryError,
)


@pytest.mark.parametrize(
    ("descriptor", "message"),
    [
        ("0", "device indexes are not accepted"),
        ("camera:0", "raw live descriptors are not accepted"),
        ("device:1", "raw live descriptors are not accepted"),
        ("rtsp://camera/live", "raw live descriptors are not accepted"),
        ("https://camera/live", "raw live descriptors are not accepted"),
        ("../outside.mp4", "raw paths and traversal are not accepted"),
        ("/tmp/outside.mp4", "raw paths and traversal are not accepted"),
        ("~/outside.mp4", "raw paths and traversal are not accepted"),
    ],
)
def test_source_registry_rejects_raw_descriptors(
    tmp_path: Path,
    descriptor: str,
    message: str,
) -> None:
    registry = SourceRegistry(base_dir=tmp_path, records={})

    with pytest.raises(SourceRegistryError, match=message):
        registry.resolve(source_id=descriptor)


def test_source_registry_rejects_registered_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"video")
    registry = SourceRegistry(
        base_dir=tmp_path,
        records={
            "outside": SourceRecord("outside", outside, 1.0, "video/mp4"),
        },
    )

    with pytest.raises(SourceRegistryError, match="escapes api base directory"):
        registry.resolve(source_id="outside")


def test_source_registry_rejects_untrusted_live_record(tmp_path: Path) -> None:
    registry = SourceRegistry(
        base_dir=tmp_path,
        records={
            "lobby": SourceRecord(
                "lobby",
                Path("rtsp:/camera/live"),
                0.0,
                "",
                kind="live",
            ),
        },
    )

    with pytest.raises(SourceRegistryError, match="allowlisted trusted source_id"):
        registry.resolve(source_id="lobby")


def test_source_registry_resolves_trusted_live_record_without_path_validation(
    tmp_path: Path,
) -> None:
    record = SourceRecord(
        "lobby",
        Path("rtsp:/camera/live"),
        0.0,
        "",
        kind="live",
        trusted_live=True,
    )
    registry = SourceRegistry(base_dir=tmp_path, records={"lobby": record})

    resolved = registry.resolve(source_id="lobby")

    assert resolved.record is record
    assert resolved.path == record.path
    assert resolved.is_live is True


def test_source_registry_resolves_safe_stored_video(tmp_path: Path) -> None:
    video = tmp_path / "nested" / "clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    record = SourceRecord("clip", Path("nested/clip.mp4"), 1.5, "video/mp4")
    registry = SourceRegistry(base_dir=tmp_path, records={"clip": record})

    resolved = registry.resolve(upload_id="clip")

    assert resolved.record is record
    assert resolved.path == video.resolve()
    assert resolved.is_live is False


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (SourceRecord("clip", Path("clip.txt"), 1.0, "video/mp4"), "unsupported video"),
        (SourceRecord("clip", Path("clip.mp4"), 1.0, "text/plain"), "unsupported source MIME"),
        (SourceRecord("clip", Path("clip.mp4"), 0.0, "video/mp4"), "must be positive"),
        (SourceRecord("clip", Path("clip.mp4"), 121.0, "video/mp4"), "exceeds registry"),
    ],
)
def test_source_registry_preserves_stored_metadata_validation(
    tmp_path: Path,
    record: SourceRecord,
    message: str,
) -> None:
    path = tmp_path / record.path
    path.write_bytes(b"video")
    registry = SourceRegistry(base_dir=tmp_path, records={"clip": record})

    with pytest.raises(SourceRegistryError, match=message):
        registry.resolve(source_id="clip")


@pytest.mark.parametrize(
    ("source_id", "upload_id"),
    [(None, None), ("clip", "upload")],
)
def test_source_registry_requires_exactly_one_identifier(
    tmp_path: Path,
    source_id: str | None,
    upload_id: str | None,
) -> None:
    registry = SourceRegistry(base_dir=tmp_path, records={})

    with pytest.raises(SourceRegistryError, match="exactly one"):
        registry.resolve(source_id=source_id, upload_id=upload_id)
