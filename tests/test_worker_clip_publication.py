from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import worker.pipeline.output.evidence.clip_publication as clip_publication
from backend.app.features.clips.catalog import strict_manifest_records
from backend.app.features.clips.store import ClipStore
from worker.pipeline.output.evidence.clip_identity import ClipIdAllocator
from worker.pipeline.output.evidence.clip_publication import (
    ClipPublicationMetadata,
    ClipPublisher,
    PublicationStage,
)
from worker.pipeline.output.evidence.evidence_media import MediaFacts
from worker.pipeline.output.evidence.evidence_outbox_types import EdgeEventId, EvidenceReasonCode

EVENT_ONE = EdgeEventId("00000000-0000-4000-8000-000000000001")
EVENT_TWO = EdgeEventId("00000000-0000-4000-8000-000000000002")
START = datetime(2026, 7, 16, 1, 2, 3, tzinfo=UTC)


def _metadata() -> ClipPublicationMetadata:
    return ClipPublicationMetadata(
        camera_id="camera-1",
        event_refs=(EVENT_ONE, EVENT_ONE, EVENT_TWO),
        event_type="fall",
        clip_start_at=START,
        clip_end_at=START + timedelta(seconds=1),
        finalized_at=START + timedelta(seconds=2),
        started_at=START,
        duration_s=1.0,
        encoder="libx264",
    )


def test_reservation_skips_collisions_and_keeps_the_caller_visible_identity(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "clips" / "collision"
    existing.mkdir(parents=True)
    ids = iter(("collision", "reserved"))
    allocator = ClipIdAllocator(tmp_path, id_factory=lambda _camera: next(ids))

    reservation = allocator.reserve("camera-1")

    assert reservation.clip_id == "reserved"
    assert reservation.staging_dir.is_dir()
    assert existing.is_dir()
    assert allocator.collision_count == 1


@pytest.mark.parametrize(
    "failure_stage",
    (
        PublicationStage.MEDIA_FSYNCED,
        PublicationStage.MEDIA_RENAMED,
        PublicationStage.MANIFEST_RENAMED,
    ),
)
def test_publication_retry_recovers_each_durability_boundary_with_same_clip_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: PublicationStage,
) -> None:
    reservation = ClipIdAllocator(
        tmp_path,
        id_factory=lambda _camera: "stable-clip-id",
    ).reserve("camera-1")
    artifact = reservation.staging_dir / "clip.mp4"
    artifact.write_bytes(b"derivative-media")
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts("a" * 64, len(b"derivative-media"), 1000),
    )
    failed = False

    def interrupt(stage: PublicationStage, _path: Path) -> None:
        nonlocal failed
        if stage is failure_stage and not failed:
            failed = True
            raise OSError(f"interrupted after {stage}")

    with pytest.raises(OSError, match="interrupted"):
        ClipPublisher(tmp_path, barrier=interrupt).publish_ready(
            reservation,
            artifact,
            _metadata(),
        )

    published = ClipPublisher(tmp_path).publish_ready(
        reservation,
        artifact,
        _metadata(),
    )
    payload = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    records = strict_manifest_records(ClipStore(tmp_path))

    assert failed
    assert published.clip_id == "stable-clip-id"
    assert payload["event_refs"] == [EVENT_ONE, EVENT_TWO]
    assert payload["state"] == "READY"
    assert payload["path"] == "clips/stable-clip-id/clip.mp4"
    assert records[0].manifest.clip_id == "stable-clip-id"
    assert not reservation.staging_dir.exists()


def test_unavailable_publication_persists_reason_without_video(tmp_path: Path) -> None:
    reservation = ClipIdAllocator(
        tmp_path,
        id_factory=lambda _camera: "unavailable-clip-id",
    ).reserve("camera-1")

    published = ClipPublisher(tmp_path).publish_unavailable(
        reservation,
        _metadata(),
        EvidenceReasonCode.ENCODER_FAILED,
    )
    payload = json.loads(published.manifest_path.read_text(encoding="utf-8"))

    assert published.video_path is None
    assert payload["state"] == "UNAVAILABLE"
    assert payload["reason_code"] == "ENCODER_FAILED"
    assert payload["path"] is None
    assert payload["video_available"] is False
    assert not (reservation.final_dir / "clip.mp4").exists()
    assert not reservation.staging_dir.exists()


def test_ready_publication_fsyncs_before_renames_and_staging_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation = ClipIdAllocator(
        tmp_path,
        id_factory=lambda _camera: "durable-clip-id",
    ).reserve("camera-1")
    artifact = reservation.staging_dir / "derivative.mp4"
    artifact.write_bytes(b"derivative-media")
    monkeypatch.setattr(
        "worker.pipeline.output.evidence.evidence_manifest.inspect_finalized_media",
        lambda _path, **_kwargs: MediaFacts("a" * 64, len(b"derivative-media"), 1000),
    )
    operations: list[tuple[str, str, str]] = []
    real_replace = os.replace
    real_rmtree = shutil.rmtree

    def record_file(path: Path) -> None:
        operations.append(("fsync-file", path.name, ""))

    def record_directory(path: Path) -> None:
        operations.append(("fsync-directory", path.name, ""))

    def record_replace(source: Path, target: Path) -> None:
        operations.append(("replace", source.name, target.name))
        real_replace(source, target)

    def record_rmtree(path: Path) -> None:
        operations.append(("rmtree", path.name, ""))
        real_rmtree(path)

    monkeypatch.setattr(clip_publication, "fsync_file", record_file)
    monkeypatch.setattr(clip_publication, "fsync_directory", record_directory)
    monkeypatch.setattr(clip_publication.os, "replace", record_replace)
    monkeypatch.setattr(clip_publication.shutil, "rmtree", record_rmtree)

    _ = ClipPublisher(tmp_path).publish_ready(reservation, artifact, _metadata())

    assert operations == [
        ("fsync-directory", "clips", ""),
        ("fsync-file", "derivative.mp4", ""),
        ("replace", "derivative.mp4", "clip.mp4"),
        ("fsync-file", "clip.mp4", ""),
        ("fsync-directory", "durable-clip-id", ""),
        ("replace", "manifest.json.tmp", "manifest.json"),
        ("fsync-file", "manifest.json", ""),
        ("fsync-directory", "durable-clip-id", ""),
        ("rmtree", "durable-clip-id", ""),
        ("fsync-directory", ".staging", ""),
    ]
