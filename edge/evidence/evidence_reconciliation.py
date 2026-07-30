"""Restart reconciliation between atomic clip manifests and the WAL outbox."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from edge.evidence.evidence_manifest import (
    ClipEvidenceError,
    ReadyClipManifest,
    UnavailableClipManifest,
    parse_manifest,
    verify_ready_manifest,
)
from edge.evidence.evidence_outbox import EvidenceOutbox
from edge.evidence.evidence_outbox_types import (
    ClipId,
    ClipLocalState,
    ClipOutcome,
    ClipOutcomeConflictError,
    EdgeEventId,
    EventClipConflictError,
    EvidenceReasonCode,
    MissingStagedEventError,
)


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    verified: int = 0
    unavailable: int = 0
    corrupt: int = 0


def reconcile_finalized_clip(
    store_dir: Path,
    clip_id: ClipId,
    outbox: EvidenceOutbox,
) -> ClipLocalState:
    """Reconcile one recorder-published immutable clip without scanning live staging."""
    clip_dir = store_dir / "clips" / clip_id
    return _reconcile_final_dir(store_dir, clip_dir, clip_id, outbox)


def reconcile_event_evidence(store_dir: Path, outbox: EvidenceOutbox) -> ReconciliationReport:
    clips_root = store_dir / "clips"
    staging_root = clips_root / ".staging"
    clips_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    verified = 0
    unavailable = 0
    corrupt = 0
    seen: set[ClipId] = set()
    for clip_dir in sorted(clips_root.iterdir(), key=lambda path: path.name):
        if clip_dir.name == ".staging":
            continue
        clip_id = ClipId(clip_dir.name)
        seen.add(clip_id)
        outcome = _reconcile_final_dir(store_dir, clip_dir, clip_id, outbox)
        match outcome:
            case ClipLocalState.VERIFIED:
                verified += 1
            case ClipLocalState.UNAVAILABLE:
                unavailable += 1
            case ClipLocalState.CORRUPT:
                corrupt += 1
            case ClipLocalState.AWAITING_FINALIZE:
                raise AssertionError("reconciliation cannot retain awaiting finalization")
    for staging_dir in sorted(staging_root.iterdir(), key=lambda path: path.name):
        clip_id = ClipId(staging_dir.name)
        if clip_id in seen or outbox.clip_outcome(clip_id) is None:
            continue
        outbox.record_clip_outcome(
            _unavailable_outcome(
                clip_id,
                EvidenceReasonCode.INTERRUPTED_FINALIZE,
                _relative_or_none(store_dir, staging_dir / "manifest.json"),
            )
        )
        seen.add(clip_id)
        unavailable += 1
    for clip_id in outbox.awaiting_clip_ids():
        if clip_id in seen:
            continue
        outbox.record_clip_outcome(
            _unavailable_outcome(clip_id, EvidenceReasonCode.MISSING, None)
        )
        unavailable += 1
    return ReconciliationReport(verified=verified, unavailable=unavailable, corrupt=corrupt)


def _reconcile_final_dir(
    store_dir: Path,
    clip_dir: Path,
    clip_id: ClipId,
    outbox: EvidenceOutbox,
) -> ClipLocalState:
    manifest_path = clip_dir / "manifest.json"
    relative_manifest = _relative_or_none(store_dir, manifest_path)
    event_ids: tuple[EdgeEventId, ...] = ()
    try:
        _validate_clip_directory(clip_dir)
        manifest = parse_manifest(manifest_path)
        _validate_clip_identity(manifest.clip_id, clip_id)
        event_ids = tuple(EdgeEventId(event_ref) for event_ref in manifest.event_refs)
        match manifest:
            case ReadyClipManifest():
                video_path = _validated_video_path(clip_dir)
                verify_ready_manifest(manifest, video_path)
                outbox.reconcile_clip(
                    event_ids,
                    ClipOutcome(
                        clip_id=clip_id,
                        local_state=ClipLocalState.VERIFIED,
                        manifest_path=relative_manifest,
                        state_version=manifest.state_version,
                        media_relpath=_relative_or_none(store_dir, video_path),
                        sha256=manifest.sha256,
                        size_bytes=manifest.size_bytes,
                        mime_type=manifest.mime_type,
                        codec=manifest.codec,
                        duration_ms=manifest.duration_ms,
                        clip_start_at=manifest.clip_start_at,
                        clip_end_at=manifest.clip_end_at,
                        finalized_at=manifest.finalized_at,
                    )
                )
                return ClipLocalState.VERIFIED
            case UnavailableClipManifest():
                outbox.reconcile_clip(
                    event_ids,
                    ClipOutcome(
                        clip_id=clip_id,
                        local_state=ClipLocalState.UNAVAILABLE,
                        manifest_path=relative_manifest,
                        state_version=manifest.state_version,
                        clip_start_at=manifest.clip_start_at,
                        clip_end_at=manifest.clip_end_at,
                        finalized_at=manifest.finalized_at,
                        unavailable_reason=manifest.reason_code,
                    )
                )
                return ClipLocalState.UNAVAILABLE
    except (
        ClipEvidenceError,
        ClipOutcomeConflictError,
        EventClipConflictError,
        MissingStagedEventError,
    ):
        _record_corrupt(outbox, clip_id, relative_manifest, event_ids)
        return ClipLocalState.CORRUPT


def _record_corrupt(
    outbox: EvidenceOutbox,
    clip_id: ClipId,
    manifest_path: str | None,
    event_ids: tuple[EdgeEventId, ...],
) -> None:
    outcome = ClipOutcome(
        clip_id=clip_id,
        local_state=ClipLocalState.CORRUPT,
        manifest_path=manifest_path,
        state_version=2,
        unavailable_reason=EvidenceReasonCode.CORRUPT,
    )
    if event_ids:
        try:
            outbox.reconcile_clip(event_ids, outcome)
        except (
            ClipOutcomeConflictError,
            EventClipConflictError,
            MissingStagedEventError,
        ):
            pass
        else:
            return
    outbox.record_clip_outcome(outcome)


def _validate_clip_directory(clip_dir: Path) -> None:
    if clip_dir.is_symlink() or not clip_dir.is_dir():
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "clip directory invalid")


def _validate_clip_identity(manifest_clip_id: str, expected_clip_id: ClipId) -> None:
    if manifest_clip_id != expected_clip_id:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "clip identity mismatch")


def _validated_video_path(clip_dir: Path) -> Path:
    video_path = clip_dir / "clip.mp4"
    if video_path.parent.resolve() != clip_dir.resolve():
        raise ClipEvidenceError(
            EvidenceReasonCode.CORRUPT,
            "media path escaped clip",
        )
    return video_path


def _unavailable_outcome(
    clip_id: ClipId,
    reason_code: EvidenceReasonCode,
    manifest_path: str | None,
) -> ClipOutcome:
    return ClipOutcome(
        clip_id=clip_id,
        local_state=ClipLocalState.UNAVAILABLE,
        manifest_path=manifest_path,
        state_version=2,
        unavailable_reason=reason_code,
    )


def _relative_or_none(store_dir: Path, path: Path) -> str | None:
    try:
        return str(path.relative_to(store_dir))
    except ValueError:
        return None


__all__ = [
    "ReconciliationReport",
    "reconcile_event_evidence",
    "reconcile_finalized_clip",
]
