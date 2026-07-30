"""Hold-aware, verified local evidence retention."""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import ClassVar, Literal, Protocol, assert_never, final

from pydantic import BaseModel, ConfigDict, ValidationError


class DiskUsage(Protocol):
    @property
    def total(self) -> int: ...

    @property
    def used(self) -> int: ...

    @property
    def free(self) -> int: ...


class PurgeResult(StrEnum):
    PURGED = "PURGED"
    HELD = "HELD"
    MISSING = "MISSING"
    UNVERIFIABLE = "UNVERIFIABLE"
    DELETE_FAILED = "DELETE_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"


@dataclass(frozen=True, slots=True)
class PurgeCandidate:
    clip_id: str
    clip_dir: Path
    finalized_at: datetime


@dataclass(frozen=True, slots=True)
class RotationReport:
    purged_clip_ids: tuple[str, ...]
    held_clip_ids: tuple[str, ...]
    failure_clip_ids: tuple[str, ...]
    pressure_blocked: bool


class _RetentionManifest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True, strict=True)

    clip_id: str
    finalized: bool = False
    manifest_schema_version: int | None = None
    finalized_at: str | None = None
    path: str | None = None
    media_relpath: str | None = None
    video_available: bool | None = None
    local_state: Literal[
        "AWAITING_FINALIZE", "VERIFIED", "UNAVAILABLE", "CORRUPT"
    ] | None = None
    state: Literal["READY", "UNAVAILABLE"] | None = None


@final
class EvidenceRetention:
    """Select and remove only owned, verified, remotely releasable evidence."""

    def __init__(
        self,
        store_dir: Path,
        *,
        is_held: Callable[[str], bool],
        disk_usage_provider: Callable[[Path], DiskUsage],
    ) -> None:
        self._store_dir = store_dir
        self._clips_dir = store_dir / "clips"
        self._is_held = is_held
        self._disk_usage_provider = disk_usage_provider

    def is_held(self, clip_id: str) -> bool:
        return self._is_held(clip_id)

    def purge(self, candidate: PurgeCandidate) -> PurgeResult:
        if self._is_held(candidate.clip_id):
            return PurgeResult.HELD
        verification = self._verify_candidate(candidate)
        if verification is not None:
            return verification
        try:
            shutil.rmtree(candidate.clip_dir)
        except FileNotFoundError:
            return PurgeResult.MISSING
        except OSError:
            return PurgeResult.DELETE_FAILED
        if os.path.lexists(candidate.clip_dir):
            return PurgeResult.VERIFICATION_FAILED
        return PurgeResult.PURGED

    def rotate(
        self,
        candidates: Iterable[PurgeCandidate],
        *,
        retention_cutoff: datetime,
        disk_high_watermark: float,
    ) -> RotationReport:
        ordered = sorted(candidates, key=lambda candidate: candidate.finalized_at)
        held: list[str] = []
        failures: list[str] = []
        purged: list[str] = []

        def record(candidate: PurgeCandidate, result: PurgeResult) -> None:
            match result:
                case PurgeResult.PURGED:
                    purged.append(candidate.clip_id)
                    return
                case PurgeResult.HELD:
                    held.append(candidate.clip_id)
                    return
                case (
                    PurgeResult.MISSING
                    | PurgeResult.UNVERIFIABLE
                    | PurgeResult.DELETE_FAILED
                    | PurgeResult.VERIFICATION_FAILED
                ):
                    failures.append(candidate.clip_id)
                    return
            assert_never(result)

        for candidate in ordered:
            if self._is_held(candidate.clip_id):
                held.append(candidate.clip_id)
                continue
            if candidate.finalized_at > retention_cutoff:
                continue
            result = self.purge(candidate)
            record(candidate, result)

        return RotationReport(
            purged_clip_ids=tuple(purged),
            held_clip_ids=tuple(held),
            failure_clip_ids=tuple(failures),
            pressure_blocked=self._over_watermark(disk_high_watermark),
        )

    def _over_watermark(self, high_watermark: float) -> bool:
        usage = self._disk_usage_provider(self._store_dir)
        return usage.total > 0 and usage.used / usage.total > high_watermark

    def _verify_candidate(self, candidate: PurgeCandidate) -> PurgeResult | None:
        expected = self._clips_dir / candidate.clip_id
        if candidate.clip_dir != expected or candidate.clip_id in {"", ".", ".."}:
            return PurgeResult.UNVERIFIABLE
        try:
            directory_stat = candidate.clip_dir.lstat()
        except FileNotFoundError:
            return PurgeResult.MISSING
        except OSError:
            return PurgeResult.UNVERIFIABLE
        if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
            return PurgeResult.UNVERIFIABLE
        manifest_path = candidate.clip_dir / "manifest.json"
        try:
            manifest_stat = manifest_path.lstat()
            if stat.S_ISLNK(manifest_stat.st_mode) or not stat.S_ISREG(manifest_stat.st_mode):
                return PurgeResult.UNVERIFIABLE
            manifest = _RetentionManifest.model_validate_json(manifest_path.read_bytes())
        except (FileNotFoundError, OSError, ValidationError):
            return PurgeResult.UNVERIFIABLE
        if manifest.clip_id != candidate.clip_id:
            return PurgeResult.UNVERIFIABLE
        finalized = manifest.finalized
        finalized_v2 = (
            manifest.manifest_schema_version == 2 and manifest.finalized_at is not None
        )
        if not finalized and not finalized_v2:
            return PurgeResult.UNVERIFIABLE
        if manifest.local_state == "CORRUPT":
            return PurgeResult.UNVERIFIABLE
        media_relpath = manifest.path or manifest.media_relpath
        if media_relpath is None:
            unavailable = (
                manifest.video_available is False
                or manifest.local_state == "UNAVAILABLE"
                or manifest.state == "UNAVAILABLE"
            )
            return None if unavailable else PurgeResult.UNVERIFIABLE
        parts = PurePosixPath(media_relpath).parts
        if len(parts) != 3 or parts[:2] != ("clips", candidate.clip_id):
            return PurgeResult.UNVERIFIABLE
        media_path = self._store_dir.joinpath(*parts)
        if media_path.parent != candidate.clip_dir:
            return PurgeResult.UNVERIFIABLE
        try:
            media_stat = media_path.lstat()
        except OSError:
            return PurgeResult.UNVERIFIABLE
        if stat.S_ISLNK(media_stat.st_mode) or not stat.S_ISREG(media_stat.st_mode):
            return PurgeResult.UNVERIFIABLE
        return None


__all__ = [
    "EvidenceRetention",
    "DiskUsage",
    "PurgeCandidate",
    "PurgeResult",
    "RotationReport",
]
