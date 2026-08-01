from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from worker.pipeline.output.evidence.clip_recorder_models import (
    ClipRecorderConfig,
    ClipRecorderStats,
)
from worker.pipeline.output.evidence.evidence_retention import (
    DiskUsage,
    EvidenceRetention,
    PurgeCandidate,
)


class ClipMaintenance:
    def __init__(
        self,
        config: ClipRecorderConfig,
        stats: ClipRecorderStats,
        *,
        is_clip_held: Callable[[str], bool],
        disk_usage_provider: Callable[[Path], DiskUsage],
    ) -> None:
        self._config = config
        self._stats = stats
        self._retention = EvidenceRetention(
            config.store_dir,
            is_held=is_clip_held,
            disk_usage_provider=disk_usage_provider,
        )
        self._last_rotate_monotonic: float | None = None

    def sweep_stale_staging(self) -> None:
        staging_root = self._config.store_dir / "clips" / ".staging"
        cutoff = time.time() - max(0.0, self._config.stale_staging_seconds)
        cleaned = 0
        for staging_dir in staging_root.iterdir():
            try:
                if self._retention.is_held(staging_dir.name):
                    continue
                if not staging_dir.is_dir() or staging_dir.stat().st_mtime > cutoff:
                    continue
                shutil.rmtree(staging_dir)
                cleaned += 1
            except OSError:
                continue
        self._stats.stale_staging_cleaned += cleaned

    def rotate(self, *, force: bool = False) -> None:
        now_monotonic = time.monotonic()
        if (
            not force
            and self._last_rotate_monotonic is not None
            and now_monotonic - self._last_rotate_monotonic
            < self._config.rotate_min_interval_seconds
        ):
            return
        self._last_rotate_monotonic = now_monotonic
        report = self._retention.rotate(
            (
                PurgeCandidate(
                    clip_id=clip_dir.name,
                    clip_dir=clip_dir,
                    finalized_at=finalized_at,
                )
                for finalized_at, clip_dir in finalized_clips(self._config.store_dir)
            ),
            retention_cutoff=datetime.now(UTC)
            - timedelta(days=self._config.retention_days),
            disk_high_watermark=self._config.disk_high_watermark,
        )
        self._stats.held_clips = len(report.held_clip_ids)
        self._stats.purge_failures += len(report.failure_clip_ids)
        self._stats.recording_suspended = report.pressure_blocked


def finalized_clips(store_dir: Path) -> list[tuple[datetime, Path]]:
    root = store_dir / "clips"
    if not root.exists():
        return []
    clips: list[tuple[datetime, Path]] = []
    for manifest_path in root.glob("*/manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("finalized") is not True:
            continue
        finalized_at = _parse_utc(str(payload.get("finalized_at", "")))
        if finalized_at is None:
            try:
                finalized_at = datetime.fromtimestamp(manifest_path.stat().st_mtime, UTC)
            except OSError:
                continue
        clips.append((finalized_at, manifest_path.parent))
    return clips


def _parse_utc(value: str) -> datetime | None:
    if value == "":
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def default_disk_usage(path: Path) -> DiskUsage:
    return shutil.disk_usage(path)


__all__ = ["ClipMaintenance", "default_disk_usage", "finalized_clips"]
