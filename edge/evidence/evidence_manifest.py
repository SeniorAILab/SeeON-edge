"""Schema-v2 event evidence manifests and immutable media validation."""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Literal, Self, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator
from pydantic_core import PydanticCustomError

from edge.evidence.evidence_media import ClipEvidenceError, inspect_finalized_media
from edge.evidence.evidence_outbox_types import ClipId, EdgeEventId, EvidenceReasonCode

MAX_MANIFEST_BYTES = 64 * 1024


class ReadyClipManifest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    manifest_schema_version: Literal[2] = 2
    state: Literal["READY"] = "READY"
    clip_id: str
    camera_id: str
    event_refs: tuple[str, ...]
    clip_start_at: str
    clip_end_at: str
    finalized_at: str
    sha256: str
    size_bytes: int
    mime_type: Literal["video/mp4"] = "video/mp4"
    codec: Literal["h264"] = "h264"
    duration_ms: int
    state_version: Literal[2] = 2

    @field_validator("clip_id", "camera_id")
    @classmethod
    def _nonempty_identity(cls, value: str) -> str:
        if value.strip() == "":
            raise PydanticCustomError("empty_identity", "identity must not be empty")
        return value

    @field_validator("event_refs")
    @classmethod
    def _ordered_unique_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_event_refs(values)

    @field_validator("clip_start_at", "clip_end_at", "finalized_at")
    @classmethod
    def _utc_timestamp(cls, value: str) -> str:
        return _normalized_timestamp(value)

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise PydanticCustomError("sha256", "sha256 must be lowercase hexadecimal")
        return value

    @field_validator("size_bytes")
    @classmethod
    def _positive_size(cls, value: int) -> int:
        if value <= 0:
            raise PydanticCustomError("size_bytes", "size_bytes must be positive")
        return value

    @field_validator("duration_ms")
    @classmethod
    def _bounded_duration(cls, value: int) -> int:
        if not 1 <= value <= 120_000:
            raise PydanticCustomError("duration_ms", "duration_ms is out of range")
        return value

    @model_validator(mode="after")
    def _ordered_timestamps(self) -> Self:
        start = _parse_utc(self.clip_start_at)
        end = _parse_utc(self.clip_end_at)
        finalized = _parse_utc(self.finalized_at)
        if not start <= end <= finalized:
            raise PydanticCustomError("timestamps", "clip timestamps are not ordered")
        return self


class UnavailableClipManifest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    manifest_schema_version: Literal[2] = 2
    state: Literal["UNAVAILABLE"] = "UNAVAILABLE"
    clip_id: str
    camera_id: str
    event_refs: tuple[str, ...]
    clip_start_at: str
    clip_end_at: str
    finalized_at: str
    state_version: Literal[2] = 2
    reason_code: EvidenceReasonCode

    @field_validator("event_refs")
    @classmethod
    def _ordered_unique_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_event_refs(values)

    @field_validator("clip_start_at", "clip_end_at", "finalized_at")
    @classmethod
    def _utc_timestamp(cls, value: str) -> str:
        return _normalized_timestamp(value)

    @model_validator(mode="after")
    def _ordered_timestamps(self) -> Self:
        start = _parse_utc(self.clip_start_at)
        end = _parse_utc(self.clip_end_at)
        finalized = _parse_utc(self.finalized_at)
        if not start <= end <= finalized:
            raise PydanticCustomError("timestamps", "clip timestamps are not ordered")
        return self


ClipManifest: TypeAlias = ReadyClipManifest | UnavailableClipManifest


def finalize_ready_manifest(
    *,
    video_path: Path,
    clip_id: ClipId,
    camera_id: str,
    event_refs: tuple[EdgeEventId, ...],
    clip_start_at: datetime,
    clip_end_at: datetime,
    finalized_at: datetime,
    ffprobe_bin: str = "ffprobe",
) -> ReadyClipManifest:
    facts = inspect_finalized_media(video_path, ffprobe_bin=ffprobe_bin)
    return ReadyClipManifest(
        clip_id=clip_id,
        camera_id=camera_id,
        event_refs=_coalesce_event_refs(event_refs),
        clip_start_at=_rfc3339_milliseconds(clip_start_at),
        clip_end_at=_rfc3339_milliseconds(clip_end_at),
        finalized_at=_rfc3339_milliseconds(finalized_at),
        sha256=facts.sha256,
        size_bytes=facts.size_bytes,
        duration_ms=facts.duration_ms,
    )


def unavailable_manifest(
    *,
    clip_id: ClipId,
    camera_id: str,
    event_refs: tuple[EdgeEventId, ...],
    clip_start_at: datetime,
    clip_end_at: datetime,
    finalized_at: datetime,
    reason_code: EvidenceReasonCode,
) -> UnavailableClipManifest:
    return UnavailableClipManifest(
        clip_id=clip_id,
        camera_id=camera_id,
        event_refs=_coalesce_event_refs(event_refs),
        clip_start_at=_rfc3339_milliseconds(clip_start_at),
        clip_end_at=_rfc3339_milliseconds(clip_end_at),
        finalized_at=_rfc3339_milliseconds(finalized_at),
        reason_code=reason_code,
    )


def verify_ready_manifest(
    manifest: ReadyClipManifest,
    video_path: Path,
    *,
    ffprobe_bin: str = "ffprobe",
) -> None:
    facts = inspect_finalized_media(video_path, ffprobe_bin=ffprobe_bin)
    if facts.sha256 != manifest.sha256 or facts.size_bytes != manifest.size_bytes:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "immutable bytes mismatch")
    if facts.duration_ms != manifest.duration_ms:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "duration mismatch")


def parse_manifest(path: Path) -> ClipManifest:
    try:
        payload = json.loads(_read_manifest(path))
        state = payload.get("state") if isinstance(payload, dict) else None
        match state:  # noqa: MATCH_OK - validates untrusted manifest state
            case "READY":
                return ReadyClipManifest.model_validate(payload)
            case "UNAVAILABLE":
                return UnavailableClipManifest.model_validate(payload)
            case _:
                raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "manifest state invalid")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ClipEvidenceError(EvidenceReasonCode.CORRUPT, "manifest invalid") from exc


def _read_manifest(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_MANIFEST_BYTES:
            raise OSError("manifest file shape invalid")
        chunks: list[bytes] = []
        remaining = MAX_MANIFEST_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(8192, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > MAX_MANIFEST_BYTES:
            raise OSError("manifest exceeds size limit")
        return payload.decode("utf-8")
    finally:
        os.close(descriptor)


def _validate_event_refs(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        raise PydanticCustomError("event_refs", "event_refs must not be empty")
    if len(set(values)) != len(values):
        raise PydanticCustomError("event_refs", "event_refs must be unique")
    for value in values:
        try:
            parsed = UUID(value)
        except (ValueError, AttributeError) as exc:
            raise PydanticCustomError(
                "event_refs",
                "event_refs must be canonical UUIDv4",
            ) from exc
        if parsed.version != 4 or str(parsed) != value:
            raise PydanticCustomError(
                "event_refs",
                "event_refs must be canonical UUIDv4",
            )
    return values


def _coalesce_event_refs(values: tuple[EdgeEventId, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _normalized_timestamp(value: str) -> str:
    parsed = _parse_utc(value)
    body = value.removesuffix("Z").removesuffix("+00:00")
    if "." in body and len(body.rsplit(".", maxsplit=1)[1]) == 3:
        return _rfc3339_milliseconds(parsed)
    return _rfc3339_z(parsed)


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PydanticCustomError("utc_timestamp", "timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PydanticCustomError("utc_timestamp", "timestamp must be aware")
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise PydanticCustomError("utc_timestamp", "timestamp must be UTC")
    return parsed.astimezone(UTC)


def _rfc3339_milliseconds(value: datetime) -> str:
    return (
        value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )


def _rfc3339_z(value: datetime) -> str:
    normalized = value.astimezone(UTC).isoformat(timespec="microseconds")
    body = normalized.removesuffix("+00:00")
    if "." in body:
        prefix, fraction = body.split(".", maxsplit=1)
        body = prefix if fraction.rstrip("0") == "" else f"{prefix}.{fraction.rstrip('0')}"
    return f"{body}Z"


__all__ = [
    "ClipEvidenceError",
    "ClipManifest",
    "EvidenceReasonCode",
    "ReadyClipManifest",
    "UnavailableClipManifest",
    "finalize_ready_manifest",
    "parse_manifest",
    "unavailable_manifest",
    "verify_ready_manifest",
]
