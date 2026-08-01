"""Validated schema-v2 evidence manifest models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar, Literal, Self, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic_core import PydanticCustomError

from worker.pipeline.output.evidence.evidence_outbox_types import EvidenceReasonCode


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
        return validate_event_refs(values)

    @field_validator("clip_start_at", "clip_end_at", "finalized_at")
    @classmethod
    def _utc_timestamp(cls, value: str) -> str:
        return normalized_timestamp(value)

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
        start = parse_utc(self.clip_start_at)
        end = parse_utc(self.clip_end_at)
        finalized = parse_utc(self.finalized_at)
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

    @field_validator("clip_id", "camera_id")
    @classmethod
    def _nonempty_identity(cls, value: str) -> str:
        if value.strip() == "":
            raise PydanticCustomError("empty_identity", "identity must not be empty")
        return value

    @field_validator("event_refs")
    @classmethod
    def _ordered_unique_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return validate_event_refs(values)

    @field_validator("clip_start_at", "clip_end_at", "finalized_at")
    @classmethod
    def _utc_timestamp(cls, value: str) -> str:
        return normalized_timestamp(value)

    @model_validator(mode="after")
    def _ordered_timestamps(self) -> Self:
        start = parse_utc(self.clip_start_at)
        end = parse_utc(self.clip_end_at)
        finalized = parse_utc(self.finalized_at)
        if not start <= end <= finalized:
            raise PydanticCustomError("timestamps", "clip timestamps are not ordered")
        return self


ClipManifest: TypeAlias = ReadyClipManifest | UnavailableClipManifest


def validate_event_refs(values: tuple[str, ...]) -> tuple[str, ...]:
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


def coalesce_event_refs(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def normalized_timestamp(value: str) -> str:
    parsed = parse_utc(value)
    body = value.removesuffix("Z").removesuffix("+00:00")
    if "." in body and len(body.rsplit(".", maxsplit=1)[1]) == 3:
        return rfc3339_milliseconds(parsed)
    return rfc3339_z(parsed)


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PydanticCustomError("utc_timestamp", "timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PydanticCustomError("utc_timestamp", "timestamp must be aware")
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise PydanticCustomError("utc_timestamp", "timestamp must be UTC")
    return parsed.astimezone(UTC)


def rfc3339_milliseconds(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def rfc3339_z(value: datetime) -> str:
    normalized = value.astimezone(UTC).isoformat(timespec="microseconds")
    body = normalized.removesuffix("+00:00")
    if "." in body:
        prefix, fraction = body.split(".", maxsplit=1)
        body = prefix if fraction.rstrip("0") == "" else f"{prefix}.{fraction.rstrip('0')}"
    return f"{body}Z"


__all__ = [
    "ClipManifest",
    "ReadyClipManifest",
    "UnavailableClipManifest",
    "coalesce_event_refs",
    "parse_utc",
    "rfc3339_milliseconds",
]
