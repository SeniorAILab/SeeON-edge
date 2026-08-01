from __future__ import annotations

from typing import ClassVar, Final, TypeAlias
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from worker.runtime.config.errors import ConfigValidationError

KNOWN_DOMAIN_NAMES: Final = frozenset({"fall", "bed_exit"})


class NightWindowConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    start: str
    end: str
    tz: str

    @field_validator("start", "end")
    @classmethod
    def _validate_time(cls, value: str) -> str:
        hour, separator, minute = value.partition(":")
        valid = (
            separator == ":"
            and hour.isdigit()
            and minute.isdigit()
            and len(hour) == 2
            and len(minute) == 2
            and 0 <= int(hour) <= 23
            and 0 <= int(minute) <= 59
        )
        if not valid:
            raise ConfigValidationError("night window time must use HH:MM")
        return value

    @field_validator("tz")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            _ = ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError) as error:
            raise ConfigValidationError(
                "night window tz must be a valid IANA timezone"
            ) from error
        return value


class FallDomainConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True


class BedExitDomainConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    night_window: NightWindowConfig | None = None


DomainConfig: TypeAlias = FallDomainConfig | BedExitDomainConfig


class DomainsConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    enabled: tuple[str, ...] | None = None
    fall: FallDomainConfig | None = None
    bed_exit: BedExitDomainConfig | None = None

    @field_validator("enabled")
    @classmethod
    def _validate_enabled(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        duplicate = sorted(name for name in set(value) if value.count(name) > 1)
        if duplicate:
            raise ConfigValidationError(
                "domains.enabled contains duplicate domain: " + ", ".join(duplicate)
            )
        unknown = sorted(set(value) - KNOWN_DOMAIN_NAMES)
        if unknown:
            raise ConfigValidationError(
                "domains.enabled contains unknown domain: " + ", ".join(unknown)
            )
        return value

    @model_validator(mode="after")
    def _reject_mixed_legacy_and_map(self) -> DomainsConfig:
        configured_names = KNOWN_DOMAIN_NAMES.intersection(self.model_fields_set)
        if self.enabled is not None and configured_names:
            raise ConfigValidationError(
                "domains.enabled cannot be combined with per-domain config"
            )
        return self

    def domain_config(self, name: str) -> DomainConfig | None:
        configs: dict[str, DomainConfig | None] = {
            "fall": self.fall,
            "bed_exit": self.bed_exit,
        }
        try:
            return configs[name]
        except KeyError as error:
            raise ConfigValidationError(f"unknown domain: {name}") from error

    @property
    def enabled_domains(self) -> tuple[str, ...] | None:
        if self.enabled is not None:
            return self.enabled
        configured = tuple(
            name
            for name in ("fall", "bed_exit")
            if (config := self.domain_config(name)) is not None and config.enabled
        )
        configured_fields = KNOWN_DOMAIN_NAMES.intersection(self.model_fields_set)
        return configured if configured_fields else None


__all__ = [
    "KNOWN_DOMAIN_NAMES",
    "BedExitDomainConfig",
    "DomainConfig",
    "DomainsConfig",
    "FallDomainConfig",
    "NightWindowConfig",
]
