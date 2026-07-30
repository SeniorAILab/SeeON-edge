from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, field_validator

from contracts.worker_config import PulledWorkerConfig
from edge.domains.base import AuditContext, DomainDetector
from edge.domains.bed_exit.detector import BedExitMonitor, NightWindow
from edge.domains.bed_exit.schema import BedExitDebugSnapshot, DomainDebugSnapshot
from edge.domains.fall.detector import FallEventLatch
from edge.perception.domain_input import DomainInput
from edge.perception.fall_window_classifier import FallModelProtocol, FallWindowClassifier

DebugSnapshotAdapter = Callable[[DomainDetector, int], DomainDebugSnapshot | None]
Factory = Callable[[object | None, FallModelProtocol | None], DomainDetector]
InputPreparer = Callable[[DomainDetector, str, DomainInput], DomainInput]
RuntimeConfigResolver = Callable[[BaseModel | None, PulledWorkerConfig | None], object | None]
RuntimeConfigUpdater = Callable[[DomainDetector, object | None], None]
AuditMetadataProvider = Callable[[DomainDetector, AuditContext | None], dict[str, object]]


@dataclass(frozen=True, slots=True)
class DomainRegistration:
    name: str
    factory: Factory
    enabled: bool
    input_view: str
    event_types: frozenset[str]
    debug_snapshot_adapter: DebugSnapshotAdapter | None = None
    audit_event_types: frozenset[str] = frozenset()
    audit_metadata_provider: AuditMetadataProvider | None = None
    input_preparer: InputPreparer = lambda _detector, _view, domain_input: domain_input
    config_schema: type[BaseModel] | None = None
    runtime_config_resolver: RuntimeConfigResolver | None = None
    runtime_config_updater: RuntimeConfigUpdater | None = None
    model_name: str | None = None
class _FallConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True


class _NightScheduleConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: str
    end: str
    tz: str

    @field_validator("start", "end")
    @classmethod
    def _validate_time(cls, value: str) -> str:
        hour, separator, minute = value.partition(":")
        if (
            separator != ":"
            or not hour.isdigit()
            or not minute.isdigit()
            or len(hour) != 2
            or len(minute) != 2
            or not (0 <= int(hour) <= 23 and 0 <= int(minute) <= 59)
        ):
            raise ValueError("night window time must use HH:MM")
        return value

    @field_validator("tz")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError("night window tz must be a valid IANA timezone") from exc
        return value


class _BedExitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    night_window: _NightScheduleConfig | None = None


def _bed_exit_runtime_config(
    config: BaseModel | None, pulled: PulledWorkerConfig | None
) -> NightWindow | None:
    window = pulled.night_window if pulled is not None else None
    if window is None and config is not None:
        window = getattr(config, "night_window", None)
    if window is None:
        return None
    return NightWindow(start=window.start, end=window.end, tz=window.tz)


def _update_bed_exit_runtime_config(detector: DomainDetector, value: object | None) -> None:
    if isinstance(detector, BedExitMonitor):
        detector.update_night_window(value if isinstance(value, NightWindow) else None)


def _fall_factory(_config: object | None, model: FallModelProtocol | None) -> DomainDetector:
    classifier = FallWindowClassifier(model) if model is not None else None
    return FallEventLatch(
        model,
        prepare_input=None if classifier is None else classifier.classify_input,
    )


def _prepare_fall_input(
    detector: DomainDetector, input_view: str, domain_input: DomainInput
) -> DomainInput:
    if input_view != "fall_window":
        raise ValueError(f"unsupported fall input view: {input_view}")
    if not isinstance(detector, FallEventLatch):
        return domain_input
    return replace(domain_input, observation=detector.prepare_input(domain_input))


def _bed_exit_factory(config: object | None, _model: FallModelProtocol | None) -> DomainDetector:
    night_window = None
    if isinstance(config, dict):
        raw_night_window = config.get("night_window")
        if isinstance(raw_night_window, dict):
            night_window = NightWindow(
                start=str(raw_night_window["start"]),
                end=str(raw_night_window["end"]),
                tz=str(raw_night_window["tz"]),
            )
    return BedExitMonitor(night_window=night_window)


def _bed_exit_debug(detector: DomainDetector, frame_index: int) -> DomainDebugSnapshot | None:
    snapshot = getattr(detector, "last_debug_snapshot", None)
    if not isinstance(snapshot, BedExitDebugSnapshot):
        return None
    return DomainDebugSnapshot(
        domain="bed_exit",
        bed_exit=replace(snapshot, frame_index=frame_index),
    )


def _model_audit_metadata(
    _detector: DomainDetector, audit_context: AuditContext | None
) -> dict[str, object]:
    if audit_context is None:
        return {}
    return {
        "model_version": audit_context.model_version,
        "operating_threshold": audit_context.operating_threshold,
    }


DOMAIN_REGISTRY: dict[str, DomainRegistration] = {
    "fall": DomainRegistration(
        "fall",
        _fall_factory,
        FallEventLatch.enabled,
        "fall_window",
        frozenset({"fall"}),
        audit_event_types=frozenset({"fall"}),
        audit_metadata_provider=_model_audit_metadata,
        input_preparer=_prepare_fall_input,
        config_schema=_FallConfig,
        model_name="fall",
    ),
    "bed_exit": DomainRegistration(
        "bed_exit",
        _bed_exit_factory,
        BedExitMonitor.enabled,
        "bed_regions",
        frozenset({"bed-exit"}),
        _bed_exit_debug,
        frozenset({"bed-exit"}),
        audit_metadata_provider=_model_audit_metadata,
        config_schema=_BedExitConfig,
        runtime_config_resolver=_bed_exit_runtime_config,
        runtime_config_updater=_update_bed_exit_runtime_config,
    ),
}


def list_domains(*, enabled: bool | None = None) -> tuple[str, ...]:
    if enabled is None:
        return tuple(DOMAIN_REGISTRY)
    return tuple(
        name for name, registration in DOMAIN_REGISTRY.items() if registration.enabled is enabled
    )


def enabled_domains() -> tuple[str, ...]:
    return list_domains(enabled=True)


__all__ = ["DOMAIN_REGISTRY", "DomainRegistration", "enabled_domains", "list_domains"]
