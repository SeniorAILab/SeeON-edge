from __future__ import annotations

from collections.abc import Mapping

from contracts.worker_config import PulledNightWindow, PulledWorkerConfig
from worker.runtime.config.domain_models import KNOWN_DOMAIN_NAMES, NightWindowConfig
from worker.runtime.config.worker_models import WorkerConfig


def resolve_runtime_config(
    yaml_config: WorkerConfig,
    pulled: PulledWorkerConfig | None,
) -> Mapping[str, NightWindowConfig | None]:
    domain_names = set(KNOWN_DOMAIN_NAMES)
    if pulled is not None:
        domain_names |= set(pulled.detection_windows)
    pulled_supplied_window_info = _pulled_supplied_window_info(pulled)
    return {
        name: _resolve_domain_window(yaml_config, pulled, name, pulled_supplied_window_info)
        for name in domain_names
    }


def _pulled_supplied_window_info(pulled: PulledWorkerConfig | None) -> bool:
    if pulled is None:
        return False
    return bool(pulled.detection_windows) or pulled.night_window is not None


def _resolve_domain_window(
    yaml_config: WorkerConfig,
    pulled: PulledWorkerConfig | None,
    name: str,
    pulled_supplied_window_info: bool,
) -> NightWindowConfig | None:
    if pulled_supplied_window_info:
        pulled_window = _pulled_domain_window(pulled, name)
        if pulled_window is None:
            return None
        return NightWindowConfig(
            start=pulled_window.start,
            end=pulled_window.end,
            tz=pulled_window.tz,
        )
    return yaml_config.domains.resolved_detection_window(name)


def _pulled_domain_window(
    pulled: PulledWorkerConfig | None,
    name: str,
) -> PulledNightWindow | None:
    if pulled is None:
        return None
    window = pulled.detection_windows.get(name)
    if window is not None:
        return window
    return pulled.night_window if name == "bed_exit" else None


__all__ = ["resolve_runtime_config"]
