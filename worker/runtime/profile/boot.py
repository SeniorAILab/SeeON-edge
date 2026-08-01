from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from worker.runtime.profile.registry import (
    ML_WORKER_PROFILE_ENV,
    PROFILE_REGISTRY,
    BootDependencies,
    DecodePolicy,
    DecodeProbe,
    DevicePolicy,
    EncodePolicy,
    ProfileError,
    ProfileSpec,
    ProfileVerifyError,
    VerifyResult,
    default_decode_probe,
    default_verifiers,
)

_LEGACY_DECODE_ENVIRONMENTS = ("ML_RTSP_BACKEND", "ML_DEFAULT_DECODE_BACKEND")


@dataclass(frozen=True, slots=True)
class BootContext:
    profile: ProfileSpec
    device: DevicePolicy
    decode: DecodePolicy
    encode: EncodePolicy


def resolve_profile(
    env: Mapping[str, str],
    registry: Mapping[str, ProfileSpec] = PROFILE_REGISTRY,
) -> ProfileSpec:
    profile_name = env.get(ML_WORKER_PROFILE_ENV)
    if profile_name is None or not profile_name.strip():
        message = "ML_WORKER_PROFILE is required (no default); set cuda|mps|cpu"
        raise ProfileError(message)

    try:
        return registry[profile_name]
    except KeyError as error:
        choices = "|".join(sorted(registry))
        message = f"unknown ML_WORKER_PROFILE {profile_name!r}; set {choices}"
        raise ProfileError(message) from error


def verify_device_or_raise(spec: ProfileSpec, deps: BootDependencies) -> VerifyResult:
    try:
        verifier = deps.verifiers[spec.name]
        result = verifier()
    except (KeyError, RuntimeError) as error:
        message = f"profile {spec.name!r} device verification failed: {error}"
        raise ProfileVerifyError(message) from error

    if not result.ok:
        message = f"profile {spec.name!r} device verification failed: {result.reason}"
        raise ProfileVerifyError(message)
    return result


def preflight_decode_or_raise(spec: ProfileSpec, decode_probe: DecodeProbe) -> VerifyResult:
    try:
        result = decode_probe(spec.decode)
    except (OSError, RuntimeError) as error:
        message = f"profile {spec.name!r} decode preflight failed: {error}"
        raise ProfileVerifyError(message) from error

    if not result.ok:
        message = f"profile {spec.name!r} decode preflight failed: {result.reason}"
        raise ProfileVerifyError(message)
    return result


def reject_legacy_conflicts(spec: ProfileSpec, env: Mapping[str, str]) -> None:
    for key in _LEGACY_DECODE_ENVIRONMENTS:
        configured = env.get(key)
        if configured and configured != "auto" and configured != spec.decode:
            message = (
                f"{key}={configured!r} conflicts with profile "
                f"{spec.name!r} decode {spec.decode!r}"
            )
            raise ProfileVerifyError(message)


def resolve_boot_context(
    env: Mapping[str, str],
    deps: BootDependencies | None = None,
    decode_probe: DecodeProbe | None = None,
) -> BootContext:
    spec = resolve_profile(env)
    _ = verify_device_or_raise(spec, deps or BootDependencies(default_verifiers()))
    _ = preflight_decode_or_raise(spec, decode_probe or default_decode_probe)
    reject_legacy_conflicts(spec, env)
    return BootContext(
        profile=spec,
        device=spec.device,
        decode=spec.decode,
        encode=spec.encode,
    )


__all__ = [
    "BootContext",
    "preflight_decode_or_raise",
    "reject_legacy_conflicts",
    "resolve_boot_context",
    "resolve_profile",
    "verify_device_or_raise",
]
