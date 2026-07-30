"""Fail-fast boot resolution for the worker device profile."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from edge.runtime.profile.registry import (
    ML_WORKER_PROFILE_ENV,
    PROFILE_REGISTRY,
    BootDependencies,
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
    device: str
    decode: str


def resolve_profile(
    env: Mapping[str, str], registry: Mapping[str, ProfileSpec] = PROFILE_REGISTRY
) -> ProfileSpec:
    profile_name = env.get(ML_WORKER_PROFILE_ENV)
    if profile_name is None or not profile_name.strip():
        raise ProfileError("ML_WORKER_PROFILE is required (no default); set cuda|mps|cpu")
    try:
        return registry[profile_name]
    except KeyError as exc:
        choices = "|".join(sorted(registry))
        raise ProfileError(f"unknown ML_WORKER_PROFILE {profile_name!r}; set {choices}") from exc


def verify_device_or_raise(spec: ProfileSpec, deps: BootDependencies) -> VerifyResult:
    try:
        result = deps.verifiers[spec.name]()
    except Exception as exc:  # noqa: BLE001 - profile verification is a global fail-fast boundary
        raise ProfileVerifyError(
            f"profile {spec.name!r} device verification failed: {exc}"
        ) from exc
    if not result.ok:
        raise ProfileVerifyError(
            f"profile {spec.name!r} device verification failed: {result.reason}"
        )
    return result


def preflight_decode_or_raise(
    spec: ProfileSpec, decode_probe: Callable[[str], VerifyResult]
) -> VerifyResult:
    try:
        result = decode_probe(spec.decode)
    except Exception as exc:  # noqa: BLE001 - decode verification is a global fail-fast boundary
        raise ProfileVerifyError(f"profile {spec.name!r} decode preflight failed: {exc}") from exc
    if not result.ok:
        raise ProfileVerifyError(f"profile {spec.name!r} decode preflight failed: {result.reason}")
    return result


def reject_legacy_conflicts(spec: ProfileSpec, env: Mapping[str, str]) -> None:
    for key in _LEGACY_DECODE_ENVIRONMENTS:
        configured = env.get(key)
        if configured and configured != "auto" and configured != spec.decode:
            raise ProfileVerifyError(
                f"{key}={configured!r} conflicts with profile {spec.name!r} decode {spec.decode!r}"
            )


def resolve_boot_context(
    env: Mapping[str, str],
    deps: BootDependencies | None = None,
    decode_probe: Callable[[str], VerifyResult] | None = None,
) -> BootContext:
    resolved_deps = deps or BootDependencies(default_verifiers())
    resolved_decode_probe = decode_probe or default_decode_probe
    spec = resolve_profile(env)
    verify_device_or_raise(spec, resolved_deps)
    preflight_decode_or_raise(spec, resolved_decode_probe)
    reject_legacy_conflicts(spec, env)
    return BootContext(spec, spec.device, spec.decode)


def profile_verify_stage(
    env: Mapping[str, str],
    deps: BootDependencies | None = None,
    decode_probe: Callable[[str], VerifyResult] | None = None,
):
    from edge.runtime.pipeline_bootstrap import Stage

    return Stage(
        name="profile_verify",
        run=lambda: resolve_boot_context(env, deps, decode_probe),
    )
