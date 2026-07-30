"""Boot-time device profile resolution and verification."""

from edge.runtime.profile.boot import (
    BootContext,
    preflight_decode_or_raise,
    profile_verify_stage,
    reject_legacy_conflicts,
    resolve_boot_context,
    resolve_profile,
    verify_device_or_raise,
)
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

__all__ = [
    "BootContext",
    "BootDependencies",
    "ML_WORKER_PROFILE_ENV",
    "PROFILE_REGISTRY",
    "ProfileError",
    "ProfileSpec",
    "ProfileVerifyError",
    "VerifyResult",
    "default_decode_probe",
    "default_verifiers",
    "preflight_decode_or_raise",
    "profile_verify_stage",
    "reject_legacy_conflicts",
    "resolve_boot_context",
    "resolve_profile",
    "verify_device_or_raise",
]
