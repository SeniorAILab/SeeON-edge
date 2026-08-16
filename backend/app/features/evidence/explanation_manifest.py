"""Privacy-bounded scalar projection of persisted runtime-manifest identities."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import StrEnum

_CURRENT_MANIFEST_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_MISSING = object()


class RuntimeManifestMissingReason(StrEnum):
    LEGACY_MANIFEST = "LEGACY_MANIFEST"
    MANIFEST_UNAVAILABLE = "MANIFEST_UNAVAILABLE"
    MANIFEST_MALFORMED = "MANIFEST_MALFORMED"
    FIELD_UNAVAILABLE = "FIELD_UNAVAILABLE"
    FIELD_MALFORMED = "FIELD_MALFORMED"


@dataclass(frozen=True, slots=True)
class RuntimeManifestProjection:
    config_version: int | None
    config_version_missing_reason: RuntimeManifestMissingReason | None
    policy_version: str | None
    policy_version_missing_reason: RuntimeManifestMissingReason | None
    model_version: str | None
    model_version_missing_reason: RuntimeManifestMissingReason | None
    detector_version: str | None
    detector_version_missing_reason: RuntimeManifestMissingReason | None
    runtime_manifest_sha256: str | None
    runtime_manifest_sha256_missing_reason: RuntimeManifestMissingReason | None
    worker_build_revision: str | None
    worker_build_revision_missing_reason: RuntimeManifestMissingReason | None
    image_revision: str | None
    image_revision_missing_reason: RuntimeManifestMissingReason | None

    def __post_init__(self) -> None:
        for name in (
            "config_version",
            "policy_version",
            "model_version",
            "detector_version",
            "runtime_manifest_sha256",
            "worker_build_revision",
            "image_revision",
        ):
            value = getattr(self, name)
            reason = getattr(self, f"{name}_missing_reason")
            if (value is None) == (reason is None):
                raise ValueError(f"{name} must have exactly one of value or missing reason")

    def as_dict(self) -> dict[str, int | str | None]:
        """Return the complete flat wire-ready allowlist; never return source objects."""
        result: dict[str, int | str | None] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            result[field.name] = value.value if isinstance(value, StrEnum) else value
        return result


def project_runtime_manifest(
    *,
    canonical_json: str | None,
    runtime_manifest_sha256: str | None,
    module_qualified_id: str | None,
) -> RuntimeManifestProjection:
    """Project event-relevant scalar identities without exposing manifest structure."""
    manifest_sha, manifest_sha_reason = _sha256_fact(runtime_manifest_sha256)
    unavailable = _manifest_unavailability(canonical_json, manifest_sha)
    if unavailable is not None:
        return _missing_projection(
            unavailable,
            manifest_sha=manifest_sha,
            manifest_sha_reason=manifest_sha_reason,
        )

    if not isinstance(canonical_json, str):
        return _missing_projection(
            RuntimeManifestMissingReason.MANIFEST_MALFORMED,
            manifest_sha=manifest_sha,
            manifest_sha_reason=manifest_sha_reason,
        )

    content = _parse_current_manifest(canonical_json)
    if isinstance(content, RuntimeManifestMissingReason):
        return _missing_projection(
            content,
            manifest_sha=manifest_sha,
            manifest_sha_reason=manifest_sha_reason,
        )

    config_version, config_reason = _config_version(content)
    detector_version, detector_reason = _build_text(content, "detector_version")
    worker_revision, worker_reason = _build_revision(content, "worker_build_revision")
    image_revision, image_reason = _build_revision(content, "image_revision")
    module, module_reason = _selected_module(content, module_qualified_id)
    if module is None:
        assert module_reason is not None
        policy_version, policy_reason = None, module_reason
        model_version, model_reason = None, module_reason
    else:
        policy_version, policy_reason = _policy_version(module)
        model_version, model_reason = _model_version(content, module)

    return RuntimeManifestProjection(
        config_version=config_version,
        config_version_missing_reason=config_reason,
        policy_version=policy_version,
        policy_version_missing_reason=policy_reason,
        model_version=model_version,
        model_version_missing_reason=model_reason,
        detector_version=detector_version,
        detector_version_missing_reason=detector_reason,
        runtime_manifest_sha256=manifest_sha,
        runtime_manifest_sha256_missing_reason=manifest_sha_reason,
        worker_build_revision=worker_revision,
        worker_build_revision_missing_reason=worker_reason,
        image_revision=image_revision,
        image_revision_missing_reason=image_reason,
    )


def _manifest_unavailability(
    canonical_json: object,
    manifest_sha: str | None,
) -> RuntimeManifestMissingReason | None:
    if canonical_json is None:
        return RuntimeManifestMissingReason.MANIFEST_UNAVAILABLE
    if not isinstance(canonical_json, str):
        return RuntimeManifestMissingReason.MANIFEST_MALFORMED
    if manifest_sha is not None:
        actual_sha = hashlib.sha256(canonical_json.encode()).hexdigest()
        if actual_sha != manifest_sha:
            return RuntimeManifestMissingReason.MANIFEST_MALFORMED
    return None


def _parse_current_manifest(
    canonical_json: str,
) -> Mapping[str, object] | RuntimeManifestMissingReason:
    try:
        content = json.loads(canonical_json)
    except json.JSONDecodeError:
        return RuntimeManifestMissingReason.MANIFEST_MALFORMED
    if not isinstance(content, dict):
        return RuntimeManifestMissingReason.MANIFEST_MALFORMED
    normalized = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    if normalized != canonical_json:
        return RuntimeManifestMissingReason.MANIFEST_MALFORMED
    schema_version = content.get("manifest_schema_version", _MISSING)
    if schema_version is _MISSING or (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version < _CURRENT_MANIFEST_SCHEMA_VERSION
    ):
        return RuntimeManifestMissingReason.LEGACY_MANIFEST
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        return RuntimeManifestMissingReason.MANIFEST_MALFORMED
    if schema_version != _CURRENT_MANIFEST_SCHEMA_VERSION:
        return RuntimeManifestMissingReason.MANIFEST_UNAVAILABLE
    return content


def _config_version(
    content: Mapping[str, object],
) -> tuple[int | None, RuntimeManifestMissingReason | None]:
    configuration = content.get("configuration", _MISSING)
    if configuration is _MISSING:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    if not isinstance(configuration, Mapping):
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    value = configuration.get("config_version", _MISSING)
    if value is _MISSING or value is None:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    return value, None


def _build_text(
    content: Mapping[str, object],
    field_name: str,
) -> tuple[str | None, RuntimeManifestMissingReason | None]:
    value, reason = _build_member(content, field_name)
    if reason is not None:
        return None, reason
    if not isinstance(value, str) or not _safe_identity_text(value):
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    return value, None


def _build_revision(
    content: Mapping[str, object],
    field_name: str,
) -> tuple[str | None, RuntimeManifestMissingReason | None]:
    value, reason = _build_member(content, field_name)
    if reason is not None:
        return None, reason
    if (
        not isinstance(value, str)
        or value == "0" * 40
        or _SOURCE_REVISION.fullmatch(value) is None
    ):
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    return value, None


def _build_member(
    content: Mapping[str, object],
    field_name: str,
) -> tuple[object | None, RuntimeManifestMissingReason | None]:
    build = content.get("build", _MISSING)
    if build is _MISSING:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    if not isinstance(build, Mapping):
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    value = build.get(field_name, _MISSING)
    if value is _MISSING or value is None:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    return value, None


def _selected_module(
    content: Mapping[str, object],
    module_qualified_id: object,
) -> tuple[Mapping[str, object] | None, RuntimeManifestMissingReason | None]:
    if module_qualified_id is None:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    if not isinstance(module_qualified_id, str) or not _safe_identity_text(module_qualified_id):
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    modules = content.get("modules", _MISSING)
    if modules is _MISSING:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    if not isinstance(modules, list):
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    matches: list[Mapping[str, object]] = []
    for module in modules:
        if not isinstance(module, Mapping):
            return None, RuntimeManifestMissingReason.FIELD_MALFORMED
        qualified_id = module.get("qualified_id", _MISSING)
        if qualified_id is _MISSING:
            continue
        if not isinstance(qualified_id, str) or not _safe_identity_text(qualified_id):
            return None, RuntimeManifestMissingReason.FIELD_MALFORMED
        if qualified_id == module_qualified_id:
            matches.append(module)
    if not matches:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    if len(matches) != 1:
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    return matches[0], None


def _policy_version(
    module: Mapping[str, object],
) -> tuple[str | None, RuntimeManifestMissingReason | None]:
    value = module.get("policy_schema", _MISSING)
    if value is _MISSING or value is None:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    if not isinstance(value, str) or not _safe_identity_text(value):
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    return value, None


def _model_version(
    content: Mapping[str, object],
    module: Mapping[str, object],
) -> tuple[str | None, RuntimeManifestMissingReason | None]:
    bindings = module.get("component_bindings", _MISSING)
    if bindings is _MISSING:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    if not isinstance(bindings, list):
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    model_component_ids: list[str] = []
    for binding in bindings:
        if not isinstance(binding, Mapping):
            return None, RuntimeManifestMissingReason.FIELD_MALFORMED
        kind = binding.get("kind", _MISSING)
        if kind is _MISSING:
            continue
        if not isinstance(kind, str):
            return None, RuntimeManifestMissingReason.FIELD_MALFORMED
        if kind != "model":
            continue
        component_id = binding.get("component_id", _MISSING)
        if not isinstance(component_id, str) or not _safe_identity_text(component_id):
            return None, RuntimeManifestMissingReason.FIELD_MALFORMED
        model_component_ids.append(component_id)
    if not model_component_ids:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    if len(model_component_ids) != 1:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE

    components = content.get("components", _MISSING)
    if components is _MISSING:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    if not isinstance(components, list):
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    matches: list[Mapping[str, object]] = []
    for component in components:
        if not isinstance(component, Mapping):
            return None, RuntimeManifestMissingReason.FIELD_MALFORMED
        component_id = component.get("component_id", _MISSING)
        if component_id == model_component_ids[0]:
            matches.append(component)
    if not matches:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    if len(matches) != 1:
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    value = matches[0].get("artifact_sha256", _MISSING)
    if value is _MISSING or value is None:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    return value, None


def _sha256_fact(
    value: object,
) -> tuple[str | None, RuntimeManifestMissingReason | None]:
    if value is None:
        return None, RuntimeManifestMissingReason.FIELD_UNAVAILABLE
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        return None, RuntimeManifestMissingReason.FIELD_MALFORMED
    return value, None


def _safe_identity_text(value: str) -> bool:
    return bool(value) and not any(
        token in value
        for token in (
            "://",
            "/",
            "\\",
            "\x00",
            "\n",
            "\r",
        )
    )


def _missing_projection(
    reason: RuntimeManifestMissingReason,
    *,
    manifest_sha: str | None,
    manifest_sha_reason: RuntimeManifestMissingReason | None,
) -> RuntimeManifestProjection:
    return RuntimeManifestProjection(
        config_version=None,
        config_version_missing_reason=reason,
        policy_version=None,
        policy_version_missing_reason=reason,
        model_version=None,
        model_version_missing_reason=reason,
        detector_version=None,
        detector_version_missing_reason=reason,
        runtime_manifest_sha256=manifest_sha,
        runtime_manifest_sha256_missing_reason=manifest_sha_reason,
        worker_build_revision=None,
        worker_build_revision_missing_reason=reason,
        image_revision=None,
        image_revision_missing_reason=reason,
    )


__all__ = [
    "RuntimeManifestMissingReason",
    "RuntimeManifestProjection",
    "project_runtime_manifest",
]
