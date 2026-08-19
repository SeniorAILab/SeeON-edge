from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from functools import cache
from pathlib import Path
from types import ModuleType

import pytest

from backend.app.features.evidence.explanation_manifest import (
    RuntimeManifestMissingReason,
    project_runtime_manifest,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}\Z")

_IDENTITY_TOP_KEYS = frozenset(
    {
        "manifest_schema_version",
        "build",
        "configuration",
        "modules",
    }
)
_BUILD_IDENTITY_KEYS = frozenset(
    {
        "detector_version",
        "worker_build_revision",
    }
)
_CONFIGURATION_IDENTITY_KEYS = frozenset({"config_version"})
_MODULE_IDENTITY_KEYS = frozenset(
    {
        "qualified_id",
        "version",
        "policy_schema",
    }
)
_SOURCE_DENYLIST_KEYS = (
    "polygon",
    "coordinate_space",
    "canonical_json",
)


@cache
def _runtime_manifest_fixtures() -> ModuleType:
    path = Path(__file__).with_name("test_runtime_manifest.py")
    spec = importlib.util.spec_from_file_location("_runtime_manifest_fixtures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_runtime_manifest_fixture_identity_shape() -> None:
    # Given an existing applied runtime-manifest fixture
    fixtures = _runtime_manifest_fixtures()
    manifest = fixtures._manifest()

    # When its canonical JSON is parsed
    content = json.loads(manifest.canonical_json)

    # Then the current identity envelope and SHA are present
    assert isinstance(content, dict)
    assert _IDENTITY_TOP_KEYS <= content.keys()
    assert content["manifest_schema_version"] == 1
    assert isinstance(content["build"], dict)
    assert _BUILD_IDENTITY_KEYS <= content["build"].keys()
    assert content["build"]["detector_version"] == "worker-domain-detectors-v1"
    assert _SOURCE_REVISION.fullmatch(content["build"]["worker_build_revision"])
    assert isinstance(content["configuration"], dict)
    assert _CONFIGURATION_IDENTITY_KEYS <= content["configuration"].keys()
    assert content["configuration"]["config_version"] == 42
    assert isinstance(content["modules"], list)
    modules = [module for module in content["modules"] if isinstance(module, dict)]
    assert modules
    assert all(_MODULE_IDENTITY_KEYS <= module.keys() for module in modules)
    assert {module["qualified_id"] for module in modules} >= {"fall.v1", "bed_exit.v1"}
    assert {module["policy_schema"] for module in modules} >= {
        "fall.policy.v1",
        "bed_exit.policy.v1",
    }
    assert {module["version"] for module in modules} == {1}
    assert _SHA256.fullmatch(manifest.sha256)
    assert "image_revision" not in content
    assert "image_revision" not in content["build"]
    assert "canonical_json" not in content

    # And the source still carries geometry/path-bearing keys that must not be projected
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert serialized == manifest.canonical_json
    assert all(token in serialized for token in ("bed_zone", "polygon", "coordinate_space"))
    persisted_manifest = fixtures._manifest(cameras=(fixtures._persisted_bed_camera(),))
    persisted = json.loads(persisted_manifest.canonical_json)
    assert persisted["cameras"][0]["bed_zone"]["polygon"] == [[1, 2], [1, 8], [9, 8], [9, 2]]
    assert all(key in persisted["cameras"][0]["bed_zone"] for key in _SOURCE_DENYLIST_KEYS[:2])


def _canonical(content: object) -> tuple[str, str]:
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return serialized, hashlib.sha256(serialized.encode()).hexdigest()


def test_projects_only_known_event_manifest_scalar_identities() -> None:
    # Given a current manifest and one event-selected module identity
    manifest = _runtime_manifest_fixtures()._manifest()

    # When the backend projects its runtime identity
    projection = project_runtime_manifest(
        canonical_json=manifest.canonical_json,
        runtime_manifest_sha256=manifest.sha256,
        module_qualified_id="fall.v1",
    )

    # Then only the requested flat scalar facts and typed missing reasons remain
    assert projection.as_dict() == {
        "config_version": 42,
        "config_version_missing_reason": None,
        "policy_version": "fall.policy.v1",
        "policy_version_missing_reason": None,
        "model_version": _runtime_manifest_fixtures()._FALL_ARTIFACT_DIGEST,
        "model_version_missing_reason": None,
        "detector_version": "worker-domain-detectors-v1",
        "detector_version_missing_reason": None,
        "runtime_manifest_sha256": manifest.sha256,
        "runtime_manifest_sha256_missing_reason": None,
        "worker_build_revision": "1" * 40,
        "worker_build_revision_missing_reason": None,
        "image_revision": None,
        "image_revision_missing_reason": "FIELD_UNAVAILABLE",
    }
    serialized = json.dumps(projection.as_dict(), sort_keys=True)
    assert all(key not in projection.as_dict() for key in _SOURCE_DENYLIST_KEYS)
    assert all(token not in serialized for token in ("polygon", "coordinate", "bed_zone"))


def test_geometry_and_path_bearing_source_never_crosses_projection() -> None:
    # Given valid identity fields alongside forbidden nested source sentinels
    source = {
        "manifest_schema_version": 1,
        "build": {
            "detector_version": "detector-v1",
            "worker_build_revision": "2" * 40,
            "image_revision": "7" * 40,
            "component_path": "/private/model-path-sentinel",
        },
        "configuration": {"config_version": 9},
        "modules": [
            {
                "qualified_id": "fall.v1",
                "policy_schema": "fall.policy.v1",
                "component_bindings": [
                    {
                        "component_id": "fall-classifier",
                        "kind": "model",
                        "component_path": "/private/binding-path-sentinel",
                    }
                ],
            }
        ],
        "components": [
            {
                "component_id": "fall-classifier",
                "artifact_sha256": "3" * 64,
                "weights_path": "/private/weights-path-sentinel",
            }
        ],
        "cameras": [
            {
                "bed_zone": {
                    "polygon": [[987654, 123456]],
                    "coordinate_space": "geometry-coordinate-sentinel",
                    "media_path": "/private/media-path-sentinel",
                }
            }
        ],
    }
    canonical_json, manifest_sha = _canonical(source)

    # When the manifest is projected
    result = project_runtime_manifest(
        canonical_json=canonical_json,
        runtime_manifest_sha256=manifest_sha,
        module_qualified_id="fall.v1",
    ).as_dict()

    # Then the output is a flat allowlist without any source structure or sentinel
    assert set(result) == {
        "config_version",
        "config_version_missing_reason",
        "policy_version",
        "policy_version_missing_reason",
        "model_version",
        "model_version_missing_reason",
        "detector_version",
        "detector_version_missing_reason",
        "runtime_manifest_sha256",
        "runtime_manifest_sha256_missing_reason",
        "worker_build_revision",
        "worker_build_revision_missing_reason",
        "image_revision",
        "image_revision_missing_reason",
    }
    assert all(value is None or isinstance(value, int | str) for value in result.values())
    assert result["image_revision"] == "7" * 40
    assert result["image_revision_missing_reason"] is None
    serialized = json.dumps(result, sort_keys=True)
    for sentinel in (
        "canonical_json",
        "component_path",
        "weights_path",
        "media_path",
        "polygon",
        "coordinate_space",
        "geometry-coordinate-sentinel",
        "987654",
        "123456",
        "/private/",
    ):
        assert sentinel not in serialized


@pytest.mark.parametrize(
    ("mutation", "field", "expected_reason"),
    (
        ({"configuration": {"config_version": True}}, "config_version", "FIELD_MALFORMED"),
        (
            {"build": {"detector_version": {"nested": "sentinel"}}},
            "detector_version",
            "FIELD_MALFORMED",
        ),
        ({"build": {"worker_build_revision": 7}}, "worker_build_revision", "FIELD_MALFORMED"),
        ({"module_policy_schema": ["fall.policy.v1"]}, "policy_version", "FIELD_MALFORMED"),
        ({"model_artifact_sha256": "/private/model.bin"}, "model_version", "FIELD_MALFORMED"),
    ),
)
def test_wrong_scalar_types_and_path_bearing_identity_values_are_typed_malformed(
    mutation: dict[str, object],
    field: str,
    expected_reason: str,
) -> None:
    # Given a current-shaped manifest with one malformed allowlisted field
    source = {
        "manifest_schema_version": 1,
        "build": {
            "detector_version": "detector-v1",
            "worker_build_revision": "4" * 40,
        },
        "configuration": {"config_version": 11},
        "modules": [
            {
                "qualified_id": "fall.v1",
                "policy_schema": "fall.policy.v1",
                "component_bindings": [
                    {"component_id": "fall-classifier", "kind": "model"}
                ],
            }
        ],
        "components": [
            {"component_id": "fall-classifier", "artifact_sha256": "5" * 64}
        ],
    }
    if "configuration" in mutation:
        source["configuration"] = mutation["configuration"]
    if "build" in mutation:
        source["build"].update(mutation["build"])
    if "module_policy_schema" in mutation:
        source["modules"][0]["policy_schema"] = mutation["module_policy_schema"]
    if "model_artifact_sha256" in mutation:
        source["components"][0]["artifact_sha256"] = mutation["model_artifact_sha256"]
    canonical_json, manifest_sha = _canonical(source)

    # When it is projected
    result = project_runtime_manifest(
        canonical_json=canonical_json,
        runtime_manifest_sha256=manifest_sha,
        module_qualified_id="fall.v1",
    ).as_dict()

    # Then that field is unavailable with a closed malformed reason
    assert result[field] is None
    assert result[f"{field}_missing_reason"] == expected_reason


@pytest.mark.parametrize(
    ("canonical_json", "reason"),
    (
        (None, RuntimeManifestMissingReason.MANIFEST_UNAVAILABLE.value),
        ("{not-json", RuntimeManifestMissingReason.MANIFEST_MALFORMED.value),
    ),
)
def test_unavailable_or_malformed_manifest_types_every_content_field_missing(
    canonical_json: str | None,
    reason: str,
) -> None:
    # Given unavailable or malformed canonical manifest input
    # When it is projected
    result = project_runtime_manifest(
        canonical_json=canonical_json,
        runtime_manifest_sha256="6" * 64,
        module_qualified_id="fall.v1",
    ).as_dict()

    # Then the independent row SHA remains known and every content fact is typed missing
    assert result["runtime_manifest_sha256"] == "6" * 64
    assert result["runtime_manifest_sha256_missing_reason"] is None
    for field in (
        "config_version",
        "policy_version",
        "model_version",
        "detector_version",
        "worker_build_revision",
        "image_revision",
    ):
        assert result[field] is None
        assert result[f"{field}_missing_reason"] == reason


def test_legacy_manifest_and_absent_revisions_have_typed_reasons() -> None:
    # Given a pre-versioned legacy manifest and a current manifest with absent revisions
    legacy_json, legacy_sha = _canonical({"configuration": {"config_version": 2}})
    current_json, current_sha = _canonical(
        {
            "manifest_schema_version": 1,
            "build": {"detector_version": "detector-v1"},
            "configuration": {"config_version": 2},
            "modules": [],
            "components": [],
        }
    )

    # When both are projected
    legacy = project_runtime_manifest(
        canonical_json=legacy_json,
        runtime_manifest_sha256=legacy_sha,
        module_qualified_id="fall.v1",
    ).as_dict()
    current = project_runtime_manifest(
        canonical_json=current_json,
        runtime_manifest_sha256=current_sha,
        module_qualified_id="fall.v1",
    ).as_dict()

    # Then legacy and absent fields use distinct closed reasons
    for field in (
        "config_version",
        "policy_version",
        "model_version",
        "detector_version",
        "worker_build_revision",
        "image_revision",
    ):
        assert legacy[field] is None
        assert legacy[f"{field}_missing_reason"] == "LEGACY_MANIFEST"
    assert current["worker_build_revision"] is None
    assert current["worker_build_revision_missing_reason"] == "FIELD_UNAVAILABLE"
    assert current["image_revision"] is None
    assert current["image_revision_missing_reason"] == "FIELD_UNAVAILABLE"
    assert current["policy_version_missing_reason"] == "FIELD_UNAVAILABLE"
    assert current["model_version_missing_reason"] == "FIELD_UNAVAILABLE"


def test_absent_runtime_manifest_sha_has_typed_unavailable_reason() -> None:
    # Given valid content without a persisted runtime-manifest SHA identity
    manifest = _runtime_manifest_fixtures()._manifest()

    # When it is projected
    result = project_runtime_manifest(
        canonical_json=manifest.canonical_json,
        runtime_manifest_sha256=None,
        module_qualified_id="fall.v1",
    ).as_dict()

    # Then content fields remain usable while the SHA absence is explicit
    assert result["config_version"] == 42
    assert result["runtime_manifest_sha256"] is None
    assert result["runtime_manifest_sha256_missing_reason"] == "FIELD_UNAVAILABLE"


def test_malformed_runtime_manifest_sha_is_not_echoed() -> None:
    # Given valid content with a malformed external SHA identity
    canonical_json, _ = _canonical({"manifest_schema_version": 1})

    # When it is projected
    result = project_runtime_manifest(
        canonical_json=canonical_json,
        runtime_manifest_sha256="/private/not-a-sha",
        module_qualified_id="fall.v1",
    ).as_dict()

    # Then the unsafe value is not returned
    assert result["runtime_manifest_sha256"] is None
    assert result["runtime_manifest_sha256_missing_reason"] == "FIELD_MALFORMED"
    assert "/private/" not in json.dumps(result, sort_keys=True)
