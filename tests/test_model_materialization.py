"""Subprocess contract for the pre-boot model materialization hash gate.

The scripts consume an operator-private, hash-only receipt and a mocked
Docker CLI adapter. Fixtures are synthetic bytes only: no production
weights, credentials, RTSP, or host paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZE = REPO_ROOT / "scripts" / "ops" / "materialize-model-artifacts.sh"
VERIFY = REPO_ROOT / "scripts" / "ops" / "verify-model-artifacts.sh"
FETCH_MODELS = REPO_ROOT / "scripts" / "fetch-models.sh"
PUBLISH_RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "edge-image-publish.md"

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
REVISION = "c" * 40
SOURCE_ROOT = "/app/models"

ARTIFACT_PATHS = (
    "syn/fall.pt",
    "syn/pose.pt",
    "syn/person.pt",
    "syn/bed.pt",
    "syn/upstream.json",
)
SIDECAR_PATHS = (
    "fall/lstm/arch.json",
    "fall/lstm/metadata.yaml",
)
FORBIDDEN_OUTPUT = (
    "rtsp://",
    "password",
    "PASSWORD",
    "token=",
    "Authorization",
    "/var/run/docker.sock",
    "--privileged",
    "AWS_",
    "HF_TOKEN",
)
_SUCCESS = re.compile(
    r"^MODEL_MATERIALIZATION_OK count=(\d+) sidecar_count=(\d+) dest_sha256=([0-9a-f]{64})$",
    re.MULTILINE,
)
_VERIFY_OK = re.compile(
    r"^MODEL_VERIFY_OK count=(\d+) sidecar_count=(\d+) dest_sha256=([0-9a-f]{64})$",
    re.MULTILINE,
)
_FAIL = re.compile(r"^MODEL_MATERIALIZATION_FAIL reason=([a-z0-9-]+)$", re.MULTILINE)
_VERIFY_FAIL = re.compile(r"^MODEL_VERIFY_FAIL reason=([a-z0-9-]+)$", re.MULTILINE)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def dest_receipt_sha256(path_to_digest: Mapping[str, str]) -> str:
    payload = "".join(f"{path}\t{digest}\n" for path, digest in sorted(path_to_digest.items()))
    return _sha256_text(payload)


def _payload(label: str) -> bytes:
    return f"synthetic-{label}\n".encode("ascii")


def _forbidden_source_locator() -> str:
    scheme = "rtsp"
    userinfo = "operator:secret"
    host = "192.0.2.10"
    return f"{scheme}://{userinfo}@{host}/live"


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _init_checkout(root: Path, sidecars: Mapping[str, bytes]) -> Path:
    checkout = root / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "model-gate@example.test"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Model Gate"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    for relative, payload in sidecars.items():
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    subprocess.run(["git", "add", "--", *sidecars], cwd=checkout, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "synthetic sidecars"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    return checkout


def _write_source_tree(root: Path, artifacts: Mapping[str, bytes]) -> Path:
    source = root / "source-tree"
    source.mkdir()
    for relative, payload in artifacts.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return source


def _receipt(
    *,
    artifacts: Mapping[str, bytes],
    sidecars: Mapping[str, bytes],
    digest: str = DIGEST_A,
    revision: str = REVISION,
    extra: dict[str, Any] | None = None,
    artifact_entries: list[dict[str, str]] | None = None,
    sidecar_entries: list[dict[str, str]] | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schemaVersion": 1,
        "source": source
        or {
            "kind": "docker-cli",
            "container": "synthetic-worker",
            "imageDigest": digest,
            "revision": revision,
            "root": SOURCE_ROOT,
        },
        "artifacts": artifact_entries
        or [
            {
                "path": path,
                "sha256": _sha256_bytes(payload),
                "class": "weight" if path.endswith(".pt") else "provenance",
            }
            for path, payload in artifacts.items()
        ],
        "sidecars": sidecar_entries
        or [
            {"path": path, "sha256": _sha256_bytes(payload), "class": "sidecar"}
            for path, payload in sidecars.items()
        ],
    }
    if extra:
        body.update(extra)
    return body


def _write_receipt(path: Path, body: dict[str, Any]) -> Path:
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _docker_stub(
    path: Path,
    *,
    source_tree: Path,
    inspect_stdout: str,
    inspect_status: int = 0,
    cp_status: int = 0,
    fail_cp_on: str | None = None,
    write_cp: bool = True,
) -> Path:
    script = f"""#!/usr/bin/env python3
import shutil
import sys
from pathlib import Path

argv_log = Path({str(path / "docker.argv")!r})
with argv_log.open("a", encoding="utf-8") as handle:
    handle.write("\\0".join(sys.argv[1:]) + "\\n")

source_tree = Path({str(source_tree)!r})
source_root = {SOURCE_ROOT!r}

if len(sys.argv) > 1 and sys.argv[1] == "inspect":
    sys.stdout.write({inspect_stdout!r})
    if not {inspect_stdout!r}.endswith("\\n"):
        sys.stdout.write("\\n")
    raise SystemExit({inspect_status})

if len(sys.argv) > 1 and sys.argv[1] == "cp":
    spec = sys.argv[2]
    dest = Path(sys.argv[3])
    container, _, container_path = spec.partition(":")
    if not container or not container_path.startswith(source_root + "/"):
        sys.stderr.write("docker-stub: unexpected cp spec\\n")
        raise SystemExit(2)
    relative = container_path[len(source_root) + 1 :]
    if {fail_cp_on!r} and relative == {fail_cp_on!r}:
        raise SystemExit({cp_status if cp_status != 0 else 1})
    if {write_cp!r}:
        src = source_tree / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    raise SystemExit({cp_status})

sys.stderr.write("docker-stub: unexpected command\\n")
raise SystemExit(2)
"""
    return _write_executable(path / "docker", script)


def _env(bin_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["SEEON_MODEL_DOCKER"] = str(bin_dir / "docker")
    return env


def _run(
    script: Path,
    tmp_path: Path,
    *,
    receipt: Path,
    dest: Path,
    checkout: Path,
    extra_env: Mapping[str, str] | None = None,
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    env = _env(tmp_path / "bin")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [
            "bash",
            str(script),
            "--receipt",
            str(receipt),
            "--dest",
            str(dest),
            "--checkout",
            str(checkout),
            *extra_args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _happy_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, bytes], dict[str, bytes]]:
    artifacts = {path: _payload(Path(path).stem) for path in ARTIFACT_PATHS}
    sidecars = {path: _payload(Path(path).name) for path in SIDECAR_PATHS}
    checkout = _init_checkout(tmp_path, sidecars)
    source = _write_source_tree(tmp_path, artifacts)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _docker_stub(
        bin_dir,
        source_tree=source,
        inspect_stdout=f"sha256:{DIGEST_A} {REVISION}",
    )
    dest = tmp_path / "dest"
    dest.mkdir()
    receipt = _write_receipt(
        tmp_path / "receipt.json",
        _receipt(artifacts=artifacts, sidecars=sidecars),
    )
    return receipt, dest, checkout, artifacts, sidecars


def _combined(artifacts: Mapping[str, bytes], sidecars: Mapping[str, bytes]) -> dict[str, str]:
    merged = {path: _sha256_bytes(payload) for path, payload in artifacts.items()}
    merged.update({path: _sha256_bytes(payload) for path, payload in sidecars.items()})
    return merged


def _assert_redacted(result: subprocess.CompletedProcess[str], *extra: str) -> None:
    blob = f"{result.stdout}\n{result.stderr}"
    for needle in (*FORBIDDEN_OUTPUT, *extra):
        assert needle not in blob, f"unredacted {needle!r} in script output"


def test_materialize_scripts_are_executable_pre_boot_interfaces() -> None:
    for script in (MATERIALIZE, VERIFY):
        assert script.is_file(), f"missing {script.relative_to(REPO_ROOT)}"
        assert script.stat().st_mode & stat.S_IXUSR
        shebang = script.read_text(encoding="utf-8").splitlines()[0]
        assert shebang == "#!/bin/bash"


def test_publish_runbook_and_fetch_script_cite_the_hash_gate() -> None:
    runbook = PUBLISH_RUNBOOK.read_text(encoding="utf-8")
    fetch = FETCH_MODELS.read_text(encoding="utf-8")
    assert "scripts/ops/materialize-model-artifacts.sh" in runbook
    assert "scripts/ops/verify-model-artifacts.sh" in runbook
    assert "scripts/ops/materialize-model-artifacts.sh" in fetch
    assert "scripts/ops/verify-model-artifacts.sh" in fetch
    assert "huggingface.co" in fetch
    assert "docker cp" not in fetch


def test_materialize_happy_path_copies_declared_set_and_emits_hash_verdict(
    tmp_path: Path,
) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode == 0, result.stderr
    match = _SUCCESS.search(result.stdout)
    assert match is not None, result.stdout
    assert match.group(1) == str(len(artifacts))
    assert match.group(2) == str(len(sidecars))
    expected = dest_receipt_sha256(_combined(artifacts, sidecars))
    assert match.group(3) == expected
    _assert_redacted(result, "synthetic-fall", "synthetic-arch.json", str(dest), "synthetic-worker")
    for path, payload in {**artifacts, **sidecars}.items():
        assert (dest / path).read_bytes() == payload
    extras = [path for path in dest.rglob("*") if path.is_file()]
    assert len(extras) == len(artifacts) + len(sidecars)
    argv = (tmp_path / "bin" / "docker.argv").read_text(encoding="utf-8")
    inspect_line = argv.splitlines()[0]
    assert inspect_line.startswith("inspect")
    assert "--format" in inspect_line
    assert "{{json ." not in inspect_line
    assert "Config.Env" not in inspect_line
    assert all(line.startswith("cp\0") for line in argv.splitlines()[1:])
    assert "/var/run/docker.sock" not in argv
    assert "--privileged" not in argv


def test_verify_happy_path_matches_materialize_dest_hash(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    materialized = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)
    assert materialized.returncode == 0, materialized.stderr

    result = _run(VERIFY, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode == 0, result.stderr
    match = _VERIFY_OK.search(result.stdout)
    assert match is not None, result.stdout
    assert match.group(3) == dest_receipt_sha256(_combined(artifacts, sidecars))
    _assert_redacted(result, "synthetic-fall")


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda body: body.pop("schemaVersion"), "malformed-receipt"),
        (lambda body: body.__setitem__("schemaVersion", 99), "malformed-receipt"),
        (lambda body: body["source"].__setitem__("kind", "tar"), "malformed-receipt"),
        (lambda body: body["source"].pop("imageDigest"), "malformed-receipt"),
        (
            lambda body: body["artifacts"].__setitem__(
                0, {**body["artifacts"][0], "sha256": "zz"}
            ),
            "malformed-receipt",
        ),
        (
            lambda body: body.__setitem__("rtsp", _forbidden_source_locator()),
            "malformed-receipt",
        ),
        (lambda body: body.__setitem__("password", "hunter2"), "malformed-receipt"),
        (
            lambda body: body["artifacts"].__setitem__(
                0, {**body["artifacts"][0], "bytes": "AAAA"}
            ),
            "malformed-receipt",
        ),
        (lambda body: body.__setitem__("artifacts", []), "malformed-receipt"),
    ],
)
def test_materialize_rejects_malformed_or_secret_bearing_receipt(
    tmp_path: Path, mutate: Any, reason: str
) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    body = json.loads(receipt.read_text(encoding="utf-8"))
    mutate(body)
    _write_receipt(receipt, body)

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None, result.stdout + result.stderr
    assert match.group(1) == reason
    _assert_redacted(result, "hunter2", _forbidden_source_locator())
    assert list(dest.rglob("*")) == []


def test_materialize_rejects_path_traversal(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    outside = tmp_path / "outside.txt"
    body = json.loads(receipt.read_text(encoding="utf-8"))
    body["artifacts"][0]["path"] = "../outside.txt"
    _write_receipt(receipt, body)

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "path-traversal"
    assert not outside.exists()
    assert list(dest.rglob("*")) == []


def test_materialize_fails_on_wrong_image_before_any_copy(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    _docker_stub(
        tmp_path / "bin",
        source_tree=tmp_path / "source-tree",
        inspect_stdout=f"sha256:{DIGEST_B} {REVISION}",
    )

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "wrong-image"
    argv = (tmp_path / "bin" / "docker.argv").read_text(encoding="utf-8")
    assert "cp\0" not in argv
    assert list(dest.rglob("*")) == []


def test_materialize_rejects_misleading_docker_inspect_success(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    _docker_stub(
        tmp_path / "bin",
        source_tree=tmp_path / "source-tree",
        inspect_stdout=f"OK Image=sha256:{DIGEST_A} revision={REVISION} success=true",
    )

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "source-identity"
    argv = (tmp_path / "bin" / "docker.argv").read_text(encoding="utf-8")
    assert "cp\0" not in argv


def test_materialize_fails_closed_on_missing_declared_artifact(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    missing = ARTIFACT_PATHS[0]
    (tmp_path / "source-tree" / missing).unlink()

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "missing-artifact"
    assert not (dest / missing).exists()


def test_materialize_fails_on_changed_destination_byte(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    target = ARTIFACT_PATHS[1]
    (tmp_path / "source-tree" / target).write_bytes(b"synthetic-altered\n")

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "altered-artifact"
    _assert_redacted(result, "synthetic-altered")


def test_verify_fails_on_extra_destination_file(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    materialized = _run(
        MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout
    )
    assert materialized.returncode == 0
    extra = dest / "syn" / "stale.pt"
    extra.write_bytes(b"stale-destination\n")

    result = _run(VERIFY, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    match = _VERIFY_FAIL.search(result.stdout) or _VERIFY_FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "extra-artifact"
    _assert_redacted(result, "stale-destination")


def test_materialize_fails_on_missing_tracked_sidecar(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    sidecar = checkout / SIDECAR_PATHS[0]
    sidecar.unlink()
    subprocess.run(
        ["git", "add", "-u", "--", SIDECAR_PATHS[0]],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "drop sidecar"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "missing-sidecar"


def test_materialize_fails_on_dirty_checkout_sidecar(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    (checkout / SIDECAR_PATHS[1]).write_bytes(b"dirty-sidecar\n")

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "dirty-checkout"
    _assert_redacted(result, "dirty-sidecar")


def test_interrupted_copy_does_not_publish_partial_destination(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    fail_on = ARTIFACT_PATHS[2]
    _docker_stub(
        tmp_path / "bin",
        source_tree=tmp_path / "source-tree",
        inspect_stdout=f"sha256:{DIGEST_A} {REVISION}",
        fail_cp_on=fail_on,
        cp_status=1,
    )
    stale = dest / "pre-existing.txt"
    stale.write_text("keep-me\n", encoding="utf-8")

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None
    assert match.group(1) == "copy-failed"
    assert stale.read_text(encoding="utf-8") == "keep-me\n"
    published = [path.relative_to(dest).as_posix() for path in dest.rglob("*") if path.is_file()]
    assert published == ["pre-existing.txt"]
    assert not (tmp_path / "dest.partial").exists()
    leftover_staging = [
        path
        for path in tmp_path.glob("**/*materialize*")
        if path.is_dir() and path.name.startswith(".")
    ]
    assert leftover_staging == []


def test_materialize_does_not_invoke_compose(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    compose = _write_executable(
        tmp_path / "bin" / "docker-compose",
        "#!/bin/sh\necho compose-must-not-run >&2\nexit 0\n",
    )
    _write_executable(
        tmp_path / "bin" / "compose",
        "#!/bin/sh\necho compose-must-not-run >&2\nexit 0\n",
    )

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode == 0, result.stderr
    assert "compose-must-not-run" not in result.stderr
    assert compose.is_file()


def _declared_dest_files(dest: Path) -> list[str]:
    entries: list[str] = []
    for dirpath, dirnames, filenames in os.walk(dest, followlinks=False):
        for name in (*dirnames, *filenames):
            path = Path(dirpath) / name
            if path.is_symlink() or path.is_file():
                entries.append(path.relative_to(dest).as_posix())
    return sorted(entries)


def _install_dest_cp_sigterm_wrapper(bin_dir: Path, dest: Path) -> Path:
    dest_root = dest.resolve()
    return _write_executable(
        bin_dir / "cp",
        "\n".join(
            [
                "#!/bin/sh",
                f'dest_root={str(dest_root)!r}',
                'eval "last=\\${$#}"',
                'case "$last" in',
                '  "$dest_root"|"$dest_root"/*)',
                '    kill -s TERM "$PPID" || true',
                '    exec /bin/cp "$@"',
                "    ;;",
                "esac",
                'exec /bin/cp "$@"',
                "",
            ]
        ),
    )


def _fail_reason(result: subprocess.CompletedProcess[str]) -> str:
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None, result.stdout + result.stderr
    return match.group(1)


def test_materialize_dest_file_fails_redacted_without_write(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    dest.rmdir()
    dest.write_text("preexisting-dest-file\n", encoding="utf-8")

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "dest-not-directory"
    assert dest.is_file()
    assert dest.read_text(encoding="utf-8") == "preexisting-dest-file\n"
    _assert_redacted(result, str(dest), str(tmp_path / "bin"), "preexisting-dest-file")


def test_materialize_component_file_fails_redacted_without_write(
    tmp_path: Path,
) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    component = dest / "syn"
    component.write_text("preexisting-component-file\n", encoding="utf-8")

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "dest-not-directory"
    assert component.is_file()
    assert component.read_text(encoding="utf-8") == "preexisting-component-file\n"
    assert not (dest / ARTIFACT_PATHS[0]).exists()
    _assert_redacted(
        result, str(component), str(dest), "preexisting-component-file"
    )


def test_materialize_rejects_destination_symlink_before_write(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    dest.rmdir()
    dest.symlink_to(outside)

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "dest-symlink"
    assert dest.is_symlink()
    assert list(outside.iterdir()) == []
    _assert_redacted(result, str(dest), str(outside))


def test_materialize_rejects_destination_component_symlink(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (dest / "syn").symlink_to(outside)

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "dest-symlink"
    assert (dest / "syn").is_symlink()
    assert list(outside.iterdir()) == []
    assert not (outside / Path(ARTIFACT_PATHS[0]).name).exists()
    _assert_redacted(result, str(dest / "syn"), str(outside))


def test_materialize_rejects_destination_artifact_symlink(tmp_path: Path) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside-target\n")
    target = dest / ARTIFACT_PATHS[0]
    target.parent.mkdir()
    target.symlink_to(outside)

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "dest-symlink"
    assert target.is_symlink()
    assert outside.read_bytes() == b"outside-target\n"
    _assert_redacted(result, str(target), str(outside), "outside-target")


def test_final_publish_interruption_leaves_no_new_destination_artifacts(
    tmp_path: Path,
) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    stale = dest / "pre-existing.txt"
    stale.write_text("keep-me\n", encoding="utf-8")

    result = _run(
        MATERIALIZE,
        tmp_path,
        receipt=receipt,
        dest=dest,
        checkout=checkout,
        extra_env={"SEEON_MODEL_TEST_PUBLISH_SIGNAL": "TERM"},
    )

    assert result.returncode != 0
    _assert_redacted(result, str(dest), str(tmp_path / "source-tree"), "keep-me")
    assert stale.read_text(encoding="utf-8") == "keep-me\n"
    assert _declared_dest_files(dest) == ["pre-existing.txt"]


def test_final_publish_failure_leaves_no_new_destination_artifacts(
    tmp_path: Path,
) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    stale = dest / "pre-existing.txt"
    stale.write_text("keep-me\n", encoding="utf-8")

    result = _run(
        MATERIALIZE,
        tmp_path,
        receipt=receipt,
        dest=dest,
        checkout=checkout,
        extra_env={"SEEON_MODEL_TEST_PUBLISH_FAIL": "1"},
    )

    assert result.returncode != 0
    assert _fail_reason(result) == "copy-failed"
    _assert_redacted(result, str(dest), "keep-me")
    assert stale.read_text(encoding="utf-8") == "keep-me\n"
    assert _declared_dest_files(dest) == ["pre-existing.txt"]


def test_real_path_cp_sigterm_during_dest_copy_leaves_only_preexisting(
    tmp_path: Path,
) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    stale = dest / "pre-existing.txt"
    stale.write_text("keep-me\n", encoding="utf-8")
    _install_dest_cp_sigterm_wrapper(tmp_path / "bin", dest)

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "copy-failed"
    _assert_redacted(result, str(dest), str(tmp_path / "source-tree"), "keep-me")
    assert stale.read_text(encoding="utf-8") == "keep-me\n"
    leftover = _declared_dest_files(dest)
    assert leftover == ["pre-existing.txt"]
    hidden = [
        path
        for path in leftover
        if Path(path).name.startswith(".") or ".seeon-" in path
    ]
    assert hidden == []


def test_checkout_tracked_sidecar_symlink_is_rejected_before_hash_or_copy(
    tmp_path: Path,
) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    outside = tmp_path / "outside-sidecar.yaml"
    outside.write_bytes(sidecars[SIDECAR_PATHS[1]])
    tracked = checkout / SIDECAR_PATHS[1]
    tracked.unlink()
    tracked.symlink_to(outside)
    subprocess.run(
        ["git", "add", "-f", "--", SIDECAR_PATHS[1]],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "tracked sidecar symlink"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    stale = dest / "pre-existing.txt"
    stale.write_text("keep-me\n", encoding="utf-8")

    result = _run(MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "sidecar-symlink"
    argv_path = tmp_path / "bin" / "docker.argv"
    if argv_path.is_file():
        assert "cp\0" not in argv_path.read_text(encoding="utf-8")
    assert tracked.is_symlink()
    assert outside.read_bytes() == sidecars[SIDECAR_PATHS[1]]
    assert stale.read_text(encoding="utf-8") == "keep-me\n"
    assert _declared_dest_files(dest) == ["pre-existing.txt"]
    _assert_redacted(result, str(tracked), str(outside), str(dest))


def test_verify_rejects_destination_symlink_and_never_follows_outside(
    tmp_path: Path,
) -> None:
    receipt, dest, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    materialized = _run(
        MATERIALIZE, tmp_path, receipt=receipt, dest=dest, checkout=checkout
    )
    assert materialized.returncode == 0, materialized.stderr
    outside = tmp_path / "outside-tree"
    dest.rename(outside)
    dest.symlink_to(outside)

    result = _run(VERIFY, tmp_path, receipt=receipt, dest=dest, checkout=checkout)

    assert result.returncode != 0
    match = _VERIFY_FAIL.search(result.stdout) or _VERIFY_FAIL.search(result.stderr)
    assert match is not None, result.stdout + result.stderr
    assert match.group(1) == "dest-symlink"
    assert _VERIFY_OK.search(result.stdout) is None
    _assert_redacted(result, str(dest), str(outside))
