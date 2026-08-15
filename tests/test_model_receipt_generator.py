"""Subprocess contract for the pre-boot private model receipt generator.

The generator inspects an approved running worker through a mocked Docker
CLI adapter, hashes the closed default artifact set, and writes the
hash-only receipt consumed by the existing materializer parser. Fixtures
are synthetic bytes only: no production weights, credentials, RTSP, or
host paths.
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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATE = REPO_ROOT / "scripts" / "ops" / "generate-model-receipt.sh"
MATERIALIZE = REPO_ROOT / "scripts" / "ops" / "materialize-model-artifacts.sh"
VERIFY = REPO_ROOT / "scripts" / "ops" / "verify-model-artifacts.sh"
PARSE_RECEIPT = REPO_ROOT / "scripts" / "ops" / "parse-model-receipt.py"
PUBLISH_RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "edge-image-publish.md"

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
REVISION = "c" * 40
SOURCE_ROOT = "/app/models"
CONTAINER = "synthetic-worker"

ARTIFACT_PATHS = (
    "pose/yolo26n-pose.pt",
    "bed/yolo26m-seg.pt",
    "fall/lstm/model.pt",
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
    r"^MODEL_RECEIPT_OK count=3 sidecar_count=2 dest_sha256=([0-9a-f]{64})$",
    re.MULTILINE,
)
_FAIL = re.compile(r"^MODEL_RECEIPT_FAIL reason=([a-z0-9-]+)$", re.MULTILINE)
_MATERIALIZE_OK = re.compile(
    r"^MODEL_MATERIALIZATION_OK count=3 sidecar_count=2 dest_sha256=([0-9a-f]{64})$",
    re.MULTILINE,
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def dest_receipt_sha256(path_to_digest: Mapping[str, str]) -> str:
    payload = "".join(f"{path}\t{digest}\n" for path, digest in sorted(path_to_digest.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _payload(label: str) -> bytes:
    return f"synthetic-{label}\n".encode("ascii")


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _init_checkout(root: Path, sidecars: Mapping[str, bytes]) -> Path:
    checkout = root / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "model-receipt@example.test"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Model Receipt"],
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


def _id_stub(bin_dir: Path, uid: int) -> Path:
    return _write_executable(bin_dir / "id", f"#!/bin/sh\necho {uid}\n")


def _docker_stub(
    path: Path,
    *,
    source_tree: Path,
    inspect_stdout: str,
    inspect_status: int = 0,
    cp_status: int = 0,
    fail_cp_on: str | None = None,
    write_cp: bool = True,
    signal_on_cp: str | None = None,
    ready_file: Path | None = None,
) -> Path:
    script = f"""#!/usr/bin/env python3
import os
import shutil
import signal
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
        sys.stderr.write("No such file\\n")
        raise SystemExit({cp_status if cp_status != 0 else 1})
    if {write_cp!r}:
        src = source_tree / relative
        if not src.is_file():
            sys.stderr.write("No such file\\n")
            raise SystemExit(1)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    if {bool(signal_on_cp)}:
        ready = Path({str(ready_file) if ready_file is not None else ""!r})
        if ready.name:
            ready.write_text(str(os.getppid()), encoding="utf-8")
        os.kill(os.getppid(), getattr(signal, {signal_on_cp!r}))
        raise SystemExit(0)
    raise SystemExit({cp_status})

sys.stderr.write("docker-stub: unexpected command\\n")
raise SystemExit(2)
"""
    return _write_executable(path / "docker", script)


def _env(bin_dir: Path, *, uid: int = 0) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["SEEON_MODEL_DOCKER"] = str(bin_dir / "docker")
    _id_stub(bin_dir, uid)
    return env


def _run(
    tmp_path: Path,
    *,
    container: str = CONTAINER,
    out: Path,
    checkout: Path,
    extra_env: Mapping[str, str] | None = None,
    extra_args: tuple[str, ...] = (),
    uid: int = 0,
    through_shell: bool = False,
    impersonate_euid: int | None = 0,
    owner_mismatch: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = _env(tmp_path / "bin", uid=uid)
    if extra_env:
        env.update(extra_env)
    script_args = [
        "--container",
        container,
        "--out",
        str(out),
        "--checkout",
        str(checkout),
        *extra_args,
    ]
    if through_shell:
        argv = ["bash", str(GENERATE), *script_args]
    else:
        helper = REPO_ROOT / "scripts" / "ops" / "generate-model-receipt.py"
        wrapper = tmp_path / "bin" / "seeon-euid-wrapper.py"
        wrapper.write_text(
            "\n".join(
                [
                    "import os",
                    "import runpy",
                    "import sys",
                    "REAL_EUID = os.geteuid()",
                    f"IMPERSONATED = {impersonate_euid}",
                    f"MISMATCH = {owner_mismatch!r}",
                    "class _Stat:",
                    "    def __init__(self, st, uid):",
                    "        object.__setattr__(self, '_st', st)",
                    "        object.__setattr__(self, '_uid', uid)",
                    "    def __getattr__(self, name):",
                    "        if name == 'st_uid':",
                    "            return self._uid",
                    "        return getattr(self._st, name)",
                    "def _adjust(st):",
                    "    if MISMATCH:",
                    "        return _Stat(st, IMPERSONATED + 1)",
                    "    if st.st_uid == REAL_EUID:",
                    "        return _Stat(st, IMPERSONATED)",
                    "    return st",
                    "_real_lstat = os.lstat",
                    "_real_stat = os.stat",
                    "os.geteuid = lambda: IMPERSONATED",
                    "def _lstat(*a, **k):",
                    "    return _adjust(_real_lstat(*a, **k))",
                    "def _stat(*a, **k):",
                    "    return _adjust(_real_stat(*a, **k))",
                    "os.lstat = _lstat",
                    "os.stat = _stat",
                    "target, *rest = sys.argv[1:]",
                    "sys.argv = [target, *rest]",
                    "runpy.run_path(target, run_name='__main__')",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        argv = ["python3", str(wrapper), str(helper), *script_args]
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _happy_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, bytes], dict[str, bytes]]:
    artifacts = {path: _payload(Path(path).name) for path in ARTIFACT_PATHS}
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
    out = tmp_path / "private" / "receipt.json"
    out.parent.mkdir()
    return out, checkout, artifacts, sidecars


def _combined(artifacts: Mapping[str, bytes], sidecars: Mapping[str, bytes]) -> dict[str, str]:
    merged = {path: _sha256_bytes(payload) for path, payload in artifacts.items()}
    merged.update({path: _sha256_bytes(payload) for path, payload in sidecars.items()})
    return merged


def _assert_redacted(result: subprocess.CompletedProcess[str], *extra: str) -> None:
    blob = f"{result.stdout}\n{result.stderr}"
    for needle in (*FORBIDDEN_OUTPUT, *extra):
        assert needle not in blob, f"unredacted {needle!r} in script output"


def _fail_reason(result: subprocess.CompletedProcess[str]) -> str:
    match = _FAIL.search(result.stdout) or _FAIL.search(result.stderr)
    assert match is not None, result.stdout + result.stderr
    return match.group(1)


def _parse_receipt(receipt: Path, work: Path) -> subprocess.CompletedProcess[str]:
    work.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["python3", str(PARSE_RECEIPT), str(receipt), str(work)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_generator_script_is_executable_pre_boot_interface() -> None:
    assert GENERATE.is_file()
    assert GENERATE.stat().st_mode & stat.S_IXUSR
    assert GENERATE.read_text(encoding="utf-8").splitlines()[0] == "#!/bin/bash"


def test_generator_has_no_production_test_hooks_or_forbidden_surfaces() -> None:
    text = GENERATE.read_text(encoding="utf-8")
    assert "SEEON_MODEL_TEST" not in text
    assert "docker.sock" not in text
    assert "docker compose" not in text
    assert "docker-compose" not in text
    assert "huggingface" not in text
    assert "privileged" not in text
    assert "fetch-models" not in text
    assert "id -u" not in text
    assert "$(id" not in text
    helper = (REPO_ROOT / "scripts" / "ops" / "generate-model-receipt.py").read_text(
        encoding="utf-8"
    )
    assert "os.geteuid()" in helper
    assert "O_EXCL" in helper
    assert "O_NOFOLLOW" in helper
    assert "id -u" not in helper
    assert "SEEON_MODEL_TEST" not in helper


def test_publish_runbook_cites_the_private_receipt_generator() -> None:
    runbook = PUBLISH_RUNBOOK.read_text(encoding="utf-8")
    assert "scripts/ops/generate-model-receipt.sh" in runbook
    assert "--checkout ./models" in runbook


def test_generator_happy_path_writes_hash_only_receipt(
    tmp_path: Path,
) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)

    result = _run(tmp_path, out=out, checkout=checkout)

    assert result.returncode == 0, result.stderr
    match = _SUCCESS.search(result.stdout)
    assert match is not None, result.stdout
    expected = dest_receipt_sha256(_combined(artifacts, sidecars))
    assert match.group(1) == expected
    _assert_redacted(
        result,
        CONTAINER,
        "synthetic-yolo26n-pose.pt",
        str(out),
        str(checkout),
        str(tmp_path / "source-tree"),
        DIGEST_A,
        REVISION,
        *[digest for digest in _combined(artifacts, sidecars).values() if digest != expected],
    )
    assert out.is_file()
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    body = json.loads(out.read_text(encoding="utf-8"))
    assert set(body) == {"schemaVersion", "source", "artifacts", "sidecars"}
    assert body["schemaVersion"] == 1
    assert body["source"] == {
        "kind": "docker-cli",
        "container": CONTAINER,
        "imageDigest": DIGEST_A,
        "revision": REVISION,
        "root": SOURCE_ROOT,
    }
    assert [item["path"] for item in body["artifacts"]] == list(ARTIFACT_PATHS)
    assert [item["class"] for item in body["artifacts"]] == ["weight", "weight", "weight"]
    assert [item["sha256"] for item in body["artifacts"]] == [
        _sha256_bytes(artifacts[path]) for path in ARTIFACT_PATHS
    ]
    assert [item["path"] for item in body["sidecars"]] == list(SIDECAR_PATHS)
    assert [item["class"] for item in body["sidecars"]] == ["sidecar", "sidecar"]
    parsed = _parse_receipt(out, tmp_path / "parsed")
    assert parsed.returncode == 0, parsed.stderr
    argv = (tmp_path / "bin" / "docker.argv").read_text(encoding="utf-8")
    inspect_line = argv.splitlines()[0]
    assert inspect_line.startswith("inspect")
    assert "--format" in inspect_line
    assert "{{json ." not in inspect_line
    assert "Config.Env" not in inspect_line
    assert ".State" not in inspect_line
    assert all(line.startswith("cp\0") for line in argv.splitlines()[1:])
    assert argv.count("cp\0") == 3
    leftover = list((tmp_path / "private").glob(".seeon-receipt.*"))
    leftover.extend(
        path
        for path in tmp_path.iterdir()
        if path.name.startswith("seeon-model-receipt.")
    )
    assert leftover == []


def test_generated_receipt_is_accepted_by_materializer_and_verifier(
    tmp_path: Path,
) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    generated = _run(tmp_path, out=out, checkout=checkout)
    assert generated.returncode == 0, generated.stderr
    dest = tmp_path / "dest"
    dest.mkdir()
    env = _env(tmp_path / "bin")
    materialized = subprocess.run(
        [
            "bash",
            str(MATERIALIZE),
            "--receipt",
            str(out),
            "--dest",
            str(dest),
            "--checkout",
            str(checkout),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert materialized.returncode == 0, materialized.stderr
    match = _MATERIALIZE_OK.search(materialized.stdout)
    assert match is not None, materialized.stdout
    assert match.group(1) == dest_receipt_sha256(_combined(artifacts, sidecars))
    verified = subprocess.run(
        [
            "bash",
            str(VERIFY),
            "--receipt",
            str(out),
            "--dest",
            str(dest),
            "--checkout",
            str(checkout),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert verified.returncode == 0, verified.stderr


def test_generator_rejects_non_root_before_any_docker(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)

    result = _run(tmp_path, out=out, checkout=checkout, impersonate_euid=501)

    assert result.returncode != 0
    assert _fail_reason(result) == "not-root"
    assert not out.exists()
    argv_path = tmp_path / "bin" / "docker.argv"
    assert not argv_path.exists()
    _assert_redacted(result, CONTAINER, str(out))


@pytest.mark.skipif(os.geteuid() == 0, reason="PATH id spoof is only observable as non-root")
def test_path_stubbed_id_zero_cannot_generate_or_touch_docker(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)

    result = _run(tmp_path, out=out, checkout=checkout, uid=0, through_shell=True)

    assert result.returncode != 0
    assert _fail_reason(result) == "not-root"
    assert not out.exists()
    argv_path = tmp_path / "bin" / "docker.argv"
    assert not argv_path.exists()
    assert not list((tmp_path / "private").glob(".seeon-receipt.*"))
    _assert_redacted(result, CONTAINER, str(out), str(checkout))


def test_usage_missing_args_exits_without_receipt(tmp_path: Path) -> None:
    _happy_fixture(tmp_path)
    env = _env(tmp_path / "bin")
    result = subprocess.run(
        ["bash", str(GENERATE), "--container", CONTAINER],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    assert not _SUCCESS.search(result.stdout)
    _assert_redacted(result, CONTAINER)


def test_generator_rejects_source_identity_before_any_copy(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    _docker_stub(
        tmp_path / "bin",
        source_tree=tmp_path / "source-tree",
        inspect_stdout="",
        inspect_status=1,
    )

    result = _run(tmp_path, out=out, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "source-identity"
    assert not out.exists()
    argv = (tmp_path / "bin" / "docker.argv").read_text(encoding="utf-8")
    assert "cp\0" not in argv
    _assert_redacted(result, CONTAINER, str(out))


def test_generator_rejects_misleading_inspect_success_before_copy(
    tmp_path: Path,
) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    _docker_stub(
        tmp_path / "bin",
        source_tree=tmp_path / "source-tree",
        inspect_stdout=f"OK Image=sha256:{DIGEST_A} revision={REVISION} success=true",
    )

    result = _run(tmp_path, out=out, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "source-identity"
    argv = (tmp_path / "bin" / "docker.argv").read_text(encoding="utf-8")
    assert "cp\0" not in argv
    assert not out.exists()


@pytest.mark.parametrize(
    "inspect_stdout",
    [
        f"sha256:{DIGEST_B.upper()} {REVISION}",
        f"sha256:{DIGEST_A} {REVISION.upper()}",
        f"sha256:{DIGEST_A[:63]} {REVISION}",
        f"{DIGEST_A} {REVISION}",
        f"sha256:{DIGEST_A}",
    ],
)
def test_generator_rejects_invalid_identity_format_before_copy(
    tmp_path: Path, inspect_stdout: str
) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    _docker_stub(
        tmp_path / "bin",
        source_tree=tmp_path / "source-tree",
        inspect_stdout=inspect_stdout,
    )

    result = _run(tmp_path, out=out, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "source-identity"
    argv = (tmp_path / "bin" / "docker.argv").read_text(encoding="utf-8")
    assert "cp\0" not in argv
    assert not out.exists()
    _assert_redacted(result, CONTAINER, DIGEST_B, inspect_stdout)


def test_sigterm_during_docker_cp_removes_staged_bytes_and_emits_copy_failed(
    tmp_path: Path,
) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    ready = tmp_path / "bin" / "cp.ready"
    _docker_stub(
        tmp_path / "bin",
        source_tree=tmp_path / "source-tree",
        inspect_stdout=f"sha256:{DIGEST_A} {REVISION}",
        signal_on_cp="SIGTERM",
        ready_file=ready,
    )
    tracked_tmp = tmp_path / "tracked-tmp"
    tracked_tmp.mkdir()

    result = _run(
        tmp_path,
        out=out,
        checkout=checkout,
        extra_env={"TMPDIR": str(tracked_tmp)},
    )

    assert result.returncode != 0
    assert _fail_reason(result) == "copy-failed"
    assert _SUCCESS.search(result.stdout) is None
    assert not out.exists()
    leftover = list(tracked_tmp.rglob("*"))
    leftover.extend(
        path
        for path in tracked_tmp.iterdir()
        if path.name.startswith("seeon-model-receipt.")
    )
    assert leftover == []
    leftover_receipts = list((tmp_path / "private").glob(".seeon-receipt.*"))
    assert leftover_receipts == []
    argv = (tmp_path / "bin" / "docker.argv").read_text(encoding="utf-8")
    assert argv.splitlines()[0].startswith("inspect")
    assert any(line.startswith("cp\0") for line in argv.splitlines()[1:])
    _assert_redacted(
        result,
        CONTAINER,
        str(out),
        str(tracked_tmp),
        "synthetic-yolo26n-pose.pt",
        artifacts[ARTIFACT_PATHS[0]].decode("ascii").strip(),
    )


def test_generator_fails_closed_on_missing_declared_artifact(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    missing = ARTIFACT_PATHS[0]
    (tmp_path / "source-tree" / missing).unlink()

    result = _run(tmp_path, out=out, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "missing-artifact"
    assert not out.exists()
    _assert_redacted(result, missing, CONTAINER, str(out))


def test_generator_rejects_dirty_checkout_before_copy(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    (checkout / SIDECAR_PATHS[1]).write_bytes(b"dirty-sidecar\n")

    result = _run(tmp_path, out=out, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "dirty-checkout"
    argv_path = tmp_path / "bin" / "docker.argv"
    assert not argv_path.exists() or "cp\0" not in argv_path.read_text(encoding="utf-8")
    assert not out.exists()
    _assert_redacted(result, "dirty-sidecar", CONTAINER)


def test_generator_rejects_missing_tracked_sidecar_before_copy(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
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

    result = _run(tmp_path, out=out, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "missing-sidecar"
    argv_path = tmp_path / "bin" / "docker.argv"
    assert not argv_path.exists() or "cp\0" not in argv_path.read_text(encoding="utf-8")
    assert not out.exists()


def test_generator_rejects_sidecar_symlink_before_copy(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
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

    result = _run(tmp_path, out=out, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "sidecar-symlink"
    argv_path = tmp_path / "bin" / "docker.argv"
    assert not argv_path.exists() or "cp\0" not in argv_path.read_text(encoding="utf-8")
    assert tracked.is_symlink()
    assert outside.read_bytes() == sidecars[SIDECAR_PATHS[1]]
    assert not out.exists()
    _assert_redacted(result, str(tracked), str(outside), CONTAINER)


def test_generator_rejects_missing_output_parent(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    missing_parent = tmp_path / "absent" / "receipt.json"

    result = _run(tmp_path, out=missing_parent, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "unsafe-output"
    assert not missing_parent.exists()
    argv_path = tmp_path / "bin" / "docker.argv"
    assert not argv_path.exists() or "cp\0" not in argv_path.read_text(encoding="utf-8")
    _assert_redacted(result, str(missing_parent), CONTAINER)


def test_generator_rejects_output_parent_file(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    parent = tmp_path / "parent-file"
    parent.write_text("not-a-directory\n", encoding="utf-8")
    target = parent / "receipt.json"

    result = _run(tmp_path, out=target, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "unsafe-output"
    assert parent.read_text(encoding="utf-8") == "not-a-directory\n"
    _assert_redacted(result, str(parent), "not-a-directory")


def test_generator_rejects_output_symlink(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    outside = tmp_path / "outside-receipt.json"
    outside.write_text("{}\n", encoding="utf-8")
    out.symlink_to(outside)

    result = _run(tmp_path, out=out, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "output-symlink"
    assert out.is_symlink()
    assert outside.read_text(encoding="utf-8") == "{}\n"
    argv_path = tmp_path / "bin" / "docker.argv"
    assert not argv_path.exists() or "cp\0" not in argv_path.read_text(encoding="utf-8")
    _assert_redacted(result, str(out), str(outside))


def test_generator_rejects_output_parent_symlink(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent)
    target = linked_parent / "receipt.json"

    result = _run(tmp_path, out=target, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "output-symlink"
    assert list(real_parent.iterdir()) == []
    _assert_redacted(result, str(linked_parent), str(target))


@pytest.mark.parametrize("mode", [0o777, 0o770, 0o707])
def test_generator_rejects_group_or_world_writable_parent(
    tmp_path: Path, mode: int
) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    out.parent.chmod(mode)

    result = _run(tmp_path, out=out, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "unsafe-output"
    assert not out.exists()
    assert not list(out.parent.glob(".seeon-receipt.*"))
    argv_path = tmp_path / "bin" / "docker.argv"
    assert not argv_path.exists() or "cp\0" not in argv_path.read_text(encoding="utf-8")
    _assert_redacted(result, str(out), CONTAINER)


def test_generator_rejects_directory_out_without_writing_body(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    out.mkdir()

    result = _run(tmp_path, out=out, checkout=checkout)

    assert result.returncode != 0
    assert _fail_reason(result) == "unsafe-output"
    assert out.is_dir()
    assert not out.is_symlink()
    assert list(out.iterdir()) == []
    assert not list(out.parent.glob(".seeon-receipt.*"))
    argv_path = tmp_path / "bin" / "docker.argv"
    assert not argv_path.exists() or "cp\0" not in argv_path.read_text(encoding="utf-8")
    _assert_redacted(result, str(out), CONTAINER)


def test_generator_does_not_follow_planted_temp_symlink(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    outside = tmp_path / "outside-temp.json"
    outside.write_text("planted-temp\n", encoding="utf-8")
    planted = [
        out.parent / ".seeon-receipt.1",
        out.parent / ".seeon-receipt.tmp",
        out.with_name(out.name + ".tmp"),
    ]
    for path in planted:
        path.symlink_to(outside)

    result = _run(tmp_path, out=out, checkout=checkout)

    assert outside.read_text(encoding="utf-8") == "planted-temp\n"
    assert not outside.is_symlink()
    for path in planted:
        assert path.is_symlink()
        assert path.resolve() == outside.resolve()
    leftover_body = [
        path
        for path in out.parent.iterdir()
        if path.is_file() and not path.is_symlink() and path != out
    ]
    assert result.returncode == 0, result.stderr
    match = _SUCCESS.search(result.stdout)
    assert match is not None, result.stdout
    assert out.is_file()
    assert not out.is_symlink()
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert leftover_body == []
    _assert_redacted(result, str(outside), "planted-temp", CONTAINER)


def test_generator_rejects_incorrect_output_parent_ownership(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)

    result = _run(tmp_path, out=out, checkout=checkout, owner_mismatch=True)

    assert result.returncode != 0
    assert _fail_reason(result) == "incorrect-ownership"
    assert not out.exists()
    argv_path = tmp_path / "bin" / "docker.argv"
    assert not argv_path.exists() or "cp\0" not in argv_path.read_text(encoding="utf-8")
    _assert_redacted(result, str(out), CONTAINER)


def test_generator_does_not_invoke_compose(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    _write_executable(
        tmp_path / "bin" / "docker-compose",
        "#!/bin/sh\necho compose-must-not-run >&2\nexit 0\n",
    )
    _write_executable(
        tmp_path / "bin" / "compose",
        "#!/bin/sh\necho compose-must-not-run >&2\nexit 0\n",
    )

    result = _run(tmp_path, out=out, checkout=checkout)

    assert result.returncode == 0, result.stderr
    assert "compose-must-not-run" not in result.stderr
    _assert_redacted(result, CONTAINER)


def test_generator_does_not_use_destination_tree(tmp_path: Path) -> None:
    out, checkout, artifacts, sidecars = _happy_fixture(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    marker = dest / "must-remain.txt"
    marker.write_text("keep-me\n", encoding="utf-8")

    result = _run(
        tmp_path,
        out=out,
        checkout=checkout,
        extra_args=("--dest", str(dest)),
    )

    assert result.returncode == 2
    assert marker.read_text(encoding="utf-8") == "keep-me\n"
    assert list(dest.iterdir()) == [marker]
    assert not out.exists()
