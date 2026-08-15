#!/usr/bin/env python3
"""Pre-boot private hash-only receipt generator.

Root-only. Inspects an approved running legacy worker through the normal
Docker CLI, hashes the closed default artifact set plus tracked model-root
sidecars, and writes the receipt consumed by materialize-model-artifacts.sh.
Emits only a count/sidecar_count/dest_sha256 verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE_ROOT = "/app/models"
ARTIFACT_PATHS = (
    "pose/yolo26n-pose.pt",
    "bed/yolo26m-seg.pt",
    "fall/lstm/model.pt",
)
SIDECAR_PATHS = (
    "fall/lstm/arch.json",
    "fall/lstm/metadata.yaml",
)
CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
REV_RE = re.compile(r"^[0-9a-f]{40}$")
INSPECT_FORMAT = '{{.Image}} {{index .Config.Labels "org.opencontainers.image.revision"}}'


class _Interrupted(Exception):
    """Raised from SIGINT/SIGTERM/SIGHUP so finally cleanup still runs."""


def _fail(reason: str) -> int:
    print(f"MODEL_RECEIPT_FAIL reason={reason}")
    return 1


def _install_interrupt_handlers() -> None:
    def _handle(_signum: int, _frame: object) -> None:
        raise _Interrupted()

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _handle)


def _usage() -> int:
    print(
        "Usage: scripts/ops/generate-model-receipt.sh --container NAME --out PATH --checkout PATH",
        file=sys.stderr,
    )
    return 2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_out(out: Path) -> str | None:
    parent = out.parent
    try:
        parent_st = os.lstat(parent)
    except OSError:
        return "unsafe-output"
    if stat.S_ISLNK(parent_st.st_mode):
        return "output-symlink"
    if not stat.S_ISDIR(parent_st.st_mode):
        return "unsafe-output"
    if parent_st.st_uid != os.geteuid():
        return "incorrect-ownership"
    if parent_st.st_mode & 0o022:
        return "unsafe-output"
    try:
        out_st = os.lstat(out)
    except FileNotFoundError:
        return None
    except OSError:
        return "unsafe-output"
    if stat.S_ISLNK(out_st.st_mode):
        return "output-symlink"
    if not stat.S_ISREG(out_st.st_mode):
        return "unsafe-output"
    return None


def _atomic_write(out: Path, data: bytes) -> str | None:
    reason = _validate_out(out)
    if reason:
        return reason
    parent = out.parent
    dir_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(parent, dir_flags)
    except OSError:
        try:
            if stat.S_ISLNK(os.lstat(parent).st_mode):
                return "output-symlink"
        except OSError:
            pass
        return "unsafe-output"
    tmp_name: str | None = None
    try:
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        fd: int | None = None
        for _ in range(8):
            candidate = f".seeon-receipt.{os.getpid()}.{secrets.token_hex(8)}"
            try:
                fd = os.open(candidate, create_flags, 0o600, dir_fd=parent_fd)
                tmp_name = candidate
                break
            except FileExistsError:
                continue
            except OSError:
                return "output-symlink"
        if fd is None or tmp_name is None:
            return "unsafe-output"
        try:
            os.write(fd, data)
            os.fchmod(fd, 0o600)
            os.fsync(fd)
        except OSError:
            return "unsafe-output"
        finally:
            os.close(fd)
        try:
            out_st = os.stat(out.name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(out_st.st_mode):
                return "output-symlink"
            if not stat.S_ISREG(out_st.st_mode):
                return "unsafe-output"
        except FileNotFoundError:
            pass
        except OSError:
            return "unsafe-output"
        try:
            os.replace(tmp_name, out.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        except OSError:
            return "unsafe-output"
        tmp_name = None
        return None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name, dir_fd=parent_fd)
            except OSError:
                pass
        os.close(parent_fd)


def _parse_args(argv: list[str]) -> tuple[str, Path, Path] | int:
    container = ""
    out = ""
    checkout = ""
    args = list(argv)
    while args:
        flag = args.pop(0)
        if flag in {"-h", "--help"}:
            return _usage()
        if flag not in {"--container", "--out", "--checkout"}:
            return _usage()
        if not args:
            return _usage()
        value = args.pop(0)
        if flag == "--container":
            container = value
        elif flag == "--out":
            out = value
        else:
            checkout = value
    if not container or not out or not checkout:
        return _usage()
    return container, Path(out), Path(checkout)


def _git(checkout: Path, *git_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(checkout), *git_args],
        check=False,
        capture_output=True,
        text=True,
    )


def _docker() -> str:
    return os.environ.get("SEEON_MODEL_DOCKER", "docker")


def _inspect_identity(container: str) -> tuple[str, str] | str:
    result = subprocess.run(
        [_docker(), "inspect", "--format", INSPECT_FORMAT, container],
        check=False,
        capture_output=True,
        text=True,
    )
    inspect_out = result.stdout.replace("\r", "").splitlines()
    inspect_out = inspect_out[0] if inspect_out else ""
    if result.returncode != 0:
        return "source-identity"
    parts = inspect_out.split(" ", 1)
    if len(parts) != 2:
        return "source-identity"
    image, revision = parts
    if not image.startswith("sha256:"):
        return "source-identity"
    digest = image[len("sha256:") :]
    if not SHA_RE.fullmatch(digest) or not REV_RE.fullmatch(revision):
        return "source-identity"
    return digest, revision


def _copy_artifact(container: str, relative: str, dest: Path) -> str | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [_docker(), "cp", f"{container}:{SOURCE_ROOT}/{relative}", str(dest)],
        check=False,
        capture_output=True,
        text=True,
    )
    err = (result.stderr or "").replace("\r", "")
    if result.returncode != 0:
        if "No such file" in err:
            return "missing-artifact"
        return "copy-failed"
    if dest.is_symlink() or not dest.is_file():
        return "missing-artifact"
    return None


def main(argv: list[str] | None = None) -> int:
    os.umask(0o077)
    _install_interrupt_handlers()
    try:
        return _run(sys.argv[1:] if argv is None else argv)
    except _Interrupted:
        return _fail("copy-failed")


def _run(argv: list[str]) -> int:
    parsed = _parse_args(argv)
    if isinstance(parsed, int):
        return parsed
    container, out, checkout = parsed
    if os.geteuid() != 0:
        return _fail("not-root")
    if not CONTAINER_RE.fullmatch(container):
        return _fail("source-identity")
    if not checkout.is_dir():
        return _fail("missing-sidecar")
    inside = _git(checkout, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0:
        return _fail("dirty-checkout")
    status = _git(checkout, "status", "--porcelain")
    if status.returncode != 0 or status.stdout.strip():
        return _fail("dirty-checkout")
    for relative in SIDECAR_PATHS:
        tracked = _git(checkout, "ls-files", "--error-unmatch", "--", relative)
        if tracked.returncode != 0:
            return _fail("missing-sidecar")
        src = checkout / relative
        if src.is_symlink():
            return _fail("sidecar-symlink")
        if not src.is_file():
            return _fail("missing-sidecar")
    early = _validate_out(out)
    if early:
        return _fail(early)

    work = Path(tempfile.mkdtemp(prefix="seeon-model-receipt."))
    try:
        identity = _inspect_identity(container)
        if isinstance(identity, str):
            return _fail(identity)
        digest, revision = identity
        mapping: dict[str, tuple[str, str]] = {}
        for relative in ARTIFACT_PATHS:
            staged = work / "stage" / relative
            copy_reason = _copy_artifact(container, relative, staged)
            if copy_reason:
                return _fail(copy_reason)
            mapping[relative] = (_sha256_file(staged), "weight")
            staged.unlink(missing_ok=True)
        for relative in SIDECAR_PATHS:
            src = checkout / relative
            if src.is_symlink():
                return _fail("sidecar-symlink")
            if not src.is_file():
                return _fail("missing-sidecar")
            mapping[relative] = (_sha256_file(src), "sidecar")
        artifacts = [
            {"path": path, "sha256": mapping[path][0], "class": mapping[path][1]}
            for path in ARTIFACT_PATHS
        ]
        sidecars = [
            {"path": path, "sha256": mapping[path][0], "class": mapping[path][1]}
            for path in SIDECAR_PATHS
        ]
        body = {
            "schemaVersion": 1,
            "source": {
                "kind": "docker-cli",
                "container": container,
                "imageDigest": digest,
                "revision": revision,
                "root": SOURCE_ROOT,
            },
            "artifacts": artifacts,
            "sidecars": sidecars,
        }
        payload = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8")
        parse = Path(__file__).with_name("parse-model-receipt.py")
        parsed_dir = work / "parsed"
        parsed_dir.mkdir()
        draft = work / "receipt.json"
        draft.write_bytes(payload)
        parsed = subprocess.run(
            [sys.executable, str(parse), str(draft), str(parsed_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
        if parsed.returncode != 0:
            return _fail("malformed-receipt")
        dest_payload = "".join(
            f"{path}\t{mapping[path][0]}\n" for path in sorted(mapping)
        )
        dest_sha256 = hashlib.sha256(dest_payload.encode("utf-8")).hexdigest()
        write_reason = _atomic_write(out, payload)
        if write_reason:
            return _fail(write_reason)
        print(
            "MODEL_RECEIPT_OK count=3 sidecar_count=2 dest_sha256="
            f"{dest_sha256}"
        )
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
