"""Characterization of the sealed edge updater's existing digest contract.

These tests pin today's mockable updater behavior (success, dry-run, digest
rollback) before the root-owned systemd carrier is added. They do not contact
Docker, GHCR, or a live backend.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UPDATER = REPO_ROOT / "scripts" / "edge-updater" / "update-edge.sh"
UPDATER_HARNESS = REPO_ROOT / "scripts" / "edge-updater" / "test.sh"
README = REPO_ROOT / "scripts" / "edge-updater" / "README.md"

API_OLD = "sha256:" + ("a" * 64)
API_NEW = "sha256:" + ("b" * 64)
WORKER_OLD = "sha256:" + ("c" * 64)
WORKER_NEW = "sha256:" + ("d" * 64)

_FORBIDDEN_PRIVILEGE = (
    "usermod",
    "docker.sock",
    "--privileged",
    "privileged:",
    "newgrp docker",
    "-G docker",
    "sg docker",
)


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _write_case_files(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / ".env.edge.prod").write_text(
        "\n".join(
            (
                "ML_SERVING_PORT=18080",
                "ML_API_IMAGE=ghcr.io/acme/ml-api:new",
                "ML_WORKER_IMAGE=ghcr.io/acme/ml-worker:new",
                "API_EDGE_RELAY_TOKEN=not-secret-for-test",
                "ML_EDGE_VERSION=test-version",
                "",
            )
        ),
        encoding="utf-8",
    )
    (case_dir / "compose.edge.yaml").write_text(
        "services:\n"
        "  ml-api:\n"
        "    image: ${ML_API_IMAGE}\n"
        "  ml-worker:\n"
        "    image: ${ML_WORKER_IMAGE}\n",
        encoding="utf-8",
    )


def _install_updater_mocks(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state = tmp_path / "mock-state"
    state.mkdir()
    (state / "docker.log").write_text("", encoding="utf-8")
    _write_executable(
        bin_dir / "docker",
        """#!/bin/sh
set -eu
last_arg=
for arg in "$@"; do
  last_arg=$arg
done
if [ "${1:-}" = "buildx" ] && [ "${2:-}" = "imagetools" ] && [ "${3:-}" = "inspect" ]; then
  case "$4" in
    *ml-api*) digest=$API_NEW ;;
    *ml-worker*) digest=$WORKER_NEW ;;
    *) exit 1 ;;
  esac
  printf 'Name: %s\\nDigest: %s\\n' "$4" "$digest"
  exit 0
fi
if [ "${1:-}" = "image" ] && [ "${2:-}" = "inspect" ]; then
  case "$last_arg" in
    *ml-api*) printf 'ghcr.io/acme/ml-api@%s\\n' "$API_OLD" ;;
    *ml-worker*) printf 'ghcr.io/acme/ml-worker@%s\\n' "$WORKER_OLD" ;;
    *) exit 1 ;;
  esac
  exit 0
fi
if [ "${1:-}" = "inspect" ]; then
  case "$last_arg" in
    cid-ml-api) printf 'image-ml-api\\n' ;;
    cid-ml-worker) printf 'image-ml-worker\\n' ;;
    *) exit 1 ;;
  esac
  exit 0
fi
if [ "${1:-}" = "compose" ]; then
  cmd=
  for arg in "$@"; do
    case "$arg" in
      ps|pull|up) cmd=$arg; break ;;
    esac
  done
  case "$cmd" in
    ps)
      printf 'cid-%s\\n' "$last_arg"
      exit 0
      ;;
    pull)
      printf 'pull %s\\n' "$*" >>"$TEST_STATE/docker.log"
      exit 0
      ;;
    up)
      count=0
      if [ -f "$TEST_STATE/up_count" ]; then
        count=$(cat "$TEST_STATE/up_count")
      fi
      count=$((count + 1))
      printf '%s\\n' "$count" >"$TEST_STATE/up_count"
      printf 'up %s\\n' "$*" >>"$TEST_STATE/docker.log"
      exit 0
      ;;
  esac
fi
printf 'unexpected docker invocation: %s\\n' "$*" >&2
exit 1
""",
    )
    _write_executable(
        bin_dir / "curl",
        """#!/bin/sh
set -eu
is_post=0
last_arg=
for arg in "$@"; do
  if [ "$arg" = "POST" ]; then
    is_post=1
  fi
  last_arg=$arg
done
if [ "$is_post" -eq 1 ]; then
  printf 'report %s\\n' "$*" >>"$TEST_STATE/report.log"
  exit 0
fi
case "$last_arg" in
  */health/ready)
    printf '200'
    exit 0
    ;;
  */api/v1/status)
    printf '{"cameras":{"cam-edge-01":{"status":"online"}},"stale_after_sec":90.0,"runtime":{}}'
    exit 0
    ;;
  */api/v1/system)
    up_count=0
    if [ -f "$TEST_STATE/up_count" ]; then
      up_count=$(cat "$TEST_STATE/up_count")
    fi
    if [ "${TEST_SCENARIO:-success}" = "rollback" ] && [ "$up_count" -lt 2 ]; then
      printf '{"version":"test-version","image_digests":'
      printf '{"ml_api":"sha256:%s","ml_worker":"sha256:%s"}}' \
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" \
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    elif [ "${TEST_SCENARIO:-success}" = "rollback" ]; then
      printf '{"version":"test-version","image_digests":{"ml_api":"%s","ml_worker":"%s"}}' \
        "$API_OLD" "$WORKER_OLD"
    else
      printf '{"version":"test-version","image_digests":{"ml_api":"%s","ml_worker":"%s"}}' \
        "$API_NEW" "$WORKER_NEW"
    fi
    exit 0
    ;;
esac
printf 'unexpected curl invocation: %s\\n' "$*" >&2
exit 1
""",
    )
    return bin_dir


def _run_updater(
    tmp_path: Path,
    *,
    scenario: str = "success",
    extra_env: dict[str, str] | None = None,
    env_file: Path | None = None,
    compose_file: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    case_dir = tmp_path / "case"
    data_dir = case_dir / "data"
    if env_file is None or compose_file is None:
        _write_case_files(case_dir)
        env_file = case_dir / ".env.edge.prod"
        compose_file = case_dir / "compose.edge.yaml"
    data_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = _install_updater_mocks(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_STATE": str(tmp_path / "mock-state"),
        "TEST_SCENARIO": scenario,
        "API_OLD": API_OLD,
        "API_NEW": API_NEW,
        "WORKER_OLD": WORKER_OLD,
        "WORKER_NEW": WORKER_NEW,
        "EDGE_UPDATER_DATA_DIR": str(data_dir),
        "EDGE_UPDATER_ENV_FILE": str(env_file),
        "EDGE_UPDATER_COMPOSE_FILE": str(compose_file),
        "EDGE_UPDATER_REPORT_URL": "http://backend.example.test/api/v1/edge-updater/report",
        "EDGE_UPDATER_VERIFY_TIMEOUT_SEC": "0",
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["sh", str(UPDATER)],
        cwd=case_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_existing_shell_harness_covers_success_and_digest_rollback() -> None:
    completed = subprocess.run(
        ["sh", str(UPDATER_HARNESS)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "edge-updater tests passed" in completed.stdout


def test_updater_success_commits_target_digests(tmp_path: Path) -> None:
    completed = _run_updater(tmp_path, scenario="success")
    data_dir = tmp_path / "case" / "data"
    docker_log = (tmp_path / "mock-state" / "docker.log").read_text(encoding="utf-8")

    assert completed.returncode == 0, completed.stderr
    assert '"status":"success"' in (data_dir / "update.log").read_text(encoding="utf-8")
    snapshot = (data_dir / "snapshot.json").read_text(encoding="utf-8")
    assert API_NEW in snapshot
    assert WORKER_NEW in snapshot
    assert "pull " in docker_log
    assert "up " in docker_log


def test_updater_dry_run_does_not_pull_apply_or_mutate_env(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_case_files(case_dir)
    env_file = case_dir / ".env.edge.prod"
    before = env_file.read_text(encoding="utf-8")
    completed = _run_updater(
        tmp_path,
        extra_env={"EDGE_UPDATER_DRY_RUN": "1"},
        env_file=env_file,
        compose_file=case_dir / "compose.edge.yaml",
    )
    docker_log = (tmp_path / "mock-state" / "docker.log").read_text(encoding="utf-8")

    assert completed.returncode == 0, completed.stderr
    assert "dry run" in completed.stdout.lower()
    assert env_file.read_text(encoding="utf-8") == before
    assert "pull " not in docker_log
    assert "up " not in docker_log
    assert not (tmp_path / "case" / "data" / "snapshot.json").exists()


def test_updater_rollback_restores_previous_image_digests(tmp_path: Path) -> None:
    completed = _run_updater(tmp_path, scenario="rollback")
    env_text = (tmp_path / "case" / ".env.edge.prod").read_text(encoding="utf-8")
    log_text = (tmp_path / "case" / "data" / "update.log").read_text(encoding="utf-8")

    assert completed.returncode != 0
    assert '"status":"rollback_success"' in log_text
    assert f"ML_API_IMAGE=ghcr.io/acme/ml-api@{API_OLD}" in env_text
    assert f"ML_WORKER_IMAGE=ghcr.io/acme/ml-worker@{WORKER_OLD}" in env_text


def test_updater_missing_env_file_fails_before_compose_mutation(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    compose = case_dir / "compose.edge.yaml"
    compose.write_text("services: {}\n", encoding="utf-8")
    completed = _run_updater(
        tmp_path,
        env_file=case_dir / "missing.env",
        compose_file=compose,
    )
    docker_log = (tmp_path / "mock-state" / "docker.log").read_text(encoding="utf-8")

    assert completed.returncode != 0
    assert "not found" in completed.stdout + completed.stderr
    assert "pull " not in docker_log
    assert "up " not in docker_log


def test_updater_and_readme_do_not_grant_interactive_docker_root() -> None:
    updater = UPDATER.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    for token in _FORBIDDEN_PRIVILEGE:
        assert token not in updater
        assert token not in readme
    assert "/var/run/docker.sock" not in updater
    assert "groupadd" not in updater


UNIT = REPO_ROOT / "scripts" / "edge-updater" / "systemd" / "seeon-edge-updater.service"
TIMER = REPO_ROOT / "scripts" / "edge-updater" / "systemd" / "seeon-edge-updater.timer"
BINDING_FILE = "/etc/seeon/edge-deploy.env"
CARRIER_EXEC = "/usr/local/libexec/seeon-edge/update-edge.sh"


def _ini_pairs(path: Path) -> list[tuple[str, str, str]]:
    section = ""
    pairs: list[tuple[str, str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        key, value = line.split("=", 1)
        pairs.append((section, key.strip(), value.strip()))
    return pairs


def _ini_map(path: Path) -> dict[str, dict[str, list[str]]]:
    payload: dict[str, dict[str, list[str]]] = {}
    for section, key, value in _ini_pairs(path):
        payload.setdefault(section, {}).setdefault(key, []).append(value)
    return payload


def test_committed_systemd_unit_uses_binding_file_not_host_checkout() -> None:
    assert UNIT.is_file()
    assert TIMER.is_file()
    unit = _ini_map(UNIT)
    timer = _ini_map(TIMER)
    service = unit["Service"]
    unit_text = UNIT.read_text(encoding="utf-8")
    timer_text = TIMER.read_text(encoding="utf-8")

    assert service["Type"] == ["oneshot"]
    assert service["User"] == ["root"]
    assert service["Group"] == ["root"]
    assert service["EnvironmentFile"] == [BINDING_FILE]
    assert not any(value.startswith("-") for value in service["EnvironmentFile"])
    assert service["ExecStart"] == [f"/bin/sh {CARRIER_EXEC}"]
    exec_start = service["ExecStart"][0]
    assert "${" not in exec_start
    assert "$(" not in exec_start
    assert "`" not in exec_start
    assert "/opt/eldercare-fall-ml" not in unit_text
    assert "happy" not in unit_text.lower()
    assert "docker.sock" not in unit_text
    assert "SupplementaryGroups" not in unit_text
    assert "Environment=EDGE_UPDATER_SCRIPT" not in unit_text
    for key in ("User", "Group", "ExecStart", "EnvironmentFile"):
        assert all("docker" not in value.lower() for value in service[key])
    assert timer["Timer"]["Unit"] == ["seeon-edge-updater.service"]
    assert "seeon-edge-updater.service" in timer_text


def test_systemd_unit_syntax_is_valid_or_systemd_is_unavailable() -> None:
    unit_text = UNIT.read_text(encoding="utf-8")
    timer_text = TIMER.read_text(encoding="utf-8")
    assert unit_text.splitlines()[0].startswith("[Unit]")
    assert "[Service]" in unit_text
    assert "[Timer]" in timer_text
    for raw in (*unit_text.splitlines(), *timer_text.splitlines()):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("["):
            continue
        assert "=" in line, raw
    try:
        analyzer = subprocess.run(
            ["systemd-analyze", "verify", str(UNIT), str(TIMER)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("systemd-analyze unavailable on this host")
    if analyzer.returncode == 127 or "No such file" in analyzer.stderr:
        pytest.skip("systemd-analyze unavailable on this host")
    assert analyzer.returncode == 0, analyzer.stderr or analyzer.stdout


def test_readme_documents_root_owned_carrier_not_interactive_docker_group() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "seeon-edge-updater.service" in readme
    assert BINDING_FILE in readme
    assert "User=root" in readme or "root-owned" in readme
    assert "usermod" not in readme
    assert "-G docker" not in readme
    assert "/opt/eldercare-fall-ml" not in readme
    assert "fresh" in readme.lower()
    assert "cutover" in readme.lower()


def test_systemd_environment_cannot_redirect_the_executable(tmp_path: Path) -> None:
    evil = tmp_path / "evil.sh"
    _write_executable(evil, "#!/bin/sh\necho PWNED\n")
    completed = _run_updater(
        tmp_path,
        extra_env={
            "EDGE_UPDATER_SCRIPT": str(evil),
            "EDGE_UPDATER_EXEC": str(evil),
            "EDGE_UPDATER_DRY_RUN": "1",
        },
    )
    combined = completed.stdout + completed.stderr
    assert "PWNED" not in combined
    assert completed.returncode == 0, completed.stderr
    assert "dry run" in completed.stdout.lower()


def _prepare_carrier_tree(tmp_path: Path, *, sealed: bool = True) -> dict[str, Path]:
    deploy = tmp_path / "deploy"
    data = tmp_path / "updater-state"
    deploy.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    env_file = deploy / ".env.edge.prod"
    compose_file = deploy / "compose.edge.yaml"
    if sealed:
        env_file.write_text(
            "\n".join(
                (
                    f"ML_API_IMAGE=ghcr.io/acme/ml-api@{API_NEW}",
                    f"ML_WORKER_IMAGE=ghcr.io/acme/ml-worker@{WORKER_NEW}",
                    "ML_EDGE_VERSION=test-version",
                    "API_EDGE_RELAY_TOKEN=not-secret-for-test",
                    "",
                )
            ),
            encoding="utf-8",
        )
    else:
        _write_case_files(deploy)
    env_file.chmod(0o600)
    compose_file.write_text(
        "\n".join(
            (
                "services:",
                "  ml-api:",
                "    image: ${ML_API_IMAGE}",
                "  ml-worker:",
                "    image: ${ML_WORKER_IMAGE}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return {"deploy": deploy, "data": data, "env": env_file, "compose": compose_file}


def _carrier_env(tree: dict[str, Path], extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "EDGE_HOST_PREFLIGHT": "1",
        "EDGE_CARRIER_UID": str(os.getuid()),
        "EDGE_DEPLOY_ROOT": str(tree["deploy"]),
        "EDGE_UPDATER_DATA_DIR": str(tree["data"]),
        "EDGE_UPDATER_DRY_RUN": "1",
    }
    if extra:
        env.update(extra)
    return env


def test_host_preflight_accepts_root_equivalent_synthetic_carrier(tmp_path: Path) -> None:
    tree = _prepare_carrier_tree(tmp_path)
    completed = _run_updater(
        tmp_path,
        extra_env=_carrier_env(tree),
        env_file=tree["env"],
        compose_file=tree["compose"],
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "dry run" in completed.stdout.lower()


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        ("env-mode", "0600"),
        ("deploy-unwritable", "deploy root"),
        ("missing-deploy", "deploy root"),
        ("state-owner", "updater state"),
        ("carrier-uid", "carrier"),
        ("unsealed-image", "sha256"),
        ("dirty-worktree", "dirty"),
        ("stale-binding", "deploy root"),
    ],
)
def test_host_preflight_rejects_unsafe_carrier_bindings(
    tmp_path: Path, mutate: str, needle: str
) -> None:
    tree = _prepare_carrier_tree(tmp_path)
    extra: dict[str, str] = {}
    if mutate == "env-mode":
        tree["env"].chmod(0o644)
    elif mutate == "deploy-unwritable":
        tree["deploy"].chmod(0o500)
    elif mutate == "missing-deploy":
        extra["EDGE_DEPLOY_ROOT"] = str(tmp_path / "missing-root")
    elif mutate == "state-owner":
        extra["EDGE_UPDATER_DATA_DIR"] = str(tmp_path / "missing-updater-state")
    elif mutate == "carrier-uid":
        extra["EDGE_CARRIER_UID"] = "0" if os.getuid() != 0 else "1"
    elif mutate == "unsealed-image":
        tree = _prepare_carrier_tree(tmp_path, sealed=False)
        tree["env"].chmod(0o600)
    elif mutate == "dirty-worktree":
        subprocess.run(["git", "init", "-q"], cwd=tree["deploy"], check=True)
        (tree["deploy"] / "dirty").write_text("stale", encoding="utf-8")
    elif mutate == "stale-binding":
        extra["EDGE_DEPLOY_ROOT"] = str(tmp_path / "old-facility")
    completed = _run_updater(
        tmp_path,
        extra_env=_carrier_env(tree, extra),
        env_file=tree["env"],
        compose_file=tree["compose"],
    )
    combined = (completed.stdout + completed.stderr).lower()
    assert completed.returncode != 0
    assert needle.lower() in combined
    assert "not-secret-for-test" not in combined
    if mutate == "deploy-unwritable":
        tree["deploy"].chmod(0o700)


def test_host_preflight_rejects_binding_path_injection(tmp_path: Path) -> None:
    tree = _prepare_carrier_tree(tmp_path)
    completed = _run_updater(
        tmp_path,
        extra_env=_carrier_env(
            tree,
            {"EDGE_DEPLOY_ROOT": str(tree["deploy"]) + "; touch /tmp/pwned"},
        ),
        env_file=tree["env"],
        compose_file=tree["compose"],
    )
    assert completed.returncode != 0
    assert "PWNED" not in completed.stdout + completed.stderr
    assert not Path("/tmp/pwned").exists() or "pwned" not in completed.stdout.lower()


def test_digest_rollback_still_works_when_host_preflight_is_disabled(
    tmp_path: Path,
) -> None:
    completed = _run_updater(tmp_path, scenario="rollback")
    env_text = (tmp_path / "case" / ".env.edge.prod").read_text(encoding="utf-8")
    assert completed.returncode != 0
    assert f"ML_API_IMAGE=ghcr.io/acme/ml-api@{API_OLD}" in env_text
    assert f"ML_WORKER_IMAGE=ghcr.io/acme/ml-worker@{WORKER_OLD}" in env_text
