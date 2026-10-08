import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_private_test_only.py"
BASELINE = "scripts/private_test_only_baseline.json"

WORKER = (
    "def _only_tests():\n    return 1\n\n"
    "def _helper():\n    return 2\n\n"
    "def public():\n    return _helper()\n"
)
TESTS = "from worker.mod import _only_tests\n\ndef test_it():\n    assert _only_tests() == 1\n"


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def make_repo(tmp_path: Path, baseline: list[str]) -> Path:
    (tmp_path / "worker").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "worker/mod.py").write_text(WORKER, encoding="utf-8")
    (tmp_path / "tests/test_mod.py").write_text(TESTS, encoding="utf-8")
    (tmp_path / BASELINE).write_text(json.dumps({"offenders": baseline}), encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(
        tmp_path,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.test",
        "commit",
        "-qm",
        "base",
    )
    return tmp_path


def test_a_private_function_only_tests_call_is_detected(tmp_path: Path) -> None:
    result = run(make_repo(tmp_path, []))

    assert result.returncode == 1
    assert "worker/mod.py::_only_tests: not in scripts/private_test_only_baseline.json" in (
        result.stdout
    )
    assert "_helper" not in result.stdout


def test_a_private_helper_used_by_production_code_passes(tmp_path: Path) -> None:
    root = make_repo(tmp_path, ["worker/mod.py::_only_tests"])

    result = run(root)

    assert result.returncode == 0, result.stdout


def test_a_fixed_offender_must_leave_the_baseline(tmp_path: Path) -> None:
    root = make_repo(tmp_path, ["worker/mod.py::_only_tests", "worker/mod.py::_gone"])

    result = run(root)

    assert result.returncode == 1
    assert "worker/mod.py::_gone: fixed or gone, remove it" in result.stdout


def test_the_baseline_cannot_gain_an_entry_against_the_base_commit(tmp_path: Path) -> None:
    root = make_repo(tmp_path, ["worker/mod.py::_only_tests"])
    (root / "worker/mod.py").write_text(WORKER + "\ndef _more():\n    return 1\n", encoding="utf-8")
    (root / "tests/test_mod.py").write_text(
        TESTS + "\ndef test_other():\n    assert _more() == 1\n", encoding="utf-8"
    )
    (root / BASELINE).write_text(
        json.dumps({"offenders": ["worker/mod.py::_more", "worker/mod.py::_only_tests"]}),
        encoding="utf-8",
    )

    result = run(root, "--against", "HEAD")

    assert result.returncode == 1
    assert "worker/mod.py::_more: scripts/private_test_only_baseline.json gained it" in (
        result.stdout
    )


def test_the_real_tree_passes_with_its_baseline() -> None:
    result = run(REPO_ROOT)

    assert result.returncode == 0, result.stdout
