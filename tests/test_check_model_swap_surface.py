import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_model_swap_surface import (
    ALLOWED_SELECTION_KEYS,
    CEREMONY_LITERAL,
    SELECTION_KEYS_DRIFT,
    STATUS_COMPARISON,
    UNPARSEABLE,
    ceremony_findings,
    in_scope,
    selection_keys_findings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_model_swap_surface.py"
SAMPLE = Path("contracts/model_selection.py")


def kinds(source: str) -> list[tuple[int, str]]:
    return [(finding.line, finding.kind) for finding in ceremony_findings(SAMPLE, source)]


def test_clean_source_passes() -> None:
    assert kinds("x = 1\n\n\ndef f(a: int) -> int:\n    return a + 1\n") == []


@pytest.mark.parametrize("literal", ["green", "verdict"])
def test_ceremony_literals_fail(literal: str) -> None:
    assert kinds(f'x = "{literal}"\n') == [(1, CEREMONY_LITERAL)]


@pytest.mark.parametrize(
    "expression",
    ["status == 1", "status != 1", "receipt.status == 'x'", "'x' != receipt.status"],
)
def test_status_comparison_fails(expression: str) -> None:
    findings = kinds(f"def f(status, receipt):\n    return {expression}\n")
    assert (2, STATUS_COMPARISON) in findings


def test_unrelated_comparison_passes() -> None:
    assert kinds("def f(a, b):\n    return a == b\n") == []


def test_unparseable_source_fails() -> None:
    assert kinds("def (:\n") == [(0, UNPARSEABLE)]


def test_matching_selection_keys_pass() -> None:
    assert selection_keys_findings(ALLOWED_SELECTION_KEYS) == []


@pytest.mark.parametrize(
    "current",
    [ALLOWED_SELECTION_KEYS | {"receipt"}, ALLOWED_SELECTION_KEYS - {"threshold_source"}],
)
def test_selection_keys_drift_fails(current: frozenset[str]) -> None:
    assert [finding.kind for finding in selection_keys_findings(current)] == [SELECTION_KEYS_DRIFT]


def test_scope_is_contract_and_provenance_only() -> None:
    assert in_scope(Path("contracts/model_selection.py"))
    assert in_scope(Path("worker/runtime/provenance/model_bundle.py"))
    assert not in_scope(Path("worker/runtime/worker.py"))
    assert not in_scope(Path("worker/runtime/provenance/notes.md"))


def test_script_passes_on_the_repo() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
