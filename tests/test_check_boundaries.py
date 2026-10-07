import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_boundaries import (
    ADVICE,
    BARE_EXCEPT,
    BLANKET_NOQA,
    BLE_NOQA,
    BROAD_EXCEPT,
    FILE_NOQA,
    SUPPRESS_BROAD,
    UNPARSEABLE,
    check_source,
    in_scope,
    load_baseline,
    select_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_boundaries.py"
SAMPLE = Path("worker/sample.py")

COPIED_COMMIT = "71978e50"
COPIED_PATH = "worker/pipeline/diagnostics/exporter.py"
COPIED_REASON = "transport exceptions cannot kill the drain thread"


def copied_exporter_source() -> str:
    result = subprocess.run(
        ["git", "show", f"{COPIED_COMMIT}:{COPIED_PATH}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"{COPIED_COMMIT} is not in this checkout (shallow clone)")
    return result.stdout


RECORDED_BASELINE = {
    "BLANKET_NOQA": 0,
    "BLE_NOQA": 51,
    "BROAD_EXCEPT": 135,
    "FILE_NOQA": 0,
    "SUPPRESS_BROAD": 4,
}


def kinds_at(source: str, path: Path) -> list[tuple[int, str]]:
    return [(finding.line, finding.kind) for finding in check_source(path, source)]


def kinds(source: str) -> list[tuple[int, str]]:
    return kinds_at(source, SAMPLE)


def test_flags_the_catch_copied_into_the_exporter_in_71978e50() -> None:
    source = copied_exporter_source()
    [line] = [
        number for number, text in enumerate(source.splitlines(), start=1) if COPIED_REASON in text
    ]
    found = kinds_at(source, Path(COPIED_PATH))
    assert (line, BLE_NOQA) in found
    assert (line, BROAD_EXCEPT) in found


def test_advice_names_the_helper_to_use() -> None:
    source = "try:\n    pass\nexcept Exception:  # noqa: BLE001\n    pass\n"
    rendered = [finding.render() for finding in check_source(SAMPLE, source)]
    assert rendered[1].startswith("worker/sample.py:3: BROAD_EXCEPT: ")
    for helper in (
        "isolate()",
        "degrade(message=...)",
        "attempt_delivery()",
        "probe()",
        "cleanup_on_failure()",
        "translate()",
        "root_sink()",
    ):
        assert helper in rendered[1]
    assert "shared.boundary helper" in rendered[0]


def test_flags_a_blanket_noqa_on_a_broad_except_line() -> None:
    source = "try:\n    pass\nexcept Exception:  # noqa\n    pass\n"
    assert kinds(source) == [(3, BLANKET_NOQA), (3, BROAD_EXCEPT)]


def test_a_blanket_noqa_away_from_an_except_line_is_not_a_boundary_finding() -> None:
    assert kinds("import os  # noqa\n") == []


@pytest.mark.parametrize(
    "directive",
    ["# ruff: noqa", "# ruff: noqa: BLE001", "# ruff: noqa: E501, BLE001", "#ruff:noqa"],
)
def test_flags_file_level_ruff_noqa_that_covers_blind_except(directive: str) -> None:
    assert kinds(f"{directive}\nx = 1\n") == [(1, FILE_NOQA)]


def test_file_level_ruff_noqa_for_other_codes_is_ignored() -> None:
    assert kinds("# ruff: noqa: E501\nx = 1\n") == []


def test_known_limit_an_aliased_exception_name_is_not_flagged() -> None:
    source = "from builtins import Exception as Broad\ntry:\n    pass\nexcept Broad:\n    pass\n"
    assert kinds(source) == []


def test_known_limit_a_starred_tuple_of_broad_types_is_not_flagged() -> None:
    source = "BROAD = (Exception,)\ntry:\n    pass\nexcept (*BROAD,):\n    pass\n"
    assert kinds(source) == []


def test_known_limit_an_aliased_suppress_is_not_flagged() -> None:
    source = "from contextlib import suppress as quiet\nwith quiet(Exception):\n    pass\n"
    assert kinds(source) == []


def test_baseline_file_records_every_kind_and_the_total() -> None:
    baseline = load_baseline()
    assert set(baseline) <= set(ADVICE)
    assert baseline == RECORDED_BASELINE


def test_cli_reports_drift_against_the_baseline() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECKER)], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert "baseline" in result.stdout.splitlines()[-1]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("try:\n    pass\nexcept:\n    raise\n", [(3, BARE_EXCEPT)]),
        ("try:\n    pass\nexcept BaseException:\n    raise\n", [(3, BROAD_EXCEPT)]),
        ("try:\n    pass\nexcept (OSError, Exception):\n    pass\n", [(3, BROAD_EXCEPT)]),
        ("try:\n    pass\nexcept builtins.Exception:\n    pass\n", [(3, BROAD_EXCEPT)]),
        (
            "import contextlib\nwith contextlib.suppress(Exception):\n    pass\n",
            [(2, SUPPRESS_BROAD)],
        ),
        (
            "from contextlib import suppress\nwith suppress(OSError, BaseException):\n    pass\n",
            [(2, SUPPRESS_BROAD)],
        ),
    ],
)
def test_flags_the_listed_broad_shapes(source: str, expected: list[tuple[int, str]]) -> None:
    assert kinds(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "try:\n    pass\nexcept OSError:\n    pass\n",
        "try:\n    pass\nexcept (TimeoutError, http.client.HTTPException):\n    pass\n",
        "import contextlib\nwith contextlib.suppress(FileNotFoundError):\n    pass\n",
        "x = 'except Exception:  # noqa: BLE001'\n",
        "y = 1  # noqa: E501\n",
    ],
)
def test_ignores_the_listed_narrow_catches_and_strings(source: str) -> None:
    assert kinds(source) == []


def test_unparseable_source_is_reported() -> None:
    assert kinds("def broken(:\n") == [(0, UNPARSEABLE)]


def test_scope_excludes_tests_and_the_boundary_owner() -> None:
    assert in_scope(Path("worker/runtime/worker.py"))
    assert in_scope(Path("backend/app/lifespan.py"))
    assert in_scope(Path("scripts/check_no_comments.py"))
    assert not in_scope(Path("shared/boundary/__init__.py"))
    assert not in_scope(Path("tests/test_check_boundaries.py"))
    assert not in_scope(Path("front/vite.config.ts"))
    assert select_paths([Path("tests/a.py"), Path("worker/b.py"), Path("worker/c.md")]) == [
        Path("worker/b.py")
    ]


def test_cli_is_report_only_and_exits_zero_with_findings() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECKER)], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert ": BROAD_EXCEPT: " in result.stdout
    assert "check_boundaries (report-only):" in result.stdout.splitlines()[-1]
    assert "shared/boundary/" not in result.stdout
    assert "tests/test_" not in result.stdout
