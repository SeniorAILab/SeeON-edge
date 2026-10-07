import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_boundaries import (
    BARE_EXCEPT,
    BLE_NOQA,
    BROAD_EXCEPT,
    SUPPRESS_BROAD,
    UNPARSEABLE,
    check_source,
    in_scope,
    select_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_boundaries.py"
SAMPLE = Path("worker/sample.py")

COPIED_IN_71978E50 = """\
def drain(self, drained, batch):
    try:
        result = self._client.post_batch(batch)
    except Exception:  # noqa: BLE001 - transport exceptions cannot kill the drain thread
        self._failed(drained, DeliveryFailure(DeliveryDisposition.RETRY, "TRANSPORT_EXCEPTION"))
        return False
    return result
"""


def kinds(source: str) -> list[tuple[int, str]]:
    return [(finding.line, finding.kind) for finding in check_source(SAMPLE, source)]


def test_flags_the_catch_copied_into_the_exporter_in_71978e50() -> None:
    assert kinds(COPIED_IN_71978E50) == [(4, BLE_NOQA), (4, BROAD_EXCEPT)]


def test_advice_names_the_helper_to_use() -> None:
    rendered = [finding.render() for finding in check_source(SAMPLE, COPIED_IN_71978E50)]
    assert rendered[1].startswith("worker/sample.py:4: BROAD_EXCEPT: ")
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
def test_flags_every_broad_shape(source: str, expected: list[tuple[int, str]]) -> None:
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
def test_ignores_narrow_catches_and_strings(source: str) -> None:
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
