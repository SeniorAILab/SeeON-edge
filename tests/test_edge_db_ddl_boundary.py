from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DDL_OWNER_MODULES = ("tests_support.sqlite_source",)
_RUNTIME_PACKAGES = ("backend", "worker", "contracts", "shared")
_SOURCE_FIXTURE_PACKAGE = "tests_support"


def _modules_loaded_by(
    import_target: str, watched: tuple[str, ...] = _DDL_OWNER_MODULES
) -> frozenset[str]:
    probe = (
        "import sys\n"
        f"import {import_target}\n"
        "import json\n"
        f"loaded = [m for m in {watched!r} if m in sys.modules]\n"
        "print(json.dumps(loaded))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    return frozenset(json.loads(completed.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("entry_point", ["backend.app.main", "worker.runtime.worker"])
def test_runtime_entry_points_do_not_load_sqlite3(entry_point: str) -> None:
    assert _modules_loaded_by(entry_point, watched=("sqlite3",)) == frozenset()


def test_the_migration_snapshot_loads_sqlite3() -> None:
    loaded = _modules_loaded_by("backend.app.edge_db.migration.snapshot", watched=("sqlite3",))
    assert loaded == frozenset({"sqlite3"})


def test_compatibility_import_does_not_reach_the_sqlite_ddl_owner() -> None:
    assert _modules_loaded_by("backend.app.edge_db.migration.compatibility") == frozenset()


def test_schema18_manifest_import_does_not_reach_the_sqlite_ddl_owner() -> None:
    assert _modules_loaded_by("backend.app.edge_db.migration.schema18_manifest") == frozenset()


def test_sqlite_source_fixture_reaches_the_ddl_owner() -> None:
    assert _modules_loaded_by("tests_support.sqlite_source") == frozenset(_DDL_OWNER_MODULES)


def _names_the_fixture(name: str | None) -> bool:
    return name is not None and (
        name == _SOURCE_FIXTURE_PACKAGE or name.startswith(f"{_SOURCE_FIXTURE_PACKAGE}.")
    )


def _fixture_references(source: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            found.extend(f"import {name}" for name in names if _names_the_fixture(name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if _names_the_fixture(node.module):
                found.append(f"from {node.module}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _names_the_fixture(node.value):
                found.append(f"string {node.value}")
    return found


def test_runtime_packages_do_not_reach_the_sqlite_source_fixture() -> None:
    offenders = {
        path.relative_to(_ROOT).as_posix(): found
        for package in _RUNTIME_PACKAGES
        for path in sorted((_ROOT / package).rglob("*.py"))
        if (found := _fixture_references(path.read_text(encoding="utf-8")))
    }

    assert offenders == {}


def test_fixture_scan_sees_static_and_dynamic_imports() -> None:
    migration_support = (_ROOT / "tests_support" / "postgres_migration.py").read_text(
        encoding="utf-8"
    )
    assert "from tests_support.sqlite_source" in _fixture_references(migration_support)
    assert sorted(
        _fixture_references(
            "def load():\n"
            "    import tests_support.sqlite_source as source\n"
            "    return importlib.import_module('tests_support')\n"
            "NEIGHBOUR = 'tests_supportive'\n"
        )
    ) == ["import tests_support.sqlite_source", "string tests_support"]
