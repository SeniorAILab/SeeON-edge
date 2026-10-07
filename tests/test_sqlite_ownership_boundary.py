from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SLOT_ROOT = ROOT / "worker"

_FORBIDDEN_MODULES = ("sqlite3", "backend.app.edge_db")
_FORBIDDEN_DYNAMIC = ("__import__", "import_module")
_FORBIDDEN_LITERALS = (".sqlite3", ".sqlite", ".db3")


@dataclass(frozen=True, slots=True)
class Violation:
    module: str
    kind: str

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.module}::{self.kind}"


def _is_forbidden_module(name: str | None) -> str | None:
    if not name:
        return None
    for forbidden in _FORBIDDEN_MODULES:
        if name == forbidden or name.startswith(f"{forbidden}."):
            return forbidden
    return None


class _SlotVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.kinds: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if forbidden := _is_forbidden_module(alias.name):
                self.kinds.add(f"import:{forbidden}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if forbidden := _is_forbidden_module(node.module):
            self.kinds.add(f"from-import:{forbidden}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name in _FORBIDDEN_DYNAMIC:
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if forbidden := _is_forbidden_module(argument.value):
                        self.kinds.add(f"dynamic-import:{forbidden}")
        if name == "connect" and isinstance(func, ast.Attribute):
            base = func.value
            if isinstance(base, ast.Name) and base.id.endswith(("sqlite3", "sqlite")):
                self.kinds.add("call:sqlite3.connect")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            lowered = node.value.lower()
            for literal in _FORBIDDEN_LITERALS:
                if literal in lowered:
                    self.kinds.add(f"literal:{literal}")
        self.generic_visit(node)


def scan_module(path: Path) -> frozenset[str]:
    visitor = _SlotVisitor()
    visitor.visit(ast.parse(path.read_text(), filename=str(path)))
    return frozenset(visitor.kinds)


def scan_slot() -> frozenset[Violation]:
    found: set[Violation] = set()
    for path in sorted(SLOT_ROOT.rglob("*.py")):
        module = path.relative_to(ROOT).as_posix()
        for kind in scan_module(path):
            found.add(Violation(module, kind))
    return frozenset(found)


def _load_baseline() -> frozenset[Violation]:
    text = (Path(__file__).parent / "sqlite_ownership_baseline.txt").read_text()
    entries: set[Violation] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        module, _, kind = stripped.partition("::")
        entries.add(Violation(module, kind))
    return frozenset(entries)


BASELINE = _load_baseline()


def test_the_baseline_names_real_modules_and_has_no_wildcard_exemption() -> None:
    for violation in BASELINE:
        assert "*" not in violation.module, f"wildcard exemption: {violation.module}"
        assert (ROOT / violation.module).is_file(), f"stale entry: {violation.module}"
        assert violation.module.startswith("worker/")


def test_no_new_sqlite_reach_is_introduced_into_the_runtime_slot() -> None:
    found = scan_slot()
    added = sorted(str(v) for v in found - BASELINE)
    assert not added, (
        "new SQLite reach in the inference-runtime slot, which ADR 0005 forbids. "
        "The slot may keep only media files, a publish-once delivery queue, a "
        "bounded config read cache, a media-integrity sidecar, lock inodes, and "
        f"startup-purged scratch: {added}"
    )


def test_the_baseline_does_not_carry_entries_that_are_already_removed() -> None:
    found = scan_slot()
    stale = sorted(str(v) for v in BASELINE - found)
    assert not stale, (
        "these baseline entries no longer exist and must be deleted so the "
        f"ratchet keeps its teeth: {stale}"
    )


def test_backend_only_sqlite_cutover_is_atomic() -> None:
    database_package = ROOT / "backend" / "app" / "edge_db"
    assert database_package.is_dir()
    assert not (ROOT / "shared" / "edge_db").exists()
    assert scan_slot() == frozenset()

    services = yaml.load((ROOT / "compose.edge.yaml").read_text(), Loader=yaml.BaseLoader)[
        "services"
    ]
    assert "edge-state:/var/lib/seeon-state" in services["ml-api"]["volumes"]
    assert "worker-local-state:/var/lib/seeon-state" in services["ml-worker"]["volumes"]
    assert not any(volume.startswith("edge-state:") for volume in services["ml-worker"]["volumes"])

    from backend.app.edge_db.migration.compatibility import (
        CURRENT_SCHEMA_RANGE,
        SCHEMA_19_IDENTITY,
    )
    from shared.release_identity import EDGE_DATABASE_SCHEMA_VERSION

    assert (
        CURRENT_SCHEMA_RANGE.minimum
        == CURRENT_SCHEMA_RANGE.maximum
        == EDGE_DATABASE_SCHEMA_VERSION
        == SCHEMA_19_IDENTITY[0]
    ), (
        "the backend compatibility range, shared release identity, and schema-19 "
        "ledger identity must identify the same schema release"
    )

    backend_image = (ROOT / "Dockerfile.backend").read_text()
    runtime_image = (ROOT / "Dockerfile.edge").read_text()
    assert "COPY scripts/ops" in backend_image, (
        "the API image does not ship scripts/ops, so the documented ops "
        "commands do not exist in the image the runbooks tell an operator to use"
    )
    assert "COPY backend" not in runtime_image, (
        "the runtime image copies the backend package, which would carry the "
        "database back into the inference-runtime slot"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import sqlite3", "import:sqlite3"),
        ("import sqlite3 as db", "import:sqlite3"),
        ("from sqlite3 import connect", "from-import:sqlite3"),
        (
            "from backend.app.edge_db import open_runtime_database",
            "from-import:backend.app.edge_db",
        ),
        ("import backend.app.edge_db as edb", "import:backend.app.edge_db"),
        ("def f():\n    import sqlite3", "import:sqlite3"),
        ("def f():\n    from backend.app.edge_db import x", "from-import:backend.app.edge_db"),
        ("__import__('sqlite3')", "dynamic-import:sqlite3"),
        (
            "import importlib\nimportlib.import_module('backend.app.edge_db')",
            "dynamic-import:backend.app.edge_db",
        ),
        ("import sqlite3\nsqlite3.connect('x')", "call:sqlite3.connect"),
        ("PATH = '/var/lib/edge.sqlite3'", "literal:.sqlite3"),
        ("URI = 'file:edge.db3?mode=ro'", "literal:.db3"),
    ],
)
def test_the_scanner_catches_every_evasion_shape(
    source: str, expected: str, tmp_path: Path
) -> None:
    module = tmp_path / "candidate.py"
    module.write_text(source)
    assert expected in scan_module(module)


def test_the_scanner_does_not_flag_unrelated_code(tmp_path: Path) -> None:
    module = tmp_path / "clean.py"
    module.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from contracts.event import EventPayload\n"
        "def write(p: Path) -> None:\n"
        "    p.write_text(json.dumps({'ok': True}))\n"
    )
    assert scan_module(module) == frozenset()


def test_the_runtime_slot_carries_no_operational_sqlite_cli() -> None:
    replay_package = ROOT / "worker" / "replay"
    assert replay_package.is_dir(), "worker/replay was removed; ADR 0007 assumes it is alive"

    offenders = sorted(
        path.relative_to(ROOT).as_posix()
        for path in replay_package.rglob("*.py")
        if "sqlite3" in path.read_text(encoding="utf-8")
        or "edge_db" in path.read_text(encoding="utf-8")
    )
    assert not offenders, (
        f"{offenders} reach for a database from the inference-runtime slot. "
        "If replay needs durable input, it needs a backend-owned trace-ingest "
        "surface (ADR 0007), not database access here."
    )


def test_retired_qa_persistence_does_not_return_sqlite_to_the_runtime_slot() -> None:
    assert not (ROOT / "backend" / "app" / "features" / "qa").exists()
    replay_package = ROOT / "worker" / "replay"
    assert replay_package.is_dir()
    offenders = sorted(
        path.relative_to(ROOT).as_posix()
        for path in replay_package.rglob("*.py")
        if "sqlite3" in path.read_text(encoding="utf-8")
        or "edge_db" in path.read_text(encoding="utf-8")
    )
    assert not offenders
