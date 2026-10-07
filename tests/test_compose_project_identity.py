from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_COMPOSE = _ROOT / "compose.edge.yaml"
_PROD_EXAMPLE = _ROOT / ".env.edge.prod.example"


def test_the_project_name_is_required_and_never_defaulted() -> None:
    compose = _COMPOSE.read_text(encoding="utf-8")

    match = re.search(
        r"^name:\s*\$\{COMPOSE_PROJECT_NAME(?P<expansion>[^}]*)\}", compose, re.MULTILINE
    )
    assert match, "compose.edge.yaml does not pin the project name to COMPOSE_PROJECT_NAME"

    expansion = match.group("expansion")
    assert expansion.startswith(":?"), (
        "COMPOSE_PROJECT_NAME is defaulted rather than required; a default binds "
        "a directory-derived volume when the operator forgets to set it"
    )


def test_the_production_example_declares_the_project_name() -> None:
    example = _PROD_EXAMPLE.read_text(encoding="utf-8")

    assert "COMPOSE_PROJECT_NAME=" in example, (
        ".env.edge.prod.example does not declare COMPOSE_PROJECT_NAME, so a "
        "deployment copied from it cannot start and the operator gets no guidance "
        "on which value is correct"
    )


def test_the_state_volume_is_not_declared_external_or_renamed() -> None:
    compose = _COMPOSE.read_text(encoding="utf-8")

    volumes_section = compose.split("\nvolumes:", 1)
    assert len(volumes_section) == 2, "compose.edge.yaml declares no volumes section"

    body = volumes_section[1]
    edge_state = re.search(
        r"^  edge-state:(?P<body>.*?)(?=^  \S|\Z)", body, re.MULTILINE | re.DOTALL
    )
    assert edge_state, "edge-state volume is no longer declared"

    declaration = edge_state.group("body")
    assert "external" not in declaration, (
        "edge-state is declared external, which decouples it from the project "
        "name and makes the bound volume ambiguous again"
    )
    assert "name:" not in declaration, (
        "edge-state carries an explicit name, which overrides the project-derived "
        "identity the required COMPOSE_PROJECT_NAME exists to establish"
    )
