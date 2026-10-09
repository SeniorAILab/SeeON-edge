import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_layers.py"
LINT_IMPORTS = Path(sys.executable).parent / "lint-imports"
RUFF = Path(sys.executable).parent / "ruff"
F = "backend.app.features"

SHARED_HTTP = "backend base code reaches HTTP frameworks only inside backend.app.shared.http"
SHARED_HTTP_IMPORTERS = (
    "backend.app.shared.http is imported only by features, routes and the app entry"
)
LAYERS = "backend feature layers run controller -> service -> repository"
SKIP = "backend controllers reach repositories only through a service"
HTTP = "backend services and repositories do not import HTTP frameworks or controllers"
DRIVERS = "backend psycopg stays in repositories"
INDEPENDENCE = "backend features meet only service to service"
COMPOSITION = "backend features do not import the composition root"
ROUTES = "backend routes reach features only through a service"
ACYCLIC = "backend features do not depend on each other in a cycle"
SHARED_PYDANTIC = "shared folders and contracts do not import pydantic"
LAYER_CONTRACTS = (
    SHARED_HTTP,
    SHARED_HTTP_IMPORTERS,
    LAYERS,
    SKIP,
    HTTP,
    DRIVERS,
    INDEPENDENCE,
    COMPOSITION,
    ROUTES,
    ACYCLIC,
    SHARED_PYDANTIC,
)

COMPOSITION_ROOT = (
    "main",
    "lifespan",
    "postgres_root",
    "audit_lifecycle",
    "clip_catalog_lifecycle",
    "routes",
)


def feature_files(name: str) -> dict[str, str]:
    base = f"backend/app/features/{name}"
    return {
        f"{base}/__init__.py": "",
        f"{base}/controller/__init__.py": "",
        f"{base}/controller/api.py": f"from {F}.{name}.service import logic\n",
        f"{base}/service/__init__.py": "",
        f"{base}/service/logic.py": (
            f"from backend.app.core import config\nfrom {F}.{name}.repository import rows\n"
        ),
        f"{base}/repository/__init__.py": "",
        f"{base}/repository/rows.py": "import psycopg\n",
    }


def clean_tree() -> dict[str, str]:
    files = {
        "backend/__init__.py": "",
        "backend/app/__init__.py": "",
        "backend/app/core/__init__.py": "",
        "backend/app/core/config.py": "import pydantic\nimport pydantic_settings\n",
        "backend/app/edge_db/__init__.py": "",
        "backend/app/features/__init__.py": "",
        "backend/app/shared/__init__.py": "",
        "backend/app/shared/http/__init__.py": "",
        "contracts/__init__.py": "",
        "shared/__init__.py": "",
    }
    files.update({f"backend/app/{name}.py": "" for name in COMPOSITION_ROOT})
    files.update(feature_files("alpha"))
    files.update(feature_files("beta"))
    return files


def write_tree(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def toml_value(value: object) -> str:
    return json.dumps(value)


def real_layer_contracts() -> list[dict[str, object]]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    contracts = data["tool"]["importlinter"]["contracts"]
    chosen = [c for c in contracts if c["name"] in LAYER_CONTRACTS]
    assert [c["name"] for c in chosen] == list(LAYER_CONTRACTS)
    return chosen


def lint_config(root: Path) -> Path:
    lines = [
        "[tool.importlinter]",
        'root_packages = ["backend", "contracts", "shared"]',
        "include_external_packages = true",
    ]
    for contract in real_layer_contracts():
        lines.append("\n[[tool.importlinter.contracts]]")
        lines.extend(
            f"{key} = {toml_value(value)}"
            for key, value in contract.items()
            if key != "unmatched_ignore_imports_alerting"
        )
        lines.append('unmatched_ignore_imports_alerting = "none"')
    path = root / "layers.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def lint(root: Path, changes: dict[str, str]) -> str:
    write_tree(root, {**clean_tree(), **changes})
    result = subprocess.run(
        [str(LINT_IMPORTS), "--config", str(lint_config(root)), "--no-cache"],
        cwd=root,
        env={"PYTHONPATH": str(root), "PATH": str(LINT_IMPORTS.parent), "COLUMNS": "400"},
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


def test_clean_layered_features_keep_every_layer_contract(tmp_path: Path) -> None:
    output = lint(tmp_path, {})
    assert all(f"{name} KEPT" in output for name in LAYER_CONTRACTS), output
    assert "BROKEN" not in output, output


ALPHA = "backend/app/features/alpha"
BASE_MODEL = "from pydantic import BaseModel\n\n\nclass Wire(BaseModel):\n    x: int\n"
LEGACY = "backend/app/features/legacy"

MUTATIONS = {
    "reverse_call_repository_to_service": (
        {f"{ALPHA}/repository/rows.py": f"from {F}.alpha.service import logic\n"},
        LAYERS,
    ),
    "reverse_call_service_to_controller": (
        {f"{ALPHA}/service/logic.py": f"from {F}.alpha.controller import api\n"},
        LAYERS,
    ),
    "layer_skip_controller_to_repository": (
        {f"{ALPHA}/controller/api.py": f"from {F}.alpha.repository import rows\n"},
        SKIP,
    ),
    "layer_skip_controller_to_edge_db": (
        {f"{ALPHA}/controller/api.py": "from backend.app import edge_db\n"},
        SKIP,
    ),
    "cross_feature_service_to_repository": (
        {f"{ALPHA}/service/logic.py": f"from {F}.beta.repository import rows\n"},
        INDEPENDENCE,
    ),
    "cross_feature_controller_to_controller": (
        {f"{ALPHA}/controller/api.py": f"from {F}.beta.controller import api\n"},
        INDEPENDENCE,
    ),
    "fastapi_in_service": ({f"{ALPHA}/service/logic.py": "import fastapi\n"}, HTTP),
    "controller_import_in_service": (
        {f"{ALPHA}/service/logic.py": f"from {F}.alpha.controller import api\n"},
        HTTP,
    ),
    "starlette_in_repository": ({f"{ALPHA}/repository/rows.py": "import starlette\n"}, HTTP),
    "psycopg_in_service": ({f"{ALPHA}/service/logic.py": "import psycopg\n"}, DRIVERS),
    "psycopg_pool_in_controller": (
        {f"{ALPHA}/controller/api.py": "import psycopg_pool\n"},
        DRIVERS,
    ),
    "feature_imports_composition_root": (
        {f"{ALPHA}/service/logic.py": "from backend.app import lifespan\n"},
        COMPOSITION,
    ),
    "routes_import_a_repository": (
        {"backend/app/routes.py": f"from {F}.alpha.repository import rows\n"},
        ROUTES,
    ),
    "fastapi_through_a_shared_http_adapter": (
        {
            "backend/app/shared/http/adapter.py": "from fastapi import Request\n",
            f"{ALPHA}/service/logic.py": "from backend.app.shared.http import adapter\n",
        },
        HTTP,
    ),
    "fastapi_in_backend_shared_outside_http": (
        {"backend/app/shared/values.py": "from fastapi import Request\n"},
        SHARED_HTTP,
    ),
    "starlette_in_edge_db": ({"backend/app/edge_db/rows.py": "import starlette\n"}, SHARED_HTTP),
    "fastapi_in_core": ({"backend/app/core/web.py": "import fastapi\n"}, SHARED_HTTP),
    "backend_shared_imports_shared_http": (
        {
            "backend/app/shared/http/adapter.py": "import starlette\n",
            "backend/app/shared/values.py": "from backend.app.shared.http import adapter\n",
        },
        SHARED_HTTP_IMPORTERS,
    ),
    "edge_db_imports_shared_http": (
        {
            "backend/app/shared/http/adapter.py": "",
            "backend/app/edge_db/rows.py": "from backend.app.shared.http import adapter\n",
        },
        SHARED_HTTP_IMPORTERS,
    ),
    "basemodel_in_top_level_shared_used_by_a_service": (
        {
            "shared/wire_models.py": BASE_MODEL,
            f"{ALPHA}/service/logic.py": "from shared.wire_models import Wire\n",
        },
        SHARED_PYDANTIC,
    ),
    "basemodel_in_contracts_used_by_a_service": (
        {
            "contracts/wire_models.py": BASE_MODEL,
            f"{ALPHA}/service/logic.py": "from contracts.wire_models import Wire\n",
        },
        SHARED_PYDANTIC,
    ),
    "basemodel_in_backend_shared_used_by_a_service": (
        {
            "backend/app/shared/wire_models.py": BASE_MODEL,
            f"{ALPHA}/service/logic.py": "from backend.app.shared.wire_models import Wire\n",
        },
        SHARED_PYDANTIC,
    ),
    "basemodel_in_edge_db_used_by_a_repository": (
        {
            "backend/app/edge_db/rows.py": BASE_MODEL,
            f"{ALPHA}/repository/rows.py": "from backend.app.edge_db.rows import Wire\n",
        },
        SHARED_PYDANTIC,
    ),
    "pydantic_settings_in_edge_db": (
        {"backend/app/edge_db/settings.py": "from pydantic_settings import BaseSettings\n"},
        SHARED_PYDANTIC,
    ),
    "service_cycle_between_features": (
        {
            f"{ALPHA}/service/logic.py": f"from {F}.beta.service import logic\n",
            "backend/app/features/beta/service/logic.py": f"from {F}.alpha.service import logic\n",
        },
        ACYCLIC,
    ),
}


def test_shared_http_adapters_may_import_http_frameworks_for_controllers(tmp_path: Path) -> None:
    output = lint(
        tmp_path,
        {
            "backend/app/shared/http/adapter.py": "import fastapi\nimport starlette\n",
            f"{ALPHA}/controller/api.py": (
                "from backend.app.shared.http import adapter\n"
                f"from {F}.alpha.service import logic\n"
            ),
        },
    )
    assert f"{SHARED_HTTP} KEPT" in output, output
    assert f"{SHARED_HTTP_IMPORTERS} KEPT" in output, output
    assert f"{HTTP} KEPT" in output, output


def test_edge_db_may_reach_pydantic_through_core_config(tmp_path: Path) -> None:
    output = lint(
        tmp_path, {"backend/app/edge_db/pool.py": "from backend.app.core import config\n"}
    )
    assert f"{SHARED_PYDANTIC} KEPT" in output, output


@pytest.mark.parametrize("case", sorted(MUTATIONS))
def test_layer_violation_breaks_its_contract(tmp_path: Path, case: str) -> None:
    changes, contract = MUTATIONS[case]
    output = lint(tmp_path, changes)
    assert f"{contract} BROKEN" in output, output


EXCEPTIONS_TOML = (
    "[[tool.importlinter.contracts]]\n"
    f'name = "{INDEPENDENCE}"\n'
    'type = "independence"\n'
    'modules = ["backend.app.features.*"]\n'
    "ignore_imports = [\n"
    '    "backend.app.features.legacy.* -> backend.app.features.alpha.**",\n'
    "]\n"
)
LEGACY_ENTRIES = {
    "FEATURE_ROOT_FILE command.py": 1,
    "FEATURE_ROOT_FILE store.py": 1,
    "FEATURE_ROOT_FILE worker.py": 1,
}


def checker_tree() -> dict[str, str]:
    files = clean_tree()
    files.update(
        {
            "pyproject.toml": EXCEPTIONS_TOML,
            f"{LEGACY}/__init__.py": "",
            f"{LEGACY}/command.py": "",
            f"{LEGACY}/store.py": "",
            f"{LEGACY}/worker.py": "",
            "scripts/layer_baseline.json": json.dumps({"features": {"legacy": LEGACY_ENTRIES}}),
        }
    )
    return files


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def checker_repo(root: Path, changes: dict[str, str] | None = None) -> Path:
    write_tree(root, checker_tree())
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    write_tree(root, changes or {})
    git(root, "add", "-A")
    return root


def run_checker(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_baselined_tree_passes(tmp_path: Path) -> None:
    result = run_checker(checker_repo(tmp_path), "--against", "HEAD")
    assert result.returncode == 0, result.stdout
    assert result.stdout.strip() == "check_layers: ok"


CHECKER_MUTATIONS = {
    "new_root_file": (
        {f"{ALPHA}/helpers.py": ""},
        f"FEATURE_ROOT_FILE helpers.py. Fix: git mv {ALPHA}/helpers.py",
    ),
    "new_root_file_in_baselined_feature": (
        {f"{LEGACY}/extra_router.py": ""},
        (
            f"FEATURE_ROOT_FILE extra_router.py. Fix: git mv {LEGACY}/extra_router.py "
            f"{LEGACY}/controller/extra.py"
        ),
    ),
    "unexpected_subfolder": (
        {f"{ALPHA}/helpers/__init__.py": ""},
        "UNKNOWN_FOLDER helpers. Fix: move the modules in",
    ),
    "nested_layer_folder": (
        {f"{ALPHA}/service/sub/__init__.py": ""},
        "NESTED_FOLDER service/sub. Fix: flatten",
    ),
    "layer_without_init": (
        {f"{ALPHA}/repository/__init__.py": None},
        f"MISSING_INIT repository/__init__.py. Fix: touch {ALPHA}/repository/__init__.py",
    ),
    "code_in_feature_init": (
        {f"{ALPHA}/__init__.py": "x = 1\n"},
        "FEATURE_INIT_CODE __init__.py. Fix: empty",
    ),
    "app_state_in_service": (
        {f"{ALPHA}/service/logic.py": "def f(request):\n    return request.app.state.db\n"},
        "APP_STATE_OUTSIDE_CONTROLLER service/logic.py:2. Fix: take the collaborator",
    ),
}


def apply(root: Path, changes: dict[str, str | None]) -> None:
    for name, text in changes.items():
        target = root / name
        if text is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("case", sorted(CHECKER_MUTATIONS))
def test_new_finding_outside_the_baseline_fails_with_a_fix(tmp_path: Path, case: str) -> None:
    changes, expected = CHECKER_MUTATIONS[case]
    root = checker_repo(tmp_path)
    apply(root, changes)
    result = run_checker(root, "--against", "HEAD")
    assert result.returncode == 1, result.stdout
    assert expected in result.stdout, result.stdout
    assert "Rules: AGENTS.md" in result.stdout


def test_untracked_ignored_files_are_not_findings(tmp_path: Path) -> None:
    root = checker_repo(tmp_path, {".gitignore": "scratch/\n", f"{ALPHA}/scratch/x.py": ""})
    result = run_checker(root, "--against", "HEAD")
    assert result.returncode == 0, result.stdout


def edit_baseline(root: Path, feature: str, entries: dict[str, int]) -> None:
    path = root / "scripts/layer_baseline.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["features"].setdefault(feature, {}).update(entries)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_hiding_a_violation_in_the_baseline_is_caught_by_against(tmp_path: Path) -> None:
    root = checker_repo(tmp_path, {f"{LEGACY}/extra.py": ""})
    edit_baseline(root, "legacy", {"FEATURE_ROOT_FILE extra.py": 1})
    assert run_checker(root).returncode == 0
    result = run_checker(root, "--against", "HEAD")
    assert result.returncode == 1, result.stdout
    assert (
        "legacy: scripts/layer_baseline.json gained FEATURE_ROOT_FILE extra.py compared with HEAD"
        in result.stdout
    )
    assert "Fix: remove the entry and fix the code instead." in result.stdout


def test_a_raised_count_in_the_baseline_is_caught_by_against(tmp_path: Path) -> None:
    root = checker_repo(tmp_path)
    edit_baseline(root, "legacy", {"FEATURE_ROOT_FILE store.py": 2})
    result = run_checker(root, "--against", "HEAD")
    assert result.returncode == 1, result.stdout
    assert "gained FEATURE_ROOT_FILE store.py (1 -> 2) compared with HEAD" in result.stdout


def test_a_feature_new_to_the_baseline_is_caught_by_against(tmp_path: Path) -> None:
    root = checker_repo(tmp_path, {f"{ALPHA}/extra.py": ""})
    edit_baseline(root, "alpha", {"FEATURE_ROOT_FILE extra.py": 1})
    result = run_checker(root, "--against", "HEAD")
    assert result.returncode == 1, result.stdout
    assert (
        'alpha: scripts/layer_baseline.json gained the feature "alpha"; FEATURE_ROOT_FILE extra.py'
        in result.stdout
    )


def test_an_empty_feature_entry_is_caught(tmp_path: Path) -> None:
    root = checker_repo(tmp_path)
    edit_baseline(root, "alpha", {})
    result = run_checker(root, "--against", "HEAD")
    assert result.returncode == 1, result.stdout
    assert 'alpha: scripts/layer_baseline.json gained the feature "alpha"' in result.stdout
    assert 'alpha: no layer findings left. Lock the gain: delete "alpha"' in result.stdout


def test_against_a_ref_without_a_baseline_checks_only_the_findings(tmp_path: Path) -> None:
    root = checker_repo(tmp_path)
    git(root, "rm", "-q", "--cached", "scripts/layer_baseline.json")
    git(root, "commit", "-q", "-m", "drop baseline")
    git(root, "add", "-A")
    result = run_checker(root, "--against", "HEAD")
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("ref", ["no-such-ref", "0" * 40, "FETCH_HEAD"])
def test_against_a_missing_ref_fails_with_a_fix(tmp_path: Path, ref: str) -> None:
    result = run_checker(checker_repo(tmp_path), "--against", ref)
    assert result.returncode == 1, result.stdout
    assert f"--against {ref}: no such commit in this clone" in result.stdout
    assert f"Fix: fetch it first (git fetch --no-tags --depth=1 origin {ref})" in result.stdout


def test_a_fixed_entry_must_leave_the_baseline(tmp_path: Path) -> None:
    root = checker_repo(tmp_path)
    apply(root, {f"{LEGACY}/store.py": None})
    result = run_checker(root, "--against", "HEAD")
    assert result.returncode == 1, result.stdout
    assert "legacy: 1 baseline entries are fixed or smaller. Lock the gain" in result.stdout
    assert "  FEATURE_ROOT_FILE store.py: 0 left, baseline 1" in result.stdout


def with_exception(line: str) -> str:
    return EXCEPTIONS_TOML.removesuffix("]\n") + f'    "{line}",\n]\n'


def test_a_new_import_exception_is_caught_by_against(tmp_path: Path) -> None:
    line = f"{F}.alpha.service.logic -> {F}.beta.repository.rows"
    root = checker_repo(tmp_path, {"pyproject.toml": with_exception(line)})
    assert run_checker(root).returncode == 0
    result = run_checker(root, "--against", "HEAD")
    assert result.returncode == 1, result.stdout
    assert f"'{INDEPENDENCE}' gained the exception \"{line}\" compared with HEAD" in result.stdout
    assert "Fix: remove the line and fix the import instead." in result.stdout


def test_spelling_out_an_earlier_wildcard_exception_is_not_growth(tmp_path: Path) -> None:
    line = f"{F}.legacy.worker -> {F}.alpha.service.logic"
    root = checker_repo(tmp_path, {"pyproject.toml": with_exception(line)})
    result = run_checker(root, "--against", "HEAD")
    assert result.returncode == 0, result.stdout


def test_main_passes_with_its_baseline() -> None:
    result = run_checker(REPO_ROOT)
    assert result.returncode == 0, result.stdout
    assert result.stdout.strip() == "check_layers: ok"


def test_baseline_lists_only_features_that_exist() -> None:
    data = json.loads((REPO_ROOT / "scripts/layer_baseline.json").read_text(encoding="utf-8"))
    features = {
        p.name
        for p in (REPO_ROOT / "backend/app/features").iterdir()
        if (p / "__init__.py").exists()
    }
    assert set(data["features"]) <= features


REVERTED_IMPORTS = (
    "cameras.roster_sync -> connection.topology_retry_coordinator",
    "cameras.topology_client -> connection.enrollment",
    "connection.router -> cameras.dependencies",
    "cameras.dependencies -> connection.dependencies",
    "cameras.router -> connection.dependencies",
    "cameras.router -> status.heartbeat_store",
    "cameras.router -> runtime_settings.dependencies",
    "evidence.router -> runtime_settings.dependencies",
    "status.router -> runtime_settings.dependencies",
    "detection_settings.router -> cameras.store",
    "detection_settings.router -> connection.dependencies",
)


def real_exceptions(contract: str) -> list[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    contracts = data["tool"]["importlinter"]["contracts"]
    return next(c["ignore_imports"] for c in contracts if c["name"] == contract)


@pytest.mark.parametrize("edge", REVERTED_IMPORTS)
def test_a_removed_cross_feature_import_has_no_exception_to_come_back_through(edge: str) -> None:
    source, target = edge.split(" -> ")
    exceptions = real_exceptions(INDEPENDENCE)
    assert f"{F}.{source} -> {F}.{target}" not in exceptions
    assert not [line for line in exceptions if "*" in line]


def ruff_dto(path: str, source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(RUFF), "check", "--select", "TID251", "--no-cache", "--stdin-filename", path, "-"],
        cwd=REPO_ROOT,
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )


DTO_SOURCES = {
    "base_model": BASE_MODEL,
    "aliased_base": "from pydantic import BaseModel as Base\n\n\nclass Row(Base):\n    x: int\n",
    "dotted_module_alias": "import pydantic.main as pm\n\n\nclass W(pm.BaseModel):\n    x: int\n",
    "root_model": "from pydantic import RootModel\n\n\nclass Ids(RootModel[int]):\n    pass\n",
    "pydantic_v1": "from pydantic.v1 import BaseModel\n\n\nclass Wire(BaseModel):\n    x: int\n",
    "pydantic_dataclass": (
        "from pydantic.dataclasses import dataclass\n\n\n@dataclass\nclass Row:\n    x: int\n"
    ),
}


@pytest.mark.parametrize("case", sorted(DTO_SOURCES))
@pytest.mark.parametrize("path", [f"{ALPHA}/service/logic.py", f"{ALPHA}/worker.py"])
def test_a_pydantic_model_outside_the_controller_fails_ruff(case: str, path: str) -> None:
    result = ruff_dto(path, DTO_SOURCES[case])
    assert result.returncode == 1, result.stdout
    assert "TID251" in result.stdout, result.stdout
    assert "DTOs live in the controller" in result.stdout, result.stdout


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (f"{ALPHA}/controller/api.py", BASE_MODEL),
        (f"{ALPHA}/router.py", BASE_MODEL),
        (f"{ALPHA}/extra_router.py", BASE_MODEL),
        ("backend/app/core/config.py", BASE_MODEL),
        (
            f"{ALPHA}/service/logic.py",
            "from pydantic import JsonValue, TypeAdapter\n\nA = TypeAdapter(list[JsonValue])\n",
        ),
        (
            f"{ALPHA}/service/logic.py",
            "from dataclasses import dataclass\n\n\n@dataclass\nclass V:\n    x: int\n",
        ),
    ],
)
def test_controller_models_and_plain_validation_pass_ruff(path: str, source: str) -> None:
    result = ruff_dto(path, source)
    assert result.returncode == 0, result.stdout
