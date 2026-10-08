import argparse
import ast
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

PRODUCTION_ROOT = "worker"
TEST_ROOT = "tests"
BASELINE = Path("scripts/private_test_only_baseline.json")
FIX = (
    "a private function under worker/ that only tests call is production code that exists "
    "for tests -> delete it and drive the test through the production path"
)


def _py_files(root: Path, top: str) -> list[Path]:
    return sorted(path for path in (root / top).rglob("*.py") if path.is_file())


def _parse(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return None


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _definitions(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _is_private(node.name)
    }


def _references(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value.rsplit(".", 1)[-1])
    return names


def offenders(root: Path) -> list[str]:
    defined: dict[str, set[str]] = {}
    used_in_production: set[str] = set()
    for path in _py_files(root, PRODUCTION_ROOT):
        tree = _parse(path)
        if tree is None:
            continue
        for name in _definitions(tree):
            defined.setdefault(name, set()).add(path.relative_to(root).as_posix())
        used_in_production |= _references(tree)
    used_in_tests: set[str] = set()
    for path in _py_files(root, TEST_ROOT):
        tree = _parse(path)
        if tree is not None:
            used_in_tests |= _references(tree)
    return sorted(
        f"{file}::{name}"
        for name, files in defined.items()
        if name in used_in_tests and name not in used_in_production
        for file in files
    )


def load_baseline(text: str) -> set[str]:
    return set(json.loads(text)["offenders"]) if text.strip() else set()


def baseline_at(repo: Path, ref: str) -> set[str] | None:
    shown = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{BASELINE.as_posix()}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return load_baseline(shown.stdout) if shown.returncode == 0 else None


def check(repo: Path, against: str | None = None) -> list[str]:
    found = set(offenders(repo))
    path = repo / BASELINE
    baseline = load_baseline(path.read_text(encoding="utf-8")) if path.exists() else set()
    errors = [f"{entry}: not in {BASELINE.as_posix()}. {FIX}" for entry in sorted(found - baseline)]
    errors.extend(
        f"{entry}: fixed or gone, remove it from {BASELINE.as_posix()} (the baseline only shrinks)"
        for entry in sorted(baseline - found)
    )
    if against is not None:
        before = baseline_at(repo, against)
        if before is not None:
            errors.extend(
                f"{entry}: {BASELINE.as_posix()} gained it compared with {against}. "
                "The baseline only shrinks. Fix: remove the entry and fix the code instead."
                for entry in sorted(baseline - before)
            )
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_private_test_only")
    parser.add_argument("repo", nargs="?", default=".")
    parser.add_argument("--against")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    if args.write_baseline:
        text = json.dumps({"offenders": offenders(repo)}, indent=2) + "\n"
        (repo / BASELINE).write_text(text, encoding="utf-8")
        print(f"wrote {BASELINE.as_posix()}")
        return 0
    errors = check(repo, args.against)
    if errors:
        print("\n".join(errors))
        return 1
    print("check_private_test_only: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
