from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

FEATURES = Path("backend/app/features")
BASELINE = Path("scripts/layer_baseline.json")
LAYERS = ("controller", "service", "repository")
ROOT_ALLOWED = frozenset({"__init__.py", "AGENTS.md"})
GUIDE = "AGENTS.md, Enforced Rules (backend feature layers)"
CONTROLLER_STEMS = frozenset(
    {"router", "dependencies", "http", "auth", "schemas", "responses", "media_response", "wire"}
)
REPOSITORY_STEMS = frozenset({"store", "repository", "records"})


@dataclass(frozen=True)
class Finding:
    feature: str
    rule: str
    subject: str
    fix: str

    @property
    def key(self) -> str:
        return f"{self.rule} {self.subject}"


@dataclass(frozen=True)
class Options:
    repo: Path
    baseline: Path
    against: str | None
    write: bool


def root_role(stem: str) -> str:
    if stem in CONTROLLER_STEMS or stem.endswith("_router"):
        return "controller"
    if (
        stem in REPOSITORY_STEMS
        or stem.endswith(("_store", "_repository", "_sql", "_rows"))
        or stem.startswith("postgres_")
    ):
        return "repository"
    return "service"


def suggest_move(path: Path) -> str:
    stem = path.stem
    role = root_role(stem)
    if role == "controller" and stem.endswith("_router"):
        return f"git mv {path} {path.parent}/controller/{stem.removesuffix('_router')}.py"
    if stem.endswith("_service"):
        return f"git mv {path} {path.parent}/service/{stem.removesuffix('_service')}.py"
    if role == "service":
        return f"git mv {path} {path.parent}/<controller|service|repository>/{path.name}"
    return f"git mv {path} {path.parent}/{role}/{path.name}"


def visible_paths(repo: Path) -> set[Path] | None:
    listed = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            str(FEATURES),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        return None
    seen: set[Path] = set()
    for line in listed.stdout.splitlines():
        path = repo / line
        if "__pycache__" not in path.parts and path.exists():
            seen.add(path)
            seen.update(path.parents)
    return seen


def reads_app_state(module: Path) -> int | None:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr == "state"):
            continue
        owner = node.value
        if (isinstance(owner, ast.Name) and owner.id == "app") or (
            isinstance(owner, ast.Attribute) and owner.attr == "app"
        ):
            return node.lineno
    return None


def scan_layer(
    repo: Path, feature: str, layer: Path, exists: Callable[[Path], bool]
) -> Iterable[Finding]:
    rel = layer.relative_to(repo)
    init = layer / "__init__.py"
    if not exists(init):
        yield Finding(
            feature, "MISSING_INIT", f"{layer.name}/__init__.py", f"touch {rel}/__init__.py"
        )
    elif init.read_text(encoding="utf-8").strip():
        yield Finding(
            feature,
            "LAYER_INIT_CODE",
            f"{layer.name}/__init__.py",
            f"empty {rel}/__init__.py and import each name from the module that defines it",
        )
    for child in sorted(c for c in layer.iterdir() if exists(c)):
        if child.is_dir():
            yield Finding(
                feature,
                "NESTED_FOLDER",
                f"{layer.name}/{child.name}",
                f"flatten {child.relative_to(repo)} into {rel}/{child.name}_<name>.py files, "
                "or split a new feature",
            )
        elif child.suffix == ".py" and layer.name != "controller":
            line = reads_app_state(child)
            if line is not None:
                yield Finding(
                    feature,
                    "APP_STATE_OUTSIDE_CONTROLLER",
                    f"{layer.name}/{child.name}:{line}",
                    "take the collaborator as a constructor argument and wire it in the lifespan",
                )


def scan_feature(
    repo: Path, feature_dir: Path, exists: Callable[[Path], bool]
) -> Iterable[Finding]:
    feature = feature_dir.name
    rel = feature_dir.relative_to(repo)
    init = feature_dir / "__init__.py"
    if not exists(init):
        yield Finding(feature, "MISSING_INIT", "__init__.py", f"touch {rel}/__init__.py")
    elif init.read_text(encoding="utf-8").strip():
        yield Finding(
            feature,
            "FEATURE_INIT_CODE",
            "__init__.py",
            f"empty {rel}/__init__.py and import each name from its layer module",
        )
    for child in sorted(
        c for c in feature_dir.iterdir() if exists(c) and c.name not in ROOT_ALLOWED
    ):
        crel = child.relative_to(repo)
        if child.is_file():
            fix = (
                suggest_move(crel)
                if child.suffix == ".py"
                else f"move {crel} into a layer folder or delete it"
            )
            yield Finding(feature, "FEATURE_ROOT_FILE", child.name, fix)
        elif child.name not in LAYERS:
            yield Finding(
                feature,
                "UNKNOWN_FOLDER",
                child.name,
                f"move the modules in {crel} into controller/, service/ or repository/, "
                "or into backend/app/shared when several features use them",
            )
        else:
            yield from scan_layer(repo, feature, child, exists)


def scan_tree(repo: Path) -> list[Finding]:
    shown = visible_paths(repo)

    def exists(path: Path) -> bool:
        return (
            path.exists() and "__pycache__" not in path.parts and (shown is None or path in shown)
        )

    features = sorted(p for p in (repo / FEATURES).iterdir() if p.is_dir() and exists(p))
    return [
        finding for feature_dir in features for finding in scan_feature(repo, feature_dir, exists)
    ]


Counts = dict[str, dict[str, int]]


def load_baseline(text: str) -> Counts:
    features = json.loads(text)["features"] if text.strip() else {}
    return {
        feature: dict.fromkeys(entries, 1) if isinstance(entries, list) else dict(entries)
        for feature, entries in features.items()
    }


def commit_exists(repo: Path, ref: str) -> bool:
    found = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return found.returncode == 0


def baseline_at(repo: Path, ref: str, path: Path) -> Counts | None:
    shown = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path.relative_to(repo).as_posix()}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return load_baseline(shown.stdout) if shown.returncode == 0 else None


def ignored_imports(text: str) -> dict[str, list[str]]:
    contracts = tomllib.loads(text).get("tool", {}).get("importlinter", {}).get("contracts", [])
    return {contract["name"]: contract.get("ignore_imports", []) for contract in contracts}


def covers(pattern: str, line: str) -> bool:
    expression = re.escape(pattern).replace(r"\*\*", ".+").replace(r"\*", r"[^.\s]+")
    return re.fullmatch(expression, line) is not None


def grown_exceptions(repo: Path, ref: str) -> list[str]:
    shown = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:pyproject.toml"],
        capture_output=True,
        text=True,
        check=False,
    )
    if shown.returncode != 0:
        return []
    before = ignored_imports(shown.stdout)
    now = ignored_imports((repo / "pyproject.toml").read_text(encoding="utf-8"))
    return [
        f"pyproject.toml: import-linter contract '{name}' gained the exception "
        f'"{line}" compared with {ref}. Exception lists only shrink. '
        "Fix: remove the line and fix the import instead."
        for name, lines in now.items()
        if name in before
        for line in lines
        if not any(covers(earlier, line) for earlier in before[name])
    ]


def check(opts: Options) -> list[str]:
    if opts.against is not None and not commit_exists(opts.repo, opts.against):
        missing = (
            f"--against {opts.against}: no such commit in this clone, so the baseline "
            "cannot be compared. Fix: fetch it first "
            f"(git fetch --no-tags --depth=1 origin {opts.against}) or pass a commit "
            "that exists, e.g. --against HEAD."
        )
        return [missing]
    findings = scan_tree(opts.repo)
    current: Counts = {}
    for finding in findings:
        entries = current.setdefault(finding.feature, {})
        entries[finding.key] = entries.get(finding.key, 0) + 1
    if opts.write:
        ordered = {feature: dict(sorted(current[feature].items())) for feature in sorted(current)}
        opts.baseline.write_text(json.dumps({"features": ordered}, indent=2) + "\n")
        return []
    baseline = (
        load_baseline(opts.baseline.read_text(encoding="utf-8")) if opts.baseline.exists() else {}
    )
    errors: list[str] = []
    if opts.against is not None:
        errors.extend(grown_exceptions(opts.repo, opts.against))
        before = baseline_at(opts.repo, opts.against, opts.baseline)
        for feature, entries in sorted(baseline.items()):
            earlier = (before or {}).get(feature, {})
            added = (
                [
                    key
                    if n == 1 and not earlier.get(key)
                    else f"{key} ({earlier.get(key, 0)} -> {n})"
                    for key, n in entries.items()
                    if n > earlier.get(key, 0)
                ]
                if before is not None
                else []
            )
            if before is not None and feature not in before:
                added.insert(0, f'the feature "{feature}"')
            if added:
                errors.append(
                    f"{feature}: {BASELINE} gained {'; '.join(added)} "
                    f"compared with {opts.against}. "
                    "The baseline only shrinks. Fix: remove the entry and fix the code instead."
                )
    fixes = {(f.feature, f.key): f.fix for f in findings}
    for feature in sorted(set(current) | set(baseline)):
        have, allowed = current.get(feature, {}), baseline.get(feature, {})
        grown = [(key, n) for key, n in have.items() if n > allowed.get(key, 0)]
        if grown:
            label = (
                f"{len(allowed)} in baseline" if feature in baseline else "strict, not in baseline"
            )
            errors.append(f"{feature}: {len(grown)} new layer finding(s), {label}.")
            for key, n in grown:
                was = allowed.get(key, 0)
                count = f" ({n} found, baseline allows {was})" if was or n > 1 else ""
                errors.append(f"  {key}{count}. Fix: {fixes[(feature, key)]}.")
        gone = [(key, n) for key, n in allowed.items() if have.get(key, 0) < n]
        if feature in baseline and not have:
            errors.append(
                f"{feature}: no layer findings left. "
                f'Lock the gain: delete "{feature}" from {BASELINE}.'
            )
        elif gone:
            errors.append(
                f"{feature}: {len(gone)} baseline entries are fixed or smaller. "
                f"Lock the gain: lower or delete them in {BASELINE}."
            )
            errors.extend(f"  {key}: {have.get(key, 0)} left, baseline {n}" for key, n in gone)
    return errors


def parse(argv: list[str]) -> Options:
    parser = argparse.ArgumentParser(prog="check_layers")
    parser.add_argument("repo", nargs="?", default=".")
    parser.add_argument("--baseline")
    parser.add_argument("--against")
    parser.add_argument("--write-baseline", action="store_true")
    ns = parser.parse_args(argv)
    repo = Path(ns.repo).resolve()
    baseline = Path(ns.baseline).resolve() if ns.baseline else repo / BASELINE
    return Options(repo, baseline, ns.against, ns.write_baseline)


def main(argv: list[str]) -> int:
    opts = parse(argv)
    errors = check(opts)
    if opts.write:
        print(f"wrote {opts.baseline}")
        return 0
    if errors:
        print("\n".join(errors))
        print(f"\nRules: {GUIDE}")
        return 1
    print("check_layers: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
