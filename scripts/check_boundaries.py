import argparse
import ast
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

BOUNDARY_OWNER = Path("shared/boundary")
SCOPE_ROOTS = (
    Path("backend"),
    Path("worker"),
    Path("shared"),
    Path("contracts"),
    Path("scripts"),
    Path("tests_support"),
)
EXCLUDED_ROOTS = (BOUNDARY_OWNER, Path("tests"))
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})
BASELINE_PATH = Path("scripts/check_boundaries_baseline.json")
ZERO_ENFORCED_ROOTS = (Path("worker/domains"),)
BROAD = frozenset({"Exception", "BaseException"})
SUPPRESS_NAMES = frozenset({"suppress"})

BROAD_EXCEPT = "BROAD_EXCEPT"
SUPPRESS_BROAD = "SUPPRESS_BROAD"
UNPARSEABLE = "UNPARSEABLE"

HELPERS = (
    "isolate() for one item in a loop, degrade(message=...) for an optional feature, "
    "attempt_delivery() for a send, probe() for a vendor import or probe, "
    "cleanup_on_failure() for rollback or close, translate() to raise a typed error, "
    "root_sink() at a process root"
)

ADVICE = {
    BROAD_EXCEPT: f"broad except belongs only in shared/boundary -> use {HELPERS}",
    SUPPRESS_BROAD: "contextlib.suppress of Exception is a hidden broad except -> use isolate()",
    UNPARSEABLE: "file could not be parsed -> fix the syntax error",
}


@dataclass(frozen=True, slots=True)
class Finding:
    path: Path
    line: int
    kind: str

    def render(self) -> str:
        return f"{self.path.as_posix()}:{self.line}: {self.kind}: {ADVICE[self.kind]}"


def _is_broad(node: ast.expr | None) -> bool:
    if isinstance(node, ast.Name):
        return node.id in BROAD
    if isinstance(node, ast.Attribute):
        return (
            node.attr in BROAD and isinstance(node.value, ast.Name) and node.value.id == "builtins"
        )
    if isinstance(node, ast.Tuple):
        return any(_is_broad(element) for element in node.elts)
    return False


def _is_suppress(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id in SUPPRESS_NAMES
    return isinstance(func, ast.Attribute) and func.attr in SUPPRESS_NAMES


def broad_handlers(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None or _is_broad(node.type):
                found.append((node.lineno, BROAD_EXCEPT))
        elif (
            isinstance(node, ast.Call)
            and _is_suppress(node.func)
            and any(_is_broad(argument) for argument in node.args)
        ):
            found.append((node.lineno, SUPPRESS_BROAD))
    return found


def check_source(path: Path, source: str) -> list[Finding]:
    try:
        handlers = broad_handlers(ast.parse(source, filename=str(path)))
    except (SyntaxError, ValueError):
        return [Finding(path, 0, UNPARSEABLE)]
    findings = [Finding(path, line, kind) for line, kind in handlers]
    return sorted(findings, key=lambda finding: (finding.line, finding.kind))


def check_file(path: Path, root: Path = Path()) -> list[Finding]:
    try:
        source = (root / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return [Finding(path, 0, UNPARSEABLE)]
    return check_source(path, source)


def load_baseline(root: Path = Path()) -> dict[str, int]:
    try:
        raw = json.loads((root / BASELINE_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(kind): int(count) for kind, count in raw.items()}


def drift_summary(counts: Counter[str], baseline: dict[str, int]) -> str:
    kinds = sorted(set(counts) | set(baseline))
    parts = [
        f"{kind}={counts.get(kind, 0)}({counts.get(kind, 0) - baseline.get(kind, 0):+d})"
        for kind in kinds
    ]
    return ", ".join(parts) or "none"


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def blocking_findings(findings: Sequence[Finding]) -> list[Finding]:
    return [
        finding
        for finding in findings
        if any(_is_under(finding.path, root) for root in ZERO_ENFORCED_ROOTS)
    ]


def in_scope(path: Path) -> bool:
    if path.suffix not in PYTHON_SUFFIXES:
        return False
    if any(_is_under(path, root) for root in EXCLUDED_ROOTS):
        return False
    return any(_is_under(path, root) for root in SCOPE_ROOTS)


def select_paths(tracked: Sequence[Path]) -> list[Path]:
    return sorted(path for path in tracked if in_scope(path))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def tracked_files(root: Path) -> list[Path]:
    names = _git("-C", str(root), "ls-files", "-z").split("\0")
    return [Path(name) for name in names if name and (root / name).is_file()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_boundaries")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)
    try:
        root = Path(_git("rev-parse", "--show-toplevel").strip())
        tracked = tracked_files(root)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f".:0: NO_GIT: cannot list tracked files ({error})", file=sys.stderr)
        return 0
    selected = select_paths(tracked)
    if args.paths:
        wanted = {(raw if raw.is_absolute() else Path.cwd() / raw).resolve() for raw in args.paths}
        selected = [
            path
            for path in selected
            if any(
                (root / path).resolve() == item or item in (root / path).resolve().parents
                for item in wanted
            )
        ]
    findings = [finding for path in selected for finding in check_file(path, root)]
    for finding in findings:
        print(finding.render())
    counts = Counter(finding.kind for finding in findings)
    summary = drift_summary(counts, load_baseline(root))
    blocking = blocking_findings(findings)
    enforced = ", ".join(root.as_posix() for root in ZERO_ENFORCED_ROOTS)
    print(
        f"check_boundaries (report-only): {len(findings)} finding(s), "
        f"vs {BASELINE_PATH.as_posix()} baseline: {summary}; "
        f"enforced at zero: {enforced} ({len(blocking)} finding(s))"
    )
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
