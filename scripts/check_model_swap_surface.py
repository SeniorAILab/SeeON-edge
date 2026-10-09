import argparse
import ast
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SCOPE_ROOT = Path("worker/runtime/provenance")
SCOPE_FILES = frozenset({Path("worker/runtime/config/local_env.py")})
CEREMONY_LITERALS = frozenset({"green", "verdict"})

CEREMONY_LITERAL = "CEREMONY_LITERAL"
STATUS_COMPARISON = "STATUS_COMPARISON"
UNPARSEABLE = "UNPARSEABLE"

ADVICE = {
    CEREMONY_LITERAL: "a green/verdict literal revives the model-swap ceremony -> swapping a "
    "model must stay a bytes check against the bundle manifest",
    STATUS_COMPARISON: "a status comparison revives the model-swap ceremony -> admission "
    "judges bytes, not a recorded status",
    UNPARSEABLE: "file could not be parsed -> fix the syntax error",
}


@dataclass(frozen=True, slots=True)
class Finding:
    path: Path
    line: int
    kind: str
    detail: str = ""

    def render(self) -> str:
        detail = f" ({self.detail})" if self.detail else ""
        return f"{self.path.as_posix()}:{self.line}: {self.kind}: {ADVICE[self.kind]}{detail}"


def _is_status(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "status"
    return isinstance(node, ast.Attribute) and node.attr == "status"


def ceremony_findings(path: Path, source: str) -> list[Finding]:
    try:
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError):
        return [Finding(path, 0, UNPARSEABLE)]
    found: list[Finding] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in CEREMONY_LITERALS
        ):
            found.append(Finding(path, node.lineno, CEREMONY_LITERAL, repr(node.value)))
        elif (
            isinstance(node, ast.Compare)
            and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops)
            and any(_is_status(part) for part in (node.left, *node.comparators))
        ):
            found.append(Finding(path, node.lineno, STATUS_COMPARISON))
    return sorted(found, key=lambda finding: (finding.line, finding.kind))


def in_scope(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    return path in SCOPE_FILES or SCOPE_ROOT in path.parents


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def tracked_files(root: Path) -> list[Path]:
    names = _git("-C", str(root), "ls-files", "-z").split("\0")
    return [Path(name) for name in names if name and (root / name).is_file()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_model_swap_surface")
    parser.parse_args(argv)
    root = Path(_git("rev-parse", "--show-toplevel").strip())
    findings: list[Finding] = []
    for path in sorted(path for path in tracked_files(root) if in_scope(path)):
        source = (root / path).read_text(encoding="utf-8")
        findings.extend(ceremony_findings(path, source))
    for finding in findings:
        print(finding.render())
    print(f"check_model_swap_surface: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
