import argparse
import ast
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

SCOPE_ROOT = Path("worker")
OWNER = Path("worker/runtime/threads.py")
BASELINE_PATH = Path("scripts/check_thread_starts_baseline.json")
ADVICE = (
    "direct Thread( start belongs only in worker/runtime/threads.py "
    "-> use start_guarded_thread(name, target, on_death)"
)


def thread_start_lines(source: str) -> list[int]:
    tree = ast.parse(source)
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "Thread")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "Thread")
        )
    )


def in_scope(path: Path) -> bool:
    return path.suffix == ".py" and SCOPE_ROOT in path.parents and path != OWNER


def load_baseline(root: Path = Path()) -> dict[str, int]:
    try:
        raw = json.loads((root / BASELINE_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(path): int(count) for path, count in raw.items()}


def findings(root: Path, tracked: Sequence[Path]) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for path in sorted(item for item in tracked if in_scope(item)):
        lines = thread_start_lines((root / path).read_text(encoding="utf-8"))
        if lines:
            found[path.as_posix()] = lines
    return found


def main(argv: Sequence[str] | None = None) -> int:
    _ = argparse.ArgumentParser(prog="check_thread_starts").parse_args(argv)
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    ).stdout.strip()
    root = Path(top)
    names = subprocess.run(
        ["git", "-C", top, "ls-files", "-z"], check=True, capture_output=True, text=True
    ).stdout.split("\0")
    found = findings(root, [Path(name) for name in names if name and (root / name).is_file()])
    baseline = load_baseline(root)
    failed = False
    for path, lines in found.items():
        if len(lines) > baseline.get(path, 0):
            failed = True
            for line in lines:
                print(f"{path}:{line}: THREAD_START: {ADVICE}")
    print(f"check_thread_starts: {sum(map(len, found.values()))} direct start(s), baselined")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
