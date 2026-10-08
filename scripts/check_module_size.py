import argparse
import ast
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

LIMIT = 250
SCOPE = Path("worker")
BASELINE_PATH = Path("scripts/check_module_size_baseline.json")


def logical_loc(source: str) -> int:
    tree = ast.parse(source)
    return len(
        {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.stmt) and not isinstance(node, ast.Import | ast.ImportFrom)
        }
    )


def evaluate(counts: dict[str, int], baseline: dict[str, int]) -> list[str]:
    out: list[str] = []
    for path, count in sorted(counts.items()):
        allowed = baseline.get(path)
        if count > LIMIT and (allowed is None or count > allowed):
            suffix = "" if allowed is None else f" (baseline {allowed})"
            out.append(f"{path}:0: MODULE_TOO_LARGE: {count} logical lines > {LIMIT}{suffix}")
        elif allowed is not None and count < allowed:
            fix = f'"{path}": {count},' if count > LIMIT else f'remove "{path}"'
            out.append(
                f"{path}:0: SHRINK_BASELINE: {count} logical lines < baseline {allowed}; "
                f"update {BASELINE_PATH.as_posix()}: {fix}"
            )
    for path in sorted(set(baseline) - set(counts)):
        out.append(f'{path}:0: SHRINK_BASELINE: module is gone; remove "{path}" from the baseline')
    return out


def load_baseline(root: Path) -> dict[str, int]:
    try:
        raw = json.loads((root / BASELINE_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(path): int(count) for path, count in raw.items()}


def tracked_worker_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", SCOPE.as_posix()],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [
        Path(name) for name in out.split("\0") if name.endswith(".py") and (root / name).is_file()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_module_size")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    counts = {
        path.as_posix(): logical_loc((root / path).read_text(encoding="utf-8"))
        for path in tracked_worker_files(root)
    }
    if args.write_baseline:
        over = {path: count for path, count in sorted(counts.items()) if count > LIMIT}
        (root / BASELINE_PATH).write_text(json.dumps(over, indent=2) + "\n", encoding="utf-8")
        return 0
    baseline = load_baseline(root)
    if args.paths:
        wanted = {(p if p.is_absolute() else Path.cwd() / p).resolve() for p in args.paths}
        keep = {
            path
            for path in counts
            if any(
                (root / path).resolve() == w or w in (root / path).resolve().parents for w in wanted
            )
        }
        counts = {path: count for path, count in counts.items() if path in keep}
        baseline = {path: count for path, count in baseline.items() if path in keep}
    findings = evaluate(counts, baseline)
    for line in findings:
        print(line)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
