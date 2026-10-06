import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

TEMPORARY_EXCLUDED_PATHS = (Path("backend/app/features/cameras"),)
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

_CODES = r"[A-Z]+[0-9]+(?:[\s,]+[A-Z]+[0-9]+)*"
_NAMES = r"[\w-]+(?:\s*,\s*[\w-]+)*"
_DIRECTIVE = (
    r"#\s*(?:"
    rf"(?:ruff:\s*)?noqa(?![\w-])(?::\s*{_CODES})?"
    rf"|type:\s*ignore(?![\w-])(?:\[{_NAMES}\])?"
    r"|pragma:\s*no\s+cover(?![\w-])"
    rf"|pyright:\s*(?:ignore(?:\[{_NAMES}\])?|basic|standard|strict|\w+\s*=\s*\w+)(?![\w-])"
    r"|isort:\s*(?:skip_file|skip|off|on)(?![\w-])"
    r"|fmt:\s*(?:off|on|skip)(?![\w-])"
    r")"
)
DIRECTIVE = re.compile(_DIRECTIVE, re.IGNORECASE)
BARE_DIRECTIVES = re.compile(rf"(?:{_DIRECTIVE}\s*)+", re.IGNORECASE)
CODING_LINE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")

NO_COMMENT = "NO_COMMENT"
NO_DOCSTRING = "NO_DOCSTRING"
DIRECTIVE_WITH_PROSE = "DIRECTIVE_WITH_PROSE"
UNPARSEABLE = "UNPARSEABLE"
OWN_LINE_FIX = "delete this comment line"
TRAILING_FIX = "delete the trailing comment, keep the code"


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    code: str
    what: str
    fix: str

    def render(self) -> str:
        return f"{self.path.as_posix()}:{self.line}: {self.code}: {self.what} -> fix: {self.fix}"


def comment_kind(text: str, row: int, line: str) -> str:
    stripped = text.rstrip()
    whole_line = line.strip() == stripped
    if row == 1 and stripped.startswith("#!") and line.startswith("#!"):
        return "shebang"
    if row <= 2 and whole_line and CODING_LINE.match(line):
        return "coding"
    if BARE_DIRECTIVES.fullmatch(stripped):
        return "directive"
    if DIRECTIVE.search(stripped):
        return "directive_with_prose"
    return "prose"


def bare_directives(text: str) -> str:
    return "  ".join(match.group(0) for match in DIRECTIVE.finditer(text))


def shorten(text: str, limit: int = 60) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def comment_findings(path: Path, source: str) -> Iterator[Finding]:
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        row, column = token.start
        kind = comment_kind(token.string, row, token.line)
        own_line = not token.line[:column].strip()
        if kind == "directive_with_prose":
            yield Finding(
                path,
                row,
                DIRECTIVE_WITH_PROSE,
                f"directive carries prose `{shorten(token.string)}`",
                f"replace the comment with `{bare_directives(token.string)}`",
            )
        elif kind == "prose":
            fix = OWN_LINE_FIX if own_line else TRAILING_FIX
            yield Finding(path, row, NO_COMMENT, f"comment `{shorten(token.string)}`", fix)


def statement_blocks(tree: ast.AST) -> Iterator[tuple[ast.AST, list[ast.stmt]]]:
    for node in ast.walk(tree):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(isinstance(i, ast.stmt) for i in value):
                yield node, value


def is_string_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def owner_label(owner: ast.AST) -> str:
    if isinstance(owner, ast.Module):
        return "module docstring"
    if isinstance(owner, ast.ClassDef):
        return f"class `{owner.name}` docstring"
    name = owner.name if isinstance(owner, ast.FunctionDef | ast.AsyncFunctionDef) else "?"
    return f"function `{name}` docstring"


def docstring_findings(path: Path, tree: ast.Module) -> Iterator[Finding]:
    for owner, block in statement_blocks(tree):
        for index, statement in enumerate(block):
            if not is_string_statement(statement):
                continue
            is_docstring = (
                index == 0
                and isinstance(owner, DOCSTRING_OWNERS)
                and block is getattr(owner, "body", None)
            )
            what = owner_label(owner) if is_docstring else "string literal statement used as prose"
            end = statement.end_lineno or statement.lineno
            fix = f"delete lines {statement.lineno}-{end}"
            if len(block) == 1 and not isinstance(owner, ast.Module):
                fix += " and put `...` in their place so the block stays non-empty"
            yield Finding(path, statement.lineno, NO_DOCSTRING, what, fix)


def check_source(path: Path, source: str) -> list[Finding]:
    try:
        tree = ast.parse(source, filename=str(path))
        comments = list(comment_findings(path, source))
    except (SyntaxError, tokenize.TokenError) as error:
        line = getattr(error, "lineno", None) or 1
        return [
            Finding(
                path,
                line,
                UNPARSEABLE,
                f"cannot parse ({error.__class__.__name__})",
                f"make `python -m py_compile {path.as_posix()}` succeed",
            )
        ]
    findings = comments + list(docstring_findings(path, tree))
    return sorted(findings, key=lambda finding: (finding.line, finding.code))


def check_file(path: Path) -> list[Finding]:
    try:
        with tokenize.open(path) as handle:
            source = handle.read()
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        return [
            Finding(
                path,
                1,
                UNPARSEABLE,
                f"cannot read ({error.__class__.__name__})",
                "save the file as UTF-8 text",
            )
        ]
    return check_source(path, source)


def is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def is_excluded(path: Path) -> bool:
    return any(is_under(path, root) for root in TEMPORARY_EXCLUDED_PATHS)


def select_paths(tracked: Sequence[Path], roots: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    python = [path for path in tracked if path.suffix in PYTHON_SUFFIXES]
    if not roots:
        return sorted(path for path in python if not is_excluded(path)), []
    selected: set[Path] = set()
    unmatched: list[Path] = []
    for root in roots:
        matches = {path for path in python if is_under(path, root)}
        if not matches and root.suffix in PYTHON_SUFFIXES and root.is_file():
            matches = {root}
        if not matches:
            unmatched.append(root)
        selected |= matches
    return sorted(path for path in selected if not is_excluded(path)), unmatched


def normalize(raw: Path, base: Path) -> Path:
    if not raw.is_absolute():
        return raw
    try:
        return raw.resolve().relative_to(base.resolve())
    except ValueError:
        return raw


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True, text=True
    )
    return [Path(name) for name in result.stdout.split("\0") if name and Path(name).is_file()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_no_comments")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)
    try:
        tracked = tracked_files()
    except (OSError, subprocess.CalledProcessError) as error:
        print(
            f".:0: NO_GIT: cannot list tracked files ({error})"
            " -> fix: run from the repository root",
            file=sys.stderr,
        )
        return 2
    roots = [normalize(raw, Path.cwd()) for raw in args.paths]
    selected, unmatched = select_paths(tracked, roots)
    for root in unmatched:
        print(
            f"{root.as_posix()}:0: NO_MATCH: no tracked Python file under this path"
            " -> fix: pass a tracked .py file or a directory that contains one",
            file=sys.stderr,
        )
    if unmatched:
        return 2
    findings = [finding for path in selected for finding in check_file(path)]
    for finding in findings:
        print(finding.render())
    if not findings:
        return 0
    counts = Counter(finding.code for finding in findings)
    summary = ", ".join(f"{code}={count}" for code, count in sorted(counts.items()))
    files = len({finding.path for finding in findings})
    print(
        f"::error::check_no_comments: {len(findings)} findings in {files} files ({summary})",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
