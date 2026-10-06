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
VENDORED_PATHS = (Path("contracts/event.py"),)
PYTHON_SUFFIXES = frozenset({".py", ".pyi"})
DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
SIDE_EFFECT_NODES = (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)

MYPY_ERROR_CODES = frozenset(
    {
        "abstract",
        "annotation-unchecked",
        "arg-type",
        "assert-type",
        "assignment",
        "attr-defined",
        "await-not-async",
        "call-arg",
        "call-overload",
        "comparison-overlap",
        "deprecated",
        "dict-item",
        "empty-body",
        "exhaustive-match",
        "exit-return",
        "explicit-any",
        "explicit-override",
        "func-returns-value",
        "has-type",
        "ignore-without-code",
        "import",
        "import-not-found",
        "import-untyped",
        "index",
        "list-item",
        "literal-required",
        "metaclass",
        "method-assign",
        "misc",
        "mutable-override",
        "name-defined",
        "name-match",
        "narrowed-type-not-subtype",
        "no-any-return",
        "no-any-unimported",
        "no-overload-impl",
        "no-redef",
        "no-untyped-call",
        "no-untyped-def",
        "operator",
        "overload-cannot-match",
        "overload-overlap",
        "override",
        "possibly-undefined",
        "prop-decorator",
        "redundant-cast",
        "redundant-expr",
        "redundant-self",
        "return",
        "return-value",
        "safe-super",
        "str-bytes-safe",
        "str-format",
        "syntax",
        "top-level-await",
        "truthy-bool",
        "truthy-function",
        "truthy-iterable",
        "type-abstract",
        "type-arg",
        "type-var",
        "typeddict-item",
        "typeddict-readonly-mutated",
        "typeddict-unknown-key",
        "unimported-reveal",
        "union-attr",
        "unreachable",
        "unused-awaitable",
        "unused-coroutine",
        "unused-ignore",
        "untyped-decorator",
        "used-before-def",
        "valid-newtype",
        "valid-type",
        "var-annotated",
    }
)

_RULE_CODE = r"[A-Z]+[0-9]+"
_RULE_CODES = rf"{_RULE_CODE}(?:[ \t]*,[ \t]*{_RULE_CODE})*"
_MYPY_CODE = r"[a-z]+(?:-[a-z]+)*"
_MYPY_CODES = rf"{_MYPY_CODE}(?:[ \t]*,[ \t]*{_MYPY_CODE})*"
_PYRIGHT_RULE = r"report[A-Z][A-Za-z]*"
_PYRIGHT_RULES = rf"{_PYRIGHT_RULE}(?:[ \t]*,[ \t]*{_PYRIGHT_RULE})*"
_PYRIGHT_SETTING = rf"{_PYRIGHT_RULE}[ \t]*=[ \t]*(?:true|false|none|information|warning|error)"
_PYRIGHT_SETTINGS = rf"{_PYRIGHT_SETTING}(?:[ \t]*,[ \t]*{_PYRIGHT_SETTING})*"
DIRECTIVE = re.compile(
    r"#[ \t]*(?:"
    rf"(?:ruff:[ \t]*)?noqa(?::[ \t]*{_RULE_CODES})?"
    rf"|type:[ \t]*ignore(?:\[(?P<mypy>{_MYPY_CODES})\])?"
    r"|pragma:[ \t]*no cover"
    rf"|pyright:[ \t]*(?:ignore(?:\[{_PYRIGHT_RULES}\])?|basic|standard|strict|{_PYRIGHT_SETTINGS})"
    r"|fmt:[ \t]*(?:off|on|skip)"
    r")(?=\s|$)"
)
SPACED_RULE_CODES = re.compile(
    rf"(noqa:[ \t]*{_RULE_CODE})((?:[ \t]+{_RULE_CODE})+)(?![\w-])"
)
CODING_LINE = re.compile(
    r"# -\*- coding: [-\w.]+ -\*-|# coding: [-\w.]+|# coding=[-\w.]+"
    r"|# vim: set fileencoding=[-\w.]+ :"
)

NO_COMMENT = "NO_COMMENT"
NO_DOCSTRING = "NO_DOCSTRING"
DIRECTIVE_WITH_PROSE = "DIRECTIVE_WITH_PROSE"
UNPARSEABLE = "UNPARSEABLE"
OWN_LINE_FIX = "delete this comment line"
TRAILING_FIX = "delete the trailing comment, keep the code"
SIDE_EFFECT_FIX = "move the calls inside the f-string into real statements, then delete the literal"


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    code: str
    what: str
    fix: str

    def render(self) -> str:
        return f"{self.path.as_posix()}:{self.line}: {self.code}: {self.what} -> fix: {self.fix}"


def directive_matches(text: str) -> list[re.Match[str]]:
    matches = []
    for match in DIRECTIVE.finditer(text):
        codes = match.group("mypy")
        if codes and not set(re.split(r"[ \t]*,[ \t]*", codes)) <= MYPY_ERROR_CODES:
            continue
        matches.append(match)
    return matches


def comment_kind(text: str, row: int, line: str) -> str:
    stripped = text.rstrip()
    whole_line = line.strip() == stripped
    if row == 1 and stripped.startswith("#!") and line.startswith("#!"):
        return "shebang"
    if row <= 2 and whole_line and CODING_LINE.fullmatch(stripped):
        return "coding"
    matches = directive_matches(stripped)
    if not matches:
        return "prose"
    rest = stripped
    for match in reversed(matches):
        rest = rest[: match.start()] + rest[match.end() :]
    return "directive" if not rest.strip() else "directive_with_prose"


def comma_separated(text: str) -> str:
    return SPACED_RULE_CODES.sub(
        lambda match: match.group(1) + "".join(f", {code}" for code in match.group(2).split()),
        text,
    )


def bare_directives(text: str) -> str:
    return "  ".join(match.group(0) for match in directive_matches(comma_separated(text)))


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
    if not isinstance(statement, ast.Expr):
        return False
    value = statement.value
    if isinstance(value, ast.Constant):
        return isinstance(value.value, str | bytes)
    return isinstance(value, ast.JoinedStr)


def has_side_effects(statement: ast.stmt) -> bool:
    return any(isinstance(node, SIDE_EFFECT_NODES) for node in ast.walk(statement))


def is_docstring_position(owner: ast.AST, block: list[ast.stmt], index: int) -> bool:
    return (
        index == 0
        and isinstance(owner, DOCSTRING_OWNERS)
        and block is getattr(owner, "body", None)
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
            if is_docstring_position(owner, block, index):
                what = owner_label(owner)
            else:
                what = "string literal statement used as prose"
            end = statement.end_lineno or statement.lineno
            if has_side_effects(statement):
                fix = SIDE_EFFECT_FIX
            else:
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


def check_file(path: Path, root: Path | None = None) -> list[Finding]:
    location = path if root is None else root / path
    try:
        with tokenize.open(location) as handle:
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
    return any(is_under(path, root) for root in (*TEMPORARY_EXCLUDED_PATHS, *VENDORED_PATHS))


def select_paths(
    tracked: Sequence[Path], roots: Sequence[Path], base: Path = Path()
) -> tuple[list[Path], list[Path]]:
    python = [path for path in tracked if path.suffix in PYTHON_SUFFIXES]
    if not roots:
        return sorted(path for path in python if not is_excluded(path)), []
    selected: set[Path] = set()
    unmatched: list[Path] = []
    for root in roots:
        matches = {path for path in python if is_under(path, root)}
        is_file = (base / root).is_file()
        if not matches and root.suffix in PYTHON_SUFFIXES and is_file:
            matches = {root}
        if not matches and not is_file:
            unmatched.append(root)
        selected |= matches
    return sorted(path for path in selected if not is_excluded(path)), unmatched


def normalize(raw: Path, cwd: Path, root: Path) -> Path:
    absolute = raw if raw.is_absolute() else cwd / raw
    try:
        return absolute.resolve().relative_to(root.resolve())
    except ValueError:
        return raw


def repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    )
    return Path(result.stdout.strip())


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True, text=True
    )
    return [Path(name) for name in result.stdout.split("\0") if name and (root / name).is_file()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_no_comments")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)
    try:
        root = repo_root()
        tracked = tracked_files(root)
    except (OSError, subprocess.CalledProcessError) as error:
        print(
            f".:0: NO_GIT: cannot list tracked files ({error})"
            " -> fix: run inside the repository",
            file=sys.stderr,
        )
        return 2
    roots = [normalize(raw, Path.cwd(), root) for raw in args.paths]
    selected, unmatched = select_paths(tracked, roots, root)
    for missing in unmatched:
        print(
            f"{missing.as_posix()}:0: NO_MATCH: no tracked Python file under this path"
            " -> fix: pass a tracked .py file or a directory that contains one",
            file=sys.stderr,
        )
    if unmatched:
        return 2
    findings = [finding for path in selected for finding in check_file(path, root)]
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
