import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_no_comments import (
    DIRECTIVE_WITH_PROSE,
    MYPY_ERROR_CODES,
    NO_COMMENT,
    NO_DOCSTRING,
    SIDE_EFFECT_FIX,
    TEMPORARY_EXCLUDED_PATHS,
    UNPARSEABLE,
    VENDORED_PATHS,
    check_file,
    check_source,
    select_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_no_comments.py"
SAMPLE = Path("sample.py")
EXCLUDED_ROUTER = TEMPORARY_EXCLUDED_PATHS[0] / "router.py"
VENDORED_FILE = VENDORED_PATHS[0]


def codes(source: str) -> list[tuple[int, str]]:
    return [(finding.line, finding.code) for finding in check_source(SAMPLE, source)]


def git_repo(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def run_checker(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_code_passes() -> None:
    assert codes("x = 1\n\n\ndef f(a: int) -> int:\n    return a + 1\n") == []


def test_hash_inside_a_string_is_not_a_comment() -> None:
    assert codes('x = "# not a comment"\n') == []


@pytest.mark.parametrize(
    "directive",
    [
        "# noqa",
        "# noqa: BLE001",
        "# noqa: BLE001, S110",
        "# noqa: BLE001,S110",
        "# noqa:E501",
        "# ruff: noqa: F401",
        "# ruff: noqa",
        "# type: ignore",
        "# type: ignore[attr-defined]",
        "# type: ignore[attr-defined, union-attr]",
        "# type: ignore[no-any-return]",
        "# pragma: no cover",
        "# pyright: ignore",
        "# pyright: ignore[reportPrivateUsage]",
        "# pyright: ignore[reportCallIssue, reportArgumentType]",
        "# pyright: basic",
        "# pyright: standard",
        "# pyright: strict",
        "# pyright: reportMissingImports=false",
        "# pyright: reportMissingImports=false, reportPrivateUsage=error",
        "# fmt: off",
        "# fmt: on",
        "# fmt: skip",
        "# type: ignore[misc]  # noqa: E501",
    ],
)
def test_bare_tool_directives_pass(directive: str) -> None:
    assert codes(f"x = 1  {directive}\n") == []


@pytest.mark.parametrize(
    ("comment", "bare"),
    [
        ("# noqa: BLE001 - a send failure is resent", "# noqa: BLE001"),
        ("# noqa: BLE001,S110 - probe must not break startup", "# noqa: BLE001,S110"),
        ("# noqa: F401 see711", "# noqa: F401"),
        ("# noqa: E501 S110", "# noqa: E501, S110"),
        ("# noqa: BLE001 S110 TRY300", "# noqa: BLE001, S110, TRY300"),
        ("# noqa: BLE001 S110 - best effort", "# noqa: BLE001, S110"),
        ("# type: ignore[attr-defined] vendor stub", "# type: ignore[attr-defined]"),
        ("# pragma: no cover because unreachable", "# pragma: no cover"),
        ("# pragma: no cover -- local invariant", "# pragma: no cover"),
        ("# fmt: off keep the table aligned", "# fmt: off"),
        ("# see below  # noqa: E501", "# noqa: E501"),
    ],
)
def test_directive_with_prose_fails_and_names_the_bare_directive(comment: str, bare: str) -> None:
    findings = check_source(SAMPLE, f"x = 1  {comment}\n")
    assert [(finding.line, finding.code) for finding in findings] == [(1, DIRECTIVE_WITH_PROSE)]
    assert findings[0].fix == f"replace the comment with `{bare}`"


@pytest.mark.parametrize(
    "comment",
    [
        "# noqa: see711",
        "# noqa: issue42 ticket7",
        "# noqa: f401",
        "# noqa: E501,",
        "# NOQA: E501",
        "# type: ignore[vendor-stub-is-wrong-here]",
        "# type: ignore[attr-defined, made-up-code]",
        "# type: ignore[Attr-Defined]",
        "# type: ignore[attr_defined]",
        "# pyright: because=reasons",
        "# pyright: ignore[because reasons]",
        "# pyright: ignore[reportfoo]",
        "# pyright: reportMissingImports=maybe",
        "# pyright: lenient",
        "# fmt: skip-this",
    ],
)
def test_prose_hidden_inside_directive_syntax_fails(comment: str) -> None:
    assert codes(f"x = 1  {comment}\n") == [(1, NO_COMMENT)]


@pytest.mark.parametrize(
    "comment",
    [
        "# pragma: no branch",
        "# type: list[int]",
        "# nosec",
        "# nosec B602",
        "# mypy: ignore-errors",
        "# pylint: disable=unused-import",
        "# isort: skip",
        "# flake8: noqa",
    ],
)
def test_directive_kinds_not_used_in_the_repo_are_not_allowed(comment: str) -> None:
    assert codes(f"x = 1  {comment}\n") == [(1, NO_COMMENT)]


def test_type_ignore_codes_used_in_the_repo_are_known_mypy_codes() -> None:
    used = {
        "arg-type",
        "attr-defined",
        "index",
        "assignment",
        "union-attr",
        "method-assign",
        "call-arg",
        "operator",
        "misc",
        "return-value",
        "no-untyped-def",
        "call-overload",
        "list-item",
        "name-defined",
    }
    assert used <= MYPY_ERROR_CODES
    assert all(code == code.lower() and " " not in code for code in MYPY_ERROR_CODES)


def test_prose_comment_on_its_own_line_fails() -> None:
    findings = check_source(SAMPLE, "# explain why\nx = 1\n")
    assert [(finding.line, finding.code, finding.fix) for finding in findings] == [
        (1, NO_COMMENT, "delete this comment line")
    ]


def test_render_names_path_line_code_and_exact_fix() -> None:
    finding = check_source(Path("pkg/mod.py"), "x = 1  # why\n")[0]
    assert finding.render() == (
        "pkg/mod.py:1: NO_COMMENT: comment `# why`"
        " -> fix: delete the trailing comment, keep the code"
    )


def test_shebang_on_line_one_and_coding_on_line_two_pass() -> None:
    assert codes("#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nx = 1\n") == []


@pytest.mark.parametrize(
    "line",
    [
        "# -*- coding: utf-8 -*-",
        "# coding: utf-8",
        "# coding=utf-8",
        "# vim: set fileencoding=utf-8 :",
    ],
)
def test_exact_pep_263_forms_pass_on_line_one_or_two(line: str) -> None:
    assert codes(f"{line}\nx = 1\n") == []
    assert codes(f"#!/usr/bin/env python3\n{line}\nx = 1\n") == []


@pytest.mark.parametrize(
    ("line", "code"),
    [
        ("# -*- coding: utf-8 -*-  legacy header kept for old editors", NO_COMMENT),
        ("# Transcoding: utf-8 is what the camera sends", NO_COMMENT),
        ("# coding: utf-8 because of the Korean labels", NO_COMMENT),
        ("#coding:utf-8", NO_COMMENT),
        ("# -*- coding: utf-8 -*- # noqa: E501", DIRECTIVE_WITH_PROSE),
    ],
)
def test_coding_lines_with_prose_or_inexact_form_fail(line: str, code: str) -> None:
    assert codes(f"{line}\nx = 1\n") == [(1, code)]


def test_shebang_after_line_one_is_a_comment() -> None:
    assert codes("x = 1\n#!/usr/bin/env python3\n") == [(2, NO_COMMENT)]


def test_coding_line_after_line_two_is_a_comment() -> None:
    assert codes("x = 1\ny = 2\n# -*- coding: utf-8 -*-\n") == [(3, NO_COMMENT)]


def test_module_class_and_function_docstrings_fail() -> None:
    source = (
        '"""Module."""\n\n\nclass A:\n    """Class."""\n\n    def f(self):\n'
        '        """Function."""\n        return 1\n\n\nasync def g():\n    """Async."""\n'
    )
    findings = check_source(SAMPLE, source)
    assert [(finding.line, finding.code, finding.what) for finding in findings] == [
        (1, NO_DOCSTRING, "module docstring"),
        (5, NO_DOCSTRING, "class `A` docstring"),
        (8, NO_DOCSTRING, "function `f` docstring"),
        (13, NO_DOCSTRING, "function `g` docstring"),
    ]


@pytest.mark.parametrize(
    ("source", "line", "what"),
    [
        ('f"""Module {1}."""\n', 1, "module docstring"),
        ('b"""Module."""\n', 1, "module docstring"),
        ('rb"""Module."""\n', 1, "module docstring"),
        ("'Module.'\n", 1, "module docstring"),
        ('class A:\n    f"""Class {1}."""\n', 2, "class `A` docstring"),
        ("class A:\n    b'Class.'\n", 2, "class `A` docstring"),
        ('async def g():\n    b"""Async."""\n', 2, "function `g` docstring"),
        ("def f():\n    'Single quoted.'\n    return 1\n", 2, "function `f` docstring"),
        (
            'def outer():\n    def inner():\n        f"""Nested {2}."""\n    return inner\n',
            3,
            "function `inner` docstring",
        ),
        (
            'class A:\n    def m(self):\n        rb"""Method."""\n',
            3,
            "function `m` docstring",
        ),
    ],
)
def test_string_ish_docstrings_of_every_owner_fail(source: str, line: int, what: str) -> None:
    findings = check_source(SAMPLE, source)
    assert [(finding.line, finding.code, finding.what) for finding in findings] == [
        (line, NO_DOCSTRING, what)
    ]


def test_bytes_and_f_string_statements_after_code_fail() -> None:
    assert codes('x = 1\nb"prose"\nf"prose {x}"\n') == [(2, NO_DOCSTRING), (3, NO_DOCSTRING)]


def test_f_string_docstring_with_calls_is_reported_with_a_fix_that_keeps_the_calls() -> None:
    findings = check_source(SAMPLE, 'def f():\n    f"""{record()}"""\n')
    assert [(finding.line, finding.code, finding.fix) for finding in findings] == [
        (2, NO_DOCSTRING, SIDE_EFFECT_FIX)
    ]


def test_a_string_used_as_a_value_is_not_a_docstring() -> None:
    assert codes('X = """not a docstring"""\n\n\ndef f():\n    return f"{X}"\n') == []


def test_docstring_that_is_the_whole_body_gets_a_fix_that_keeps_the_block() -> None:
    findings = check_source(SAMPLE, 'def f():\n    """One.\n\n    Two.\n    """\n')
    assert findings[0].fix == (
        "delete lines 2-5 and put `...` in their place so the block stays non-empty"
    )


def test_string_statement_after_code_fails() -> None:
    assert codes('x = 1\n"""prose"""\n') == [(2, NO_DOCSTRING)]


def test_unparseable_source_is_reported() -> None:
    assert codes("def f(:\n") == [(1, UNPARSEABLE)]


def test_select_paths_without_roots_takes_tracked_python_minus_temporary_exclusions() -> None:
    tracked = [Path("a.py"), Path("b/c.pyi"), Path("README.md"), EXCLUDED_ROUTER, VENDORED_FILE]
    assert select_paths(tracked, []) == ([Path("a.py"), Path("b/c.pyi")], [])


def test_select_paths_with_roots_checks_only_those_roots() -> None:
    tracked = [Path("scripts/x.py"), Path("tests/test_x.py"), Path("worker/y.py")]
    roots = [Path("scripts"), Path("tests/test_x.py")]
    assert select_paths(tracked, roots) == ([Path("scripts/x.py"), Path("tests/test_x.py")], [])


def test_select_paths_reports_a_root_without_python_files() -> None:
    assert select_paths([Path("a.py")], [Path("missing")]) == ([], [Path("missing")])


def test_select_paths_applies_the_temporary_exclusion_to_explicit_roots() -> None:
    assert select_paths([EXCLUDED_ROUTER], [EXCLUDED_ROUTER]) == ([], [])


def test_vendored_file_is_excluded_even_when_named_explicitly() -> None:
    sibling = VENDORED_FILE.with_name("sibling.py")
    tracked = [VENDORED_FILE, sibling]
    assert select_paths(tracked, [VENDORED_FILE]) == ([], [])
    assert select_paths(tracked, [VENDORED_FILE.parent]) == ([sibling], [])


def test_vendored_paths_are_kept_apart_from_the_temporary_exclusions() -> None:
    assert not set(VENDORED_PATHS) & set(TEMPORARY_EXCLUDED_PATHS)


def test_checker_and_its_tests_have_no_comments_or_docstrings() -> None:
    assert check_file(CHECKER) == []
    assert check_file(Path(__file__)) == []


def test_cli_checks_tracked_files_and_honours_path_arguments(tmp_path: Path) -> None:
    repo = git_repo(
        tmp_path,
        {
            "clean.py": "x = 1\n",
            "dirty/mod.py": "y = 2  # why\n",
            EXCLUDED_ROUTER.as_posix(): "z = 3  # excluded until the camera PR lands\n",
        },
    )
    full = run_checker(repo)
    assert full.returncode == 1
    assert full.stdout.splitlines() == [
        (
            "dirty/mod.py:1: NO_COMMENT: comment `# why`"
            " -> fix: delete the trailing comment, keep the code"
        )
    ]
    assert run_checker(repo, "clean.py").returncode == 0
    assert run_checker(repo, "dirty").returncode == 1


def test_cli_keeps_exclusions_anchored_to_the_repo_root_from_a_subdirectory(
    tmp_path: Path,
) -> None:
    repo = git_repo(
        tmp_path,
        {
            "backend/app/clean.py": "x = 1\n",
            "backend/app/dirty.py": "y = 2  # why\n",
            EXCLUDED_ROUTER.as_posix(): "z = 3  # excluded until the camera PR lands\n",
            VENDORED_FILE.as_posix(): '"""Vendored."""\n',
        },
    )
    subdirectory = repo / "backend"
    full = run_checker(subdirectory)
    assert full.returncode == 1
    assert [line.split(":")[0] for line in full.stdout.splitlines()] == ["backend/app/dirty.py"]
    assert run_checker(subdirectory, "app/features").returncode == 0
    assert run_checker(subdirectory, "app/clean.py").returncode == 0
    assert run_checker(subdirectory, "app/dirty.py").returncode == 1


def test_cli_skips_non_python_files_instead_of_failing(tmp_path: Path) -> None:
    repo = git_repo(
        tmp_path,
        {"clean.py": "x = 1\n", "README.md": "# Title\n", "tool": "#!/usr/bin/env python3\n"},
    )
    result = run_checker(repo, "README.md", "tool", "clean.py")
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_cli_rejects_a_root_without_python_files(tmp_path: Path) -> None:
    repo = git_repo(tmp_path, {"clean.py": "x = 1\n"})
    result = run_checker(repo, "nowhere")
    assert result.returncode == 2
    assert "nowhere:0: NO_MATCH" in result.stderr
