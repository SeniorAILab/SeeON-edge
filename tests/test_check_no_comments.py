import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_no_comments import (
    DIRECTIVE_WITH_PROSE,
    NO_COMMENT,
    NO_DOCSTRING,
    TEMPORARY_EXCLUDED_PATHS,
    UNPARSEABLE,
    check_file,
    check_source,
    select_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_no_comments.py"
SAMPLE = Path("sample.py")
EXCLUDED_ROUTER = TEMPORARY_EXCLUDED_PATHS[0] / "router.py"


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
        "# noqa:E501",
        "# noqa: E501 S110",
        "# ruff: noqa: F401",
        "# type: ignore",
        "# type: ignore[attr-defined]",
        "# type: ignore[attr-defined, union-attr]",
        "# pragma: no cover",
        "# pyright: ignore[reportPrivateUsage]",
        "# pyright: strict",
        "# isort: skip",
        "# fmt: off",
        "# fmt: on",
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
        ("# type: ignore[attr-defined] vendor stub", "# type: ignore[attr-defined]"),
        ("# pragma: no cover because unreachable", "# pragma: no cover"),
        ("# fmt: off keep the table aligned", "# fmt: off"),
        ("# see below  # noqa: E501", "# noqa: E501"),
    ],
)
def test_directive_with_prose_fails_and_names_the_bare_directive(comment: str, bare: str) -> None:
    findings = check_source(SAMPLE, f"x = 1  {comment}\n")
    assert [(finding.line, finding.code) for finding in findings] == [(1, DIRECTIVE_WITH_PROSE)]
    assert findings[0].fix == f"replace the comment with `{bare}`"


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
    tracked = [Path("a.py"), Path("b/c.pyi"), Path("README.md"), EXCLUDED_ROUTER]
    assert select_paths(tracked, []) == ([Path("a.py"), Path("b/c.pyi")], [])


def test_select_paths_with_roots_checks_only_those_roots() -> None:
    tracked = [Path("scripts/x.py"), Path("tests/test_x.py"), Path("worker/y.py")]
    roots = [Path("scripts"), Path("tests/test_x.py")]
    assert select_paths(tracked, roots) == ([Path("scripts/x.py"), Path("tests/test_x.py")], [])


def test_select_paths_reports_a_root_without_python_files() -> None:
    assert select_paths([Path("a.py")], [Path("missing")]) == ([], [Path("missing")])


def test_select_paths_applies_the_temporary_exclusion_to_explicit_roots() -> None:
    assert select_paths([EXCLUDED_ROUTER], [EXCLUDED_ROUTER]) == ([], [])


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


def test_cli_rejects_a_root_without_python_files(tmp_path: Path) -> None:
    repo = git_repo(tmp_path, {"clean.py": "x = 1\n"})
    result = run_checker(repo, "nowhere")
    assert result.returncode == 2
    assert "nowhere:0: NO_MATCH" in result.stderr
