from pathlib import Path

from scripts.check_module_size import LIMIT, evaluate, logical_loc

REPO_ROOT = Path(__file__).resolve().parents[1]


def body(count: int) -> str:
    return "".join(f"x{i} = {i}\n" for i in range(count))


def test_imports_are_not_counted() -> None:
    assert logical_loc("import os\nfrom sys import path\nx = 1\n") == 1


def test_multiline_statement_counts_once() -> None:
    assert logical_loc("x = [\n    1,\n    2,\n]\n") == 1


def test_limit_boundary() -> None:
    assert evaluate({"a.py": logical_loc(body(LIMIT))}, {}) == []
    findings = evaluate({"a.py": logical_loc(body(LIMIT + 1))}, {})
    assert findings == [f"a.py:0: MODULE_TOO_LARGE: {LIMIT + 1} logical lines > {LIMIT}"]


def test_baseline_growth_fails() -> None:
    findings = evaluate({"a.py": 312}, {"a.py": 300})
    assert findings == ["a.py:0: MODULE_TOO_LARGE: 312 logical lines > 250 (baseline 300)"]


def test_baseline_at_value_passes() -> None:
    assert evaluate({"a.py": 300}, {"a.py": 300}) == []


def test_baseline_shrink_is_reported() -> None:
    [finding] = evaluate({"a.py": 280}, {"a.py": 300})
    assert "SHRINK_BASELINE" in finding
    assert '"a.py": 280,' in finding
    [finding] = evaluate({"a.py": 250}, {"a.py": 300})
    assert 'remove "a.py"' in finding


def test_repo_tree_passes() -> None:
    from scripts.check_module_size import main

    assert main([]) == 0
