from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.check_thread_starts import in_scope, thread_start_lines

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_flags_attribute_and_bare_thread_calls() -> None:
    source = (
        "import threading\n"
        "from threading import Thread\n"
        "threading.Thread(target=f)\n"
        "Thread(target=f)\n"
        "threading.Event()\n"
    )
    assert thread_start_lines(source) == [3, 4]


def test_scope_excludes_owner_and_non_worker() -> None:
    assert in_scope(Path("worker/runtime/worker.py"))
    assert not in_scope(Path("worker/runtime/threads.py"))
    assert not in_scope(Path("backend/app/main.py"))
    assert not in_scope(Path("worker/AGENTS.md"))


def test_repository_has_no_unbaselined_thread_starts() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_thread_starts.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
