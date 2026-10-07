from __future__ import annotations

from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
SCRIPTS_DIR: Final = REPO_ROOT / "scripts"

PIPE_BUF: Final = 512

UNPINNED_SHEBANG: Final = "#!/usr/bin/env bash"


def _shell_scripts() -> list[Path]:
    return sorted(SCRIPTS_DIR.rglob("*.sh"))


def _heredoc_bodies(text: str) -> list[tuple[int, int]]:
    bodies: list[tuple[int, int]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        marker = line.rfind("<<")
        if marker == -1:
            index += 1
            continue
        tag = line[marker + 2 :].strip()
        if tag.startswith("-"):
            tag = tag[1:].strip()
        tag = tag.strip("'\"")
        if not tag or not tag.replace("_", "").isalnum() or line[marker + 2 : marker + 3] == "<":
            index += 1
            continue
        start = index + 1
        size = 0
        index += 1
        while index < len(lines) and lines[index].strip() != tag:
            size += len(lines[index].encode("utf-8")) + 1
            index += 1
        bodies.append((start, size))
        index += 1
    return bodies


def test_scripts_with_large_heredocs_pin_their_interpreter() -> None:
    offenders: list[str] = []

    for script in _shell_scripts():
        text = script.read_text(encoding="utf-8")
        if not text.startswith(UNPINNED_SHEBANG):
            continue
        for start_line, size in _heredoc_bodies(text):
            if size > PIPE_BUF:
                offenders.append(
                    f"{script.relative_to(REPO_ROOT)}:{start_line} "
                    f"has a {size}-byte heredoc but uses {UNPINNED_SHEBANG}"
                )

    assert not offenders, (
        "These scripts deadlock under Homebrew bash 5.3.15, which writes heredoc "
        f"bodies into a pipe before exec'ing the reader (>{PIPE_BUF} bytes blocks "
        "forever). Pin the shebang to #!/bin/bash, or move the body into a file. "
        "See issue #9.\n  " + "\n  ".join(offenders)
    )


def test_heredoc_measurement_finds_the_known_sizes() -> None:
    body = "\n".join(f"line {n}" for n in range(10))
    script = f"f() {{\n  cat <<'PY'\n{body}\nPY\n}}\n"

    found = _heredoc_bodies(script)

    assert len(found) == 1
    _start, size = found[0]
    assert size == len(body.encode("utf-8")) + 1


def test_guard_would_have_caught_the_original_defect() -> None:
    oversized = "\n".join("x" * 60 for _ in range(12))
    text = f"{UNPINNED_SHEBANG}\nf() {{\n  python3 - <<'PY'\n{oversized}\nPY\n}}\n"

    sizes = [size for _start, size in _heredoc_bodies(text)]

    assert sizes and max(sizes) > PIPE_BUF
