from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_publishers_wait_for_mediamtx_health_when_compose_is_rendered(tmp_path: Path) -> None:
    # Given: a fresh isolated run root.
    evidence = tmp_path / "canary"

    # When: one loopback publisher is generated.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "worker.tools.deepstream_canary",
            "run",
            "--rungs",
            "zero,loopback",
            "--evidence-dir",
            str(evidence),
            "--render-only",
            "--worker-image",
            "seeon-edge@sha256:" + "b" * 64,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # Then: publisher startup cannot race the digest-pinned RTSP server.
    assert completed.returncode == 0, completed.stderr
    rendered = (evidence / "compose.rendered.yaml").read_text()
    publisher = rendered.split("  publisher-01:", maxsplit=1)[1]
    assert "mediamtx:\n        condition: service_healthy" in publisher
