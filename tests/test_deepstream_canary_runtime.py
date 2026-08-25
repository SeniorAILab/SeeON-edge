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


def test_authorization_rejects_camera_count_that_does_not_match_rung() -> None:
    from datetime import UTC, datetime, timedelta

    import pytest

    from worker.tools.deepstream_canary.authorization import AuthorizationRequest, authorize_rungs
    from worker.tools.deepstream_canary.models import AuthorizationArtifact, CanaryMode

    # Given: rung 4 authorization carrying only one camera identity.
    artifact = AuthorizationArtifact(
        schema_version=1,
        appliance_id="edge-lab-01",
        worker_image="sha256:" + "b" * 64,
        camera_ids=("camera-01",),
        owner="owner@example.invalid",
        issue="https://github.com/SeniorAILab/SeeON-edge/issues/381",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        authorized_rungs=(4,),
        projected_slack_mib=4096.0,
    )
    request = AuthorizationRequest(
        rungs=(4,),
        mode=CanaryMode.COMMISSIONING,
        appliance_id="edge-lab-01",
        worker_image_digest="sha256:" + "b" * 64,
        camera_ids=("camera-01",),
        now=datetime.now(UTC),
        minimum_projected_slack_mib=3072.0,
    )

    # When / Then: authorization fails before Docker.
    with pytest.raises(ValueError, match="camera_count_mismatch"):
        authorize_rungs(request, artifact)
