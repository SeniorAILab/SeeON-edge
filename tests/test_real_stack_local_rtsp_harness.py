"""Harness contract: loopback RTSP allowance is scoped to ``real_stack`` only."""

from __future__ import annotations

import os

import pytest

from shared.rtsp_url_policy import ALLOW_LOCAL_RTSP_ENV, reject_rtsp_url_reason

LOOPBACK_RTSP_URL = "rtsp://127.0.0.1:8554/live"


@pytest.fixture
def allow_local_env_at_fixture_setup() -> str | None:
    return os.environ.get(ALLOW_LOCAL_RTSP_ENV)


@pytest.mark.real_stack
def test_real_stack_sees_local_rtsp_allowance_before_fixture_setup(
    allow_local_env_at_fixture_setup: str | None,
) -> None:
    assert allow_local_env_at_fixture_setup == "1"
    assert os.environ.get(ALLOW_LOCAL_RTSP_ENV) == "1"
    assert reject_rtsp_url_reason(LOOPBACK_RTSP_URL) is None


def test_non_real_stack_observes_local_rtsp_denied_and_unset(
    allow_local_env_at_fixture_setup: str | None,
) -> None:
    assert allow_local_env_at_fixture_setup is None
    assert os.environ.get(ALLOW_LOCAL_RTSP_ENV) is None
    reason = reject_rtsp_url_reason(LOOPBACK_RTSP_URL)
    assert reason is not None
    assert "loopback" in reason
