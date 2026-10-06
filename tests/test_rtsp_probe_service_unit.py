from __future__ import annotations

import json

import pytest

from backend.app.features.cameras.rtsp_probe_service import RTSPProbeInputs, probe_rtsp_url
from backend.app.features.cameras.store import ProbeResult


class _Endpoint:
    def __init__(self, original_url: str) -> None:
        self.original_url = original_url


def test_blank_origin_returns_probe_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.app.features.cameras.rtsp_probe_service.assert_rtsp_endpoint_allowed",
        lambda url: _Endpoint(url),
        raising=True,
    )
    result = probe_rtsp_url(
        RTSPProbeInputs(rtsp_url="rtsp://x", origin="  ", relay_token="t", timeout_s=1)
    )
    assert result == ProbeResult(ok=False, probe_unavailable=True)


def test_missing_token_returns_probe_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.app.features.cameras.rtsp_probe_service.assert_rtsp_endpoint_allowed",
        lambda url: _Endpoint(url),
        raising=True,
    )
    result = probe_rtsp_url(
        RTSPProbeInputs(rtsp_url="rtsp://x", origin="http://w", relay_token=None, timeout_s=1)
    )
    assert result == ProbeResult(ok=False, probe_unavailable=True)


def test_value_error_maps_to_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.app.features.cameras.rtsp_probe_service.assert_rtsp_endpoint_allowed",
        lambda url: (_ for _ in ()).throw(ValueError("bad url")),
        raising=True,
    )
    result = probe_rtsp_url(
        RTSPProbeInputs(rtsp_url="rtsp://x", origin="http://w", relay_token="t", timeout_s=1)
    )
    assert result == ProbeResult(ok=False, error_class="unsupported")


def test_timeout_is_probe_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.app.features.cameras.rtsp_probe_service.assert_rtsp_endpoint_allowed",
        lambda url: _Endpoint(url),
        raising=True,
    )

    def _raise(*_a, **_k):  # type: ignore[no-untyped-def]
        raise TimeoutError("deadline exceeded")

    monkeypatch.setattr("urllib.request.urlopen", _raise, raising=True)
    result = probe_rtsp_url(
        RTSPProbeInputs(rtsp_url="rtsp://x", origin="http://w", relay_token="t", timeout_s=1)
    )
    assert result == ProbeResult(ok=False, probe_unavailable=True)


def test_non_dict_payload_is_decode_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.app.features.cameras.rtsp_probe_service.assert_rtsp_endpoint_allowed",
        lambda url: _Endpoint(url),
        raising=True,
    )

    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_) -> None:  # noqa: ANN002, ANN003
            return None

        def read(self) -> bytes:
            return json.dumps([1, 2, 3]).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_, **__: _Resp(), raising=True)
    result = probe_rtsp_url(
        RTSPProbeInputs(rtsp_url="rtsp://x", origin="http://w", relay_token="t", timeout_s=1)
    )
    assert result == ProbeResult(ok=False, error_class="decode")


def test_worker_payload_maps_to_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.app.features.cameras.rtsp_probe_service.assert_rtsp_endpoint_allowed",
        lambda url: _Endpoint(url),
        raising=True,
    )

    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_) -> None:  # noqa: ANN002, ANN003
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"ok": False, "error_class": "timeout", "width": 640, "height": 480}
            ).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_, **__: _Resp(), raising=True)
    result = probe_rtsp_url(
        RTSPProbeInputs(rtsp_url="rtsp://x", origin="http://w", relay_token="t", timeout_s=1)
    )
    assert result == ProbeResult(ok=False, error_class="timeout", width=640, height=480)

