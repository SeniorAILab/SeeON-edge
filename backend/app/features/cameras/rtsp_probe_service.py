"""RTSP probe logic extracted from the cameras router into a FastAPI-free service.

Frozen input/output types and pure function behavior keep this boundary stable and
unit-testable without a running app or database.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from backend.app.features.cameras.store import ProbeErrorClass, ProbeResult
from shared.rtsp_url_policy import assert_rtsp_endpoint_allowed

# Duplicated constant to avoid importing the router (which pulls FastAPI).
RELAY_TOKEN_HEADER = "X-Edge-Relay-Token"
PROBE_PATH = "/probe"


@dataclass(frozen=True, slots=True)
class RTSPProbeInputs:
    rtsp_url: str
    origin: str
    relay_token: str | None
    timeout_s: int


def _optional_positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _probe_result_from_worker(payload: dict[object, object]) -> ProbeResult:
    raw_error_class = payload.get("error_class")
    error_class: ProbeErrorClass | None
    if raw_error_class == "timeout":
        error_class = "timeout"
    elif raw_error_class == "decode":
        error_class = "decode"
    elif raw_error_class == "auth":
        error_class = "auth"
    elif raw_error_class == "unsupported":
        error_class = "unsupported"
    elif raw_error_class == "unavailable":
        error_class = "unavailable"
    else:
        error_class = None
    width = _optional_positive_int(payload.get("width"))
    height = _optional_positive_int(payload.get("height"))
    return ProbeResult(
        ok=payload.get("ok") is True,
        error_class=error_class,
        width=width,
        height=height,
    )


def probe_rtsp_url(inputs: RTSPProbeInputs) -> ProbeResult:
    """Probe an RTSP URL via the worker's /probe endpoint.

    Mirrors the legacy router behavior byte-for-byte:
    - Admission policy re-checks and maps ValueError to unsupported.
    - Blank origin or missing relay token => probe_unavailable=True.
    - Timeout/OS/HTTP/JSON errors => probe_unavailable=True.
    - Non-dict payload => error_class=decode.
    """
    try:
        endpoint = assert_rtsp_endpoint_allowed(inputs.rtsp_url)
    except ValueError:
        return ProbeResult(ok=False, error_class="unsupported")
    rtsp_url = endpoint.original_url
    origin = inputs.origin.strip().rstrip("/")
    if not origin:
        return ProbeResult(ok=False, probe_unavailable=True)
    if inputs.relay_token is None:
        return ProbeResult(ok=False, probe_unavailable=True)

    body = json.dumps({"rtsp_url": rtsp_url}, separators=(",", ":")).encode("utf-8")
    probe_request = urllib.request.Request(
        f"{origin}{PROBE_PATH}",
        data=body,
        headers={"Content-Type": "application/json", RELAY_TOKEN_HEADER: inputs.relay_token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(probe_request, timeout=inputs.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except TimeoutError:
        return ProbeResult(ok=False, probe_unavailable=True)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return ProbeResult(ok=False, probe_unavailable=True)
    if not isinstance(payload, dict):
        return ProbeResult(ok=False, error_class="decode")
    return _probe_result_from_worker(payload)


__all__ = ["RTSPProbeInputs", "probe_rtsp_url"]

