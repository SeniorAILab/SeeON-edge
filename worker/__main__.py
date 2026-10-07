from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from types import FrameType

from pydantic import ValidationError

from shared.events.evidence_export_contract import DeliveryDisposition, DeliveryFailure
from shared.events.evidence_http_transport import (
    bounded_request,
    classify_http_failure,
    encode_json,
)
from shared.events.relay_failure_log import classify_relay_failure
from shared.release_identity import ReleaseIdentityMismatchError
from worker.adapters.model.in_process import InProcessServingClient
from worker.adapters.model.registry import flow_registry
from worker.runtime.bootstrap import REFUSE_TO_START_EXIT_CODE
from worker.runtime.config import (
    RELAY_HEARTBEAT_PATH,
    RELAY_TOKEN_ENV,
    RELAY_URL_ENV,
    ConfigSnapshot,
    LiveClipExportPolicy,
    RelayConfig,
    WorkerConfig,
    WorkerConfigError,
    WorkerConfigLkgStore,
    load_worker_config,
    load_worker_config_from_relay,
    make_restart_check,
    pull_worker_config_poll,
    reject_retired_worker_environment,
    resolve_config_path,
    resolve_local_overrides,
    resolve_startup_config,
)
from worker.runtime.config.release_pair import require_api_release_identity
from worker.runtime.provenance.environment import resolve_worker_build_revision
from worker.runtime.state_dir import resolve_state_dir
from worker.runtime.worker import WorkerRuntime

LOGGER = logging.getLogger(__name__)

CLEAN_SHUTDOWN_EXIT_CODE = 0
GENERIC_RUNTIME_ERROR_EXIT_CODE = 1
CONFIG_ERROR_EXIT_CODE = 2

_HEARTBEAT_ON_START_TIMEOUT_SEC = 0.5
_EDGE_RELAY_URL = "http://ml-api:8000"


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {raw!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {raw!r}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m worker",
        description="Eldercare fall/bed-exit worker inference pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "YAML camera roster path (developer/e2e escape hatch). Omit on a "
            "real Edge: the roster is pulled from ml-api. There is no "
            "environment-variable equivalent, by design."
        ),
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate config and exit without starting cameras (no side effects)",
    )
    parser.add_argument(
        "--heartbeat-on-start",
        action="store_true",
        help="Send one relay heartbeat per camera immediately at start",
    )
    parser.add_argument(
        "--max-frames-per-camera",
        type=_positive_int,
        default=None,
        help=(
            "Bounded-run cap: exit cleanly once every camera has processed "
            "this many frames (default: run indefinitely)"
        ),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help=(
            "Runtime-slot state root holding the durable delivery queue and "
            "GPU lease. Containers MUST pass the mounted volume path: the "
            "default resolves under the user's home, which inside a container "
            "is the writable layer, so every pending evidence envelope would be "
            "destroyed when the container is replaced "
            "(default: ~/.local/state/ml-worker)"
        ),
    )
    return parser


def _send_heartbeat_on_start(config: WorkerConfig) -> None:
    headers = {
        "Content-Type": "application/json",
        "X-Edge-Relay-Token": config.relay.token.get_secret_value(),
    }
    for camera in config.cameras:
        payload = {
            "camera_id": camera.camera_id,
            "facility_id": camera.facility_id,
            "config_version": config.version,
        }
        try:
            result = bounded_request(
                config.relay_heartbeat_url,
                "POST",
                headers,
                encode_json(payload),
                _HEARTBEAT_ON_START_TIMEOUT_SEC,
            )
        except Exception as exc:  # noqa: BLE001
            failure = DeliveryFailure(
                DeliveryDisposition.RETRY,
                "UNEXPECTED",
                transport_error=f"{type(exc).__name__}: {exc}",
            )
            _log_heartbeat_on_start_failure(camera.camera_id, failure)
            continue
        if isinstance(result, DeliveryFailure):
            _log_heartbeat_on_start_failure(camera.camera_id, result)
            continue
        status, headers_out, _body = result
        if not 200 <= status < 300:
            failure = classify_http_failure(status, headers_out)
            _log_heartbeat_on_start_failure(camera.camera_id, failure)


def _log_heartbeat_on_start_failure(camera_id: str, failure: DeliveryFailure) -> None:
    outcome = classify_relay_failure(failure)
    LOGGER.warning(
        "heartbeat-on-start POST %s -> %s (%s: %s) camera_id=%s",
        RELAY_HEARTBEAT_PATH,
        outcome.reason,
        outcome.failure_class.value,
        outcome.hint,
        camera_id,
        extra={"camera_id": camera_id, "relay_failure_class": outcome.failure_class.value},
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    state_dir = args.state_dir if args.state_dir is not None else resolve_state_dir()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        reject_retired_worker_environment(os.environ)
    except WorkerConfigError as exc:
        LOGGER.error("worker configuration refused: %s", exc)  # noqa: TRY400
        return CONFIG_ERROR_EXIT_CODE

    yaml_requested = args.config is not None
    snapshot: ConfigSnapshot | None = None

    if yaml_requested:
        try:
            yaml_config = load_worker_config(resolve_config_path(args.config))
        except WorkerConfigError:
            LOGGER.exception("config resolution failed")
            return CONFIG_ERROR_EXIT_CODE

        if args.check_config:
            LOGGER.info("config validation passed (%d camera(s))", len(yaml_config.cameras))
            return CLEAN_SHUTDOWN_EXIT_CODE

        relay_url = yaml_config.relay.url
        relay_token = (
            os.environ.get(RELAY_TOKEN_ENV, "").strip()
            or yaml_config.relay.token.get_secret_value()
        )
        try:
            require_api_release_identity(relay_url)
        except ReleaseIdentityMismatchError:
            LOGGER.exception("mixed edge image schema identity refused")
            return REFUSE_TO_START_EXIT_CODE
        except OSError:
            pass
        try:
            snapshot = resolve_startup_config(yaml_config, relay_url, relay_token)
        except (WorkerConfigError, ValidationError):
            LOGGER.exception("worker config resolution failed")
            return CONFIG_ERROR_EXIT_CODE
        config = snapshot.config
    else:
        relay_url = _EDGE_RELAY_URL
        relay_token = os.environ.get(RELAY_TOKEN_ENV, "").strip() or None

        if not relay_url:
            LOGGER.error(
                "worker config: no --config provided and %s is unset; cannot resolve a live config",
                RELAY_URL_ENV,
            )
            return CONFIG_ERROR_EXIT_CODE
        if not relay_token:
            LOGGER.error(
                "worker config: no --config provided and %s is unset; "
                "cannot authenticate with the relay",
                RELAY_TOKEN_ENV,
            )
            return CONFIG_ERROR_EXIT_CODE
        try:
            RelayConfig.model_validate({"url": relay_url, "token": relay_token})
        except ValidationError:
            LOGGER.exception(
                "worker config: %s %r is not a valid absolute HTTP(S) URL",
                RELAY_URL_ENV,
                relay_url,
            )
            return CONFIG_ERROR_EXIT_CODE

        if args.check_config:
            stored = WorkerConfigLkgStore(state_dir=state_dir).load()
            if stored is None:
                LOGGER.info(
                    "config validation passed (static check): baked relay endpoint "
                    "%s + %s set; no last-known-good cache yet -- the live pull "
                    "happens at boot",
                    relay_url,
                    RELAY_TOKEN_ENV,
                )
            else:
                LOGGER.info(
                    "config validation passed (static check): baked relay endpoint "
                    "%s + %s set; last-known-good cache present (registry_version=%d) "
                    "-- the live pull still happens at boot",
                    relay_url,
                    RELAY_TOKEN_ENV,
                    stored.registry_version,
                )
            return CLEAN_SHUTDOWN_EXIT_CODE

        try:
            models, clip, dev_mjpeg = resolve_local_overrides(None, os.environ)
        except (WorkerConfigError, ValidationError):
            LOGGER.exception("worker local model/clip configuration failed")
            return CONFIG_ERROR_EXIT_CODE
        try:
            require_api_release_identity(relay_url)
        except ReleaseIdentityMismatchError:
            LOGGER.exception("mixed edge image schema identity refused")
            return REFUSE_TO_START_EXIT_CODE
        except OSError:
            pass
        try:
            snapshot = load_worker_config_from_relay(
                relay_url, relay_token, models=models, clip=clip, dev_mjpeg=dev_mjpeg
            )
        except (WorkerConfigError, ValidationError):
            LOGGER.exception("worker config pull failed")
            return CONFIG_ERROR_EXIT_CODE
        if snapshot is None:
            LOGGER.error(
                "worker has no usable configuration: either the relay at %s "
                "was unreachable, or it returned a config with no usable "
                "camera (e.g. a camera registered without an RTSP URL) -- and "
                "no cached last-known-good config exists either. See the "
                "preceding stderr line ('request failed' vs 'malformed "
                "payload') for which. Fix relay reachability or %s/%s, finish "
                "camera setup in the dashboard, or provide --config as a "
                "fallback.",
                relay_url,
                RELAY_URL_ENV,
                RELAY_TOKEN_ENV,
            )
            return CONFIG_ERROR_EXIT_CODE
        config = snapshot.config

    LOGGER.info(
        "worker config resolved: source=%s stale=%s registry_version=%d directive=%s",
        snapshot.source,
        snapshot.stale,
        snapshot.registry_version,
        snapshot.directive,
    )

    if args.heartbeat_on_start:
        _send_heartbeat_on_start(config)

    relay_url = config.relay.url
    relay_token = (
        os.environ.get(RELAY_TOKEN_ENV, "").strip() or config.relay.token.get_secret_value()
    )
    clip_export_policy = LiveClipExportPolicy(
        config.clip_export_enabled,
        config.clip_export_version,
    )

    def _pull_and_apply_runtime_settings(url: str, token: str | None):
        polled = pull_worker_config_poll(url, token)
        if polled is None:
            return None
        clip_export_policy.apply(
            enabled=polled.clip_export_enabled,
            version=polled.clip_export_version,
        )
        return polled.restart_config

    restart_check = make_restart_check(
        relay_url,
        relay_token,
        snapshot.directive,
        pull_config=_pull_and_apply_runtime_settings,
    )

    profile_name = os.environ.get("ML_WORKER_PROFILE", "").strip()
    serving_registry = flow_registry() if profile_name == "flow" else None
    runtime = WorkerRuntime(
        config,
        serving_client=InProcessServingClient(serving_registry),
        restart_check=restart_check,
        clip_export_policy=clip_export_policy,
        max_frames_per_camera=args.max_frames_per_camera,
        state_dir=state_dir,
        restart_generation=snapshot.directive.generation,
        build_revision=resolve_worker_build_revision(os.environ.get("ML_WORKER_BUILD_REVISION")),
    )

    def _handle_signal(signum: int, frame: FrameType | None) -> None:
        del frame
        LOGGER.info("received signal %s; shutting down", signum)
        runtime.stop()

    previous_sigint = signal.signal(signal.SIGINT, _handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
    try:
        runtime.run()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else GENERIC_RUNTIME_ERROR_EXIT_CODE
    except Exception:
        LOGGER.exception("worker runtime error")
        return GENERIC_RUNTIME_ERROR_EXIT_CODE
    else:
        return CLEAN_SHUTDOWN_EXIT_CODE
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    sys.exit(main())
