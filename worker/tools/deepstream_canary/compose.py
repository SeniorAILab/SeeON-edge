"""Render the standalone canary Compose project and run-local worker config."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from worker.tools.deepstream_canary.report import write_receipt_manifest

PROJECT_NAME: Final = "seeon-ds-canary"
SUPPORT_DIR: Final = Path("scripts/qa/deepstream-canary")
BASE_COMPOSE: Final = SUPPORT_DIR / "compose.canary.yaml"


@dataclass(frozen=True, slots=True)
class RenderRequest:
    evidence_dir: Path
    worker_image: str
    relay_token: str
    camera_count: int
    model_dir: Path


def _publisher_block(camera_count: int, worker_image: str, corpus_dir: Path) -> str:
    blocks: list[str] = []
    for index in range(1, camera_count + 1):
        camera = f"loop-{index:02d}"
        offset_ms = (index - 1) * 67
        blocks.append(
            f"  publisher-{index:02d}:\n"
            f"    image: {worker_image}\n"
            "    pull_policy: never\n"
            "    restart: \"no\"\n"
            "    depends_on:\n      mediamtx:\n        condition: service_healthy\n"
            "    networks: [canary]\n"
            f"    volumes:\n      - {corpus_dir}:/corpus:ro\n"
            "    command: [ffmpeg, -re, -stream_loop, \"-1\", "
            f"-itsoffset, \"0.{offset_ms:03d}\", -i, /corpus/loopback.mp4, "
            "-map, \"0:v:0\", -c:v, copy, -f, rtsp, -rtsp_transport, tcp, "
            f"rtsp://mediamtx:8554/{camera}]\n"
            "    cap_drop: [ALL]\n"
            "    security_opt: [no-new-privileges:true]\n"
        )
    return "".join(blocks)


def _worker_config(camera_count: int, relay_token: str) -> bytes:
    cameras = "".join(
        f"  - camera_id: loop-{index:02d}\n"
        "    facility_id: canary-facility\n"
        f"    resident_id: canary-resident-{index:02d}\n"
        f"    rtsp_url: rtsp://mediamtx:8554/loop-{index:02d}\n"
        "    heartbeat_interval_sec: 10\n"
        "    frame_stride: 1\n"
        f"    label: Canary {index:02d}\n"
        for index in range(1, camera_count + 1)
    )
    return (
        "version: 1\n"
        "relay:\n  url: http://ml-api:8000\n"
        f"  token: {relay_token}\n"
        "runtime:\n  max_failures: 3\n  open_timeout_ms: 5000\n  read_timeout_ms: 5000\n"
        "models:\n"
        "  fall:\n"
        "    type: lstm\n    framework: pytorch\n    mode: sequence\n"
        "    artifact_dir: /app/models/fall/lstm\n"
        "    weights: model.pt\n    architecture: arch.json\n    metadata: metadata.yaml\n"
        "    window: 30\n    stride: 5\n    input_shape: [30, 51]\n"
        "    operating_threshold: 0.5\n    schema_version: 1\n"
        "    preprocessing_identity: legacy-coco17-xyc-frame-normalized-zero-fill-v1\n"
        "domains:\n  fall:\n    enabled: true\n  bed_exit:\n    enabled: true\n"
        "    night_window:\n      start: '21:00'\n      end: '05:00'\n      tz: Asia/Seoul\n"
        "clip:\n  enabled: true\n"
        f"cameras:\n{cameras}"
    ).encode()


def render_compose(request: RenderRequest) -> tuple[Path, str]:
    """Create one self-contained Compose file with one publisher per camera."""
    root = request.evidence_dir.resolve()
    run_dir = root / "run"
    paths = {
        "CANARY_ASSET_DIR": SUPPORT_DIR.resolve(),
        "CANARY_RECEIPT_DIR": root / "raw",
        "CANARY_MODEL_DIR": run_dir / "models",
        "CANARY_STATE_DIR": run_dir / "state",
        "CANARY_SOCKET_DIR": run_dir / "sockets",
        "CANARY_ENGINE_DIR": run_dir / "engine-cache",
        "CANARY_SCRATCH_DIR": run_dir / "scratch",
        "CANARY_CLIP_DIR": run_dir / "clips",
        "CANARY_CONFIG_PATH": run_dir / "worker.yaml",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for key in (
        "CANARY_RECEIPT_DIR",
        "CANARY_STATE_DIR",
        "CANARY_SOCKET_DIR",
        "CANARY_ENGINE_DIR",
        "CANARY_SCRATCH_DIR",
        "CANARY_CLIP_DIR",
    ):
        paths[key].mkdir(parents=True, exist_ok=True, mode=0o700)
    if paths["CANARY_MODEL_DIR"].exists():
        raise FileExistsError(paths["CANARY_MODEL_DIR"])
    shutil.copytree(request.model_dir.resolve(), paths["CANARY_MODEL_DIR"])
    paths["CANARY_CONFIG_PATH"].write_bytes(
        _worker_config(request.camera_count, request.relay_token)
    )
    template = BASE_COMPOSE.read_text(encoding="utf-8")
    replacements = {
        "CANARY_WORKER_IMAGE": request.worker_image,
        **{name: str(path) for name, path in paths.items()},
    }
    for name, value in replacements.items():
        template = template.replace(f"${{{name}}}", value)
    publisher = _publisher_block(
        request.camera_count, request.worker_image, paths["CANARY_SCRATCH_DIR"]
    )
    rendered = template.replace("\nnetworks:\n", f"{publisher}\nnetworks:\n")
    compose_path = root / "compose.rendered.yaml"
    compose_path.write_text(rendered, encoding="utf-8")
    compose_digest = hashlib.sha256(compose_path.read_bytes()).hexdigest()
    _ = write_receipt_manifest(root, (compose_path, paths["CANARY_CONFIG_PATH"]))
    return compose_path, compose_digest
