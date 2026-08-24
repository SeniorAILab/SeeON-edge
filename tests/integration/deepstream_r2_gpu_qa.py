"""Real-GPU round-2 QA: latest-only, two-source readiness, and EOS rebuild."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import torch

from worker.native.deepstream.control import ControlIdentity, DeepStreamControlClient
from worker.native.deepstream.metadata import SourceBinding
from worker.runtime.deepstream import ChildConfig
from worker.runtime.deepstream.readiness import wait_for_ready
from worker.runtime.deepstream.supervisor import DeepStreamChildSupervisor
from worker.runtime.deepstream.transport import spawn_child


def config(root: Path, *, qa_mode: bool) -> ChildConfig:
    return ChildConfig(
        executable=Path("/usr/local/bin/seeon-deepstream-child"),
        worker_boot_id=uuid.uuid4(),
        socket_dir=root / "ipc",
        first_fault_path=root / "first-fault.json",
        lease_state_dir=root,
        startup_timeout_sec=10.0,
        stop_timeout_sec=2.0,
        qa_mode=qa_mode,
    )


def main() -> int:
    if os.getpid() != 1:
        raise RuntimeError("GPU QA must run in an isolated PID namespace as PID 1")
    root = Path("/tmp/seeon-r2-gpu")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(mode=0o700)

    raw_config = config(root / "latest", qa_mode=True)
    transport = spawn_child(raw_config)
    if not wait_for_ready(transport.ready_fd, transport.process, 10.0):
        raise RuntimeError("native ready handshake failed")
    arguments = Path(f"/proc/{transport.process.pid}/cmdline").read_bytes().split(b"\0")
    inherited = dict(zip(arguments[1::2], arguments[2::2], strict=False))
    for flag in (b"--control-fd", b"--wake-fd", b"--failure-fd"):
        descriptor = inherited[flag].decode()
        fdinfo = Path(f"/proc/{transport.process.pid}/fdinfo/{descriptor}").read_text()
        flags_line = next(line for line in fdinfo.splitlines() if line.startswith("flags:"))
        if int(flags_line.split()[1], 8) & os.O_CLOEXEC == 0:
            raise RuntimeError(f"inherited descriptor lacks CLOEXEC: {flag.decode()}")
    control = DeepStreamControlClient(
        transport.control,
        ControlIdentity(
            raw_config.worker_boot_id,
            raw_config.child_instance_id,
            "seeon-perception-v1",
        ),
        timeout_sec=5.0,
    )
    control.connect()
    _ = control.status()
    _ = control.add_source("camera-a", "loopback://camera-a")
    _ = control.add_source("camera-b", "loopback://camera-b")
    control.wait_for_publish(10)
    paused = control.status()
    latest_a = control.pull_latest("camera-a")
    latest_b = control.pull_latest("camera-b")
    if latest_a is None or latest_b is None or paused.metadata_overwritten < 8:
        raise RuntimeError("capacity-one latest behavior failed")
    control.remove_source("camera-a")
    readded = control.add_source("camera-a", "loopback://camera-a")
    if readded.source_generation != 2 or readded.stream_epoch != 1:
        raise RuntimeError("real-child remove/tombstone/re-add lifecycle failed")
    reconnected = control.source_failure("camera-a", "eos")
    if reconnected.source_generation != 2 or reconnected.stream_epoch != 2:
        raise RuntimeError("real-child reconnect epoch lifecycle failed")
    control.shutdown()
    if transport.process.wait(timeout=5.0) != 0:
        raise RuntimeError("raw child did not stop cleanly")
    control.close()
    transport.wake.close()
    transport.failures.close()

    supervisor = DeepStreamChildSupervisor(config(root / "eos", qa_mode=True))
    supervisor.start()
    ready_a = supervisor.sources.add("camera-a", "loopback://camera-a")
    ready_b = supervisor.sources.add("camera-b", "loopback://camera-b")
    binding_a = supervisor.metadata.expected_binding("camera-a")
    binding_b = supervisor.metadata.expected_binding("camera-b")
    if binding_a is None or binding_b is None:
        raise RuntimeError("source binding absent after readiness")
    epoch_two = SourceBinding(
        binding_a.worker_boot_id,
        binding_a.child_instance_id,
        binding_a.camera_id,
        binding_a.source_generation,
        2,
        binding_a.transform_id,
    )
    rebuilt_token = supervisor.metadata.subscribe(epoch_two)
    healthy_token = supervisor.metadata.subscribe(binding_b)
    supervisor.control.inject_source_eos("camera-a")
    rebuilt = supervisor.metadata.wait_accepted(rebuilt_token, timeout_sec=10.0)
    healthy = supervisor.metadata.wait_accepted(healthy_token, timeout_sec=10.0)
    snapshot_a = supervisor.sources.snapshot("camera-a")
    snapshot_b = supervisor.sources.snapshot("camera-b")
    supervisor.stop()
    supervisor.stop()
    if rebuilt.identity.stream_epoch != 2 or snapshot_a.stream_epoch != 2:
        raise RuntimeError("EOS source did not rebuild to epoch 2")
    if healthy.identity.stream_epoch != 1 or snapshot_b.stream_epoch != 1:
        raise RuntimeError("healthy source did not keep flowing at epoch 1")
    if (root / "eos" / "first-fault.json").exists():
        raise RuntimeError("clean EOS rebuild persisted a fatal fault")
    print(
        json.dumps(
            {
                "custom_transform": paused.custom_transform_available,
                "healthy_epoch": snapshot_b.stream_epoch,
                "latest_sequences": [
                    latest_a.native_publish_sequence,
                    latest_b.native_publish_sequence,
                ],
                "metadata_overwritten": paused.metadata_overwritten,
                "python_cuda_initialized": torch.cuda.is_initialized(),
                "raw_readd_generation": readded.source_generation,
                "raw_reconnect_epoch": reconnected.stream_epoch,
                "rebuilt_epoch": snapshot_a.stream_epoch,
                "status": "ok",
                "two_source_ready": [ready_a.state, ready_b.state],
            },
            sort_keys=True,
        )
    )
    shutil.rmtree(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
