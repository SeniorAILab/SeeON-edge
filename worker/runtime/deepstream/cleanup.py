"""Idempotent teardown of Python-owned dark child resources."""

from __future__ import annotations

import signal
import subprocess
from dataclasses import dataclass

from worker.native.deepstream.control import ChildControlError, DeepStreamControlClient
from worker.native.deepstream.metadata import MetadataReceiver
from worker.runtime.deepstream.child_monitor import ChildExitMonitor
from worker.runtime.deepstream.failure_receiver import NativeFailureReceiver
from worker.runtime.lease import GpuLease


@dataclass(frozen=True, slots=True)
class ChildResources:
    process: subprocess.Popen[bytes] | None
    control: DeepStreamControlClient | None
    metadata: MetadataReceiver | None
    failures: NativeFailureReceiver | None
    monitor: ChildExitMonitor | None
    lease: GpuLease | None
    stop_timeout_sec: float


def stop_child_resources(resources: ChildResources) -> None:
    if resources.failures is not None:
        resources.failures.stop()
    if resources.metadata is not None:
        resources.metadata.__exit__(None, None, None)
    process = resources.process
    control = resources.control
    if process is not None and process.poll() is None:
        if control is not None:
            try:
                control.shutdown()
            except ChildControlError:
                process.send_signal(signal.SIGTERM)
        else:
            process.send_signal(signal.SIGTERM)
        monitor = resources.monitor
        if monitor is None or not monitor.exited.wait(resources.stop_timeout_sec):
            process.kill()
            if monitor is not None:
                _ = monitor.exited.wait(resources.stop_timeout_sec)
    if control is not None:
        control.close()
    if resources.monitor is not None:
        resources.monitor.join(resources.stop_timeout_sec)
    if resources.lease is not None:
        resources.lease.close()


__all__ = ["ChildResources", "stop_child_resources"]
