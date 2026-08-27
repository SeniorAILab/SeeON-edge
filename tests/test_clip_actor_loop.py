

def test_upkeep_runs_under_continuous_traffic_not_only_when_the_queue_drains() -> None:
    """The load-bearing case: 13 cameras never let the queue go quiet.

    Disk-pressure suspension is cleared only by `rotate`, and an active clip on
    a stalled camera is closed only by `expire`. Both used to run exclusively in
    the `queue.Empty` branch, which needs 100ms of total silence across a queue
    shared by every camera. Thirteen cameras at 15fps put roughly 195 messages/s
    into it, so that branch effectively never fires in production and one
    healthy camera is enough to starve it.

    An idle-queue test passes happily against that bug, which is exactly what
    happened here: the first version of this fix was measured "working" on the
    live stack when in truth the restart had reset the flag.

    So this test never lets the queue drain.
    """
    import queue as queue_mod
    import threading

    from worker.pipeline.output.evidence.clip_actor_loop import run_actor_loop
    from worker.pipeline.output.evidence.clip_recorder_models import FrameMessage

    calls: list[bool] = []
    stop = threading.Event()

    class _ReleasablePacket:
        def release(self) -> None: ...

    class _NeverEmpty(queue_mod.Queue):
        """Always yields a frame; `queue.Empty` is never raised."""

        def __init__(self) -> None:
            super().__init__()
            self.served = 0

        def get(self, block: bool = True, timeout: float | None = None) -> object:
            del block, timeout
            self.served += 1
            if self.served > 400:
                stop.set()
                raise queue_mod.Empty
            return FrameMessage(packet=_ReleasablePacket())

        def task_done(self) -> None: ...

        def empty(self) -> bool:
            return self.served > 400

    class _Actor:
        def __init__(self) -> None:
            self.expiries = 0

        def handle_frame(self, message: object) -> None:
            del message

        def expire(self) -> None:
            self.expiries += 1

        def shutdown(self) -> None: ...

    def rotate(*, force: bool) -> None:
        calls.append(force)

    actor = _Actor()
    messages = _NeverEmpty()
    run_actor_loop(actor, messages, stop, rotate)

    assert messages.served > 400, "the queue must never have drained"
    assert calls, (
        "retention was never re-evaluated under continuous traffic, so disk "
        "pressure stays latched and no clip can ever be recorded again -- the "
        "queue.Empty branch cannot carry this, 13 cameras never let it fire"
    )
    assert not any(calls), (
        "upkeep must not force a rotate; rotate()'s own interval guard is what "
        "keeps this cheap"
    )
    assert actor.expiries, (
        "a stalled camera's active clip could never wall-clock expire while "
        "another camera kept the shared queue busy"
    )
