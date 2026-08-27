

def test_idle_ticks_re_evaluate_retention_so_pressure_cannot_latch() -> None:
    """Disk-pressure suspension used to be permanent until a worker restart.

    `rotate` is what clears `recording_suspended`. While suspended no clip is
    admitted, so none finalizes, so no FlushMessage is queued -- and a flush was
    the ONLY thing that called rotate. Crossing the high watermark once latched
    recording off for good; freeing the disk did not bring it back.

    Observed live: the clip store sat at 80.68% against a 0.80 watermark and
    every fall for ninety minutes recorded no footage, while detection,
    snapshots and event delivery all looked healthy. Reclaiming 264 GB changed
    nothing until the worker was restarted.

    So the idle tick must re-evaluate, and it must do so unforced -- it runs
    every 100 ms and the interval guard inside rotate() is what bounds the real
    work.
    """
    import queue as queue_mod
    import threading

    from worker.pipeline.output.evidence.clip_actor_loop import run_actor_loop

    calls: list[bool] = []
    stop = threading.Event()
    messages: queue_mod.Queue = queue_mod.Queue()

    class _IdleActor:
        def __init__(self) -> None:
            self.expiries = 0

        def expire(self) -> None:
            self.expiries += 1
            if self.expiries >= 3:
                stop.set()

        def shutdown(self) -> None: ...

    def rotate(*, force: bool) -> None:
        calls.append(force)

    run_actor_loop(_IdleActor(), messages, stop, rotate)

    assert calls, (
        "an idle loop never re-evaluated retention, so disk pressure stayed "
        "latched and no clip could ever be recorded again"
    )
    assert not any(calls), (
        "the idle tick runs every 100ms and must not force a rotate; the "
        "interval guard inside rotate() is what keeps this cheap"
    )
