import threading

from idphoto.fba_jobs import FbaJobs


def test_latest_preview_is_bounded_and_capture_takes_priority():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    observed = []
    errors = []

    def preview(value):
        observed.append(("start", value))
        if value == 1:
            started.set()
            assert release.wait(5)
        if value == 3:
            finished.set()
        return value

    jobs = FbaJobs(preview, lambda x: x, lambda x: observed.append(("preview", x)),
                   lambda x: observed.append(("capture", x)), lambda *x: errors.append(x))
    try:
        jobs.submit_preview(1)
        assert started.wait(5)
        jobs.submit_preview(2)
        jobs.submit_preview(3)
        assert jobs.submit_capture("still")
        assert not jobs.submit_capture("duplicate")
        release.set()
        assert finished.wait(5)
    finally:
        release.set()
        jobs.stop()
    assert not errors
    assert observed == [("start", 1), ("preview", 1), ("capture", "still"),
                        ("start", 3), ("preview", 3)]


def test_fba_failure_is_reported_and_capture_slot_released():
    failed = threading.Event()
    errors = []

    def capture(*_):
        raise ValueError("bad capture")

    jobs = FbaJobs(lambda: None, capture, lambda _: None, lambda _: None,
                   lambda *args: (errors.append(args), failed.set()))
    try:
        assert jobs.submit_capture("frame")
        assert failed.wait(5)
    finally:
        jobs.stop()
    assert errors == [(True, ("frame",), "bad capture")]
