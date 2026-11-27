import threading
import time

from idphoto.fba_jobs import FbaJobs, ModelJobs
from idphoto.gui import _run_serialized_fba


def test_preview_is_latest_only_and_capture_runs_independently():
    started = threading.Event()
    release = threading.Event()
    capture_finished = threading.Event()
    preview_finished = threading.Event()
    observed = []
    errors = []

    def preview(value):
        observed.append(("start", value))
        if value == 1:
            started.set()
            assert release.wait(5)
        if value == 3:
            preview_finished.set()
        return value

    def capture(value):
        observed.append(("capture", value))
        capture_finished.set()
        return value

    jobs = FbaJobs(preview, capture, lambda x: observed.append(("preview", x)),
                   lambda x: observed.append(("capture-result", x)),
                   lambda *x: errors.append(x))
    try:
        jobs.submit_preview(1)
        assert started.wait(5)
        jobs.submit_preview(2)
        jobs.submit_preview(3)
        assert jobs.submit_capture("still")
        assert not jobs.submit_capture("duplicate")
        assert capture_finished.wait(5)
        assert not release.is_set()
        release.set()
        assert preview_finished.wait(5)
    finally:
        release.set()
        jobs.stop()
    assert not errors
    assert observed.count(("start", 1)) == 1
    assert observed.count(("start", 3)) == 1
    assert ("start", 2) not in observed
    assert ("capture", "still") in observed
    assert ("preview", 1) in observed and ("preview", 3) in observed


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


def test_capture_job_records_submit_start_and_finish_timestamps():
    finished = threading.Event()
    results = []
    jobs = ModelJobs(
        lambda: None,
        lambda *_: {"capture": True},
        lambda _result: None,
        lambda result: (results.append(result), finished.set()),
        lambda *_: None,
    )
    request = {
        "frame_id": 7,
        "timestamp_ms": 80.0,
        "source_read_timestamp_ms": 80.0,
        "manual_request_timestamp_ms": 79.0,
    }
    try:
        assert jobs.submit_capture(None, None, None, request)
        assert finished.wait(5)
    finally:
        jobs.stop()

    timing = results[0]["model_job_timing_ms"]
    assert timing["submitted"] <= timing["started"] <= timing["finished"]
    assert results[0]["capture_worker_input"] == {
        **request,
        "started_ms": timing["started"],
    }


def test_shutdown_suppresses_in_flight_callbacks_and_releases_capture_slot():
    preview_started = threading.Event()
    release_preview = threading.Event()
    capture_started = threading.Event()
    release_capture = threading.Event()
    results = []
    jobs = FbaJobs(
        lambda: (preview_started.set(), release_preview.wait(5)),
        lambda value: (capture_started.set(), release_capture.wait(5), value)[-1],
        results.append,
        results.append,
        lambda *_: None,
    )
    try:
        jobs.submit_preview()
        assert preview_started.wait(5)
        assert jobs.submit_capture("pending")
        assert capture_started.wait(5)
        stopped = threading.Event()
        thread = threading.Thread(target=lambda: (jobs.stop(), stopped.set()))
        thread.start()
        release_preview.set()
        release_capture.set()
        assert stopped.wait(5)
        thread.join()
        assert results == []
        assert not jobs.capture_busy
    finally:
        release_preview.set()
        release_capture.set()
        jobs.stop()


def test_fba_preview_and_capture_share_one_serialized_model():
    lock = threading.Lock()
    state_lock = threading.Lock()
    completed = []
    preview_done = threading.Event()
    capture_done = threading.Event()

    class FakePipeline:
        def __init__(self):
            self.model = None
            self.model_count = 0
            self.active = 0
            self.max_active = 0

        def run(self, label):
            with state_lock:
                if self.model is None:
                    self.model = object()
                    self.model_count += 1
                model = self.model
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(.02)
            with state_lock:
                self.active -= 1
            return label, id(model)

    pipeline = FakePipeline()
    jobs = ModelJobs(
        lambda: _run_serialized_fba(lock, pipeline.run, "preview"),
        lambda: _run_serialized_fba(lock, pipeline.run, "capture"),
        lambda result: (completed.append(result), preview_done.set()),
        lambda result: (completed.append(result), capture_done.set()),
        lambda *_: None,
    )
    try:
        jobs.submit_preview()
        assert jobs.submit_capture()
        assert preview_done.wait(5)
        assert capture_done.wait(5)
    finally:
        jobs.stop()

    assert pipeline.model_count == 1
    assert pipeline.max_active == 1
    assert len({model_id for _, model_id in completed}) == 1
