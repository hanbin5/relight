"""Bounded independent workers for optional model previews and captures."""

import threading
import time


class ModelJobs:
    def __init__(self, preview, capture, on_preview, on_capture, on_error):
        self.preview = preview
        self.capture = capture
        self.on_preview = on_preview
        self.on_capture = on_capture
        self.on_error = on_error
        self._condition = threading.Condition()
        self._preview = None
        self._capture = None
        self._capture_busy = False
        self._capture_running = False
        self._closed = False
        self._preview_thread = threading.Thread(
            target=self._run, args=(False,), name="matting-preview"
        )
        self._capture_thread = threading.Thread(
            target=self._run, args=(True,), name="matting-capture"
        )
        self._preview_thread.start()
        self._capture_thread.start()

    @property
    def capture_busy(self):
        with self._condition:
            return self._capture_busy

    def submit_preview(self, *args):
        with self._condition:
            if not self._closed:
                self._preview = (args, time.monotonic() * 1000)
                self._condition.notify_all()

    def clear_preview(self):
        with self._condition:
            self._preview = None

    def submit_capture(self, *args):
        with self._condition:
            if self._closed or self._capture_busy:
                return False
            self._capture_busy = True
            self._capture = (args, time.monotonic() * 1000)
            self._condition.notify_all()
            return True

    def stop(self):
        with self._condition:
            self._closed = True
            self._preview = None
            self._capture = None
            if not self._capture_running:
                self._capture_busy = False
            self._condition.notify_all()
        self._preview_thread.join()
        self._capture_thread.join()

    def _run(self, is_capture):
        while True:
            with self._condition:
                pending = lambda: self._capture if is_capture else self._preview
                self._condition.wait_for(lambda: self._closed or pending() is not None)
                if self._closed:
                    return
                if is_capture:
                    (args, submitted_ms), self._capture = self._capture, None
                    self._capture_running = True
                else:
                    (args, submitted_ms), self._preview = self._preview, None
            started_ms = time.monotonic() * 1000
            try:
                result = self.capture(*args) if is_capture else self.preview(*args)
            except Exception as exc:
                with self._condition:
                    closed = self._closed
                if not closed:
                    self.on_error(is_capture, args, str(exc))
            else:
                finished_ms = time.monotonic() * 1000
                with self._condition:
                    closed = self._closed
                if not closed:
                    if isinstance(result, dict):
                        result["model_job_timing_ms"] = {
                            "submitted": submitted_ms,
                            "started": started_ms,
                            "finished": finished_ms,
                        }
                    if is_capture and args and isinstance(args[-1], dict):
                        request = args[-1]
                        result["capture_worker_input"] = {
                            "frame_id": request.get("frame_id"),
                            "timestamp_ms": request.get("timestamp_ms"),
                            "source_read_timestamp_ms":
                                request.get("source_read_timestamp_ms"),
                            "manual_request_timestamp_ms":
                                request.get("manual_request_timestamp_ms"),
                            "started_ms": started_ms,
                        }
                    if is_capture:
                        self.on_capture(result)
                    else:
                        self.on_preview(result)
            finally:
                if is_capture:
                    with self._condition:
                        self._capture_running = False
                        self._capture_busy = False


FbaJobs = ModelJobs
