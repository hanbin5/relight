"""Single-worker, bounded scheduler for the optional local FBA demo."""

import threading


class FbaJobs:
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
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="fba-demo")
        self._thread.start()

    @property
    def capture_busy(self):
        with self._condition:
            return self._capture_busy

    def submit_preview(self, *args):
        with self._condition:
            if not self._closed:
                self._preview = args
                self._condition.notify()

    def clear_preview(self):
        with self._condition:
            self._preview = None

    def submit_capture(self, *args):
        with self._condition:
            if self._closed or self._capture_busy:
                return False
            self._capture_busy = True
            self._capture = args
            self._condition.notify()
            return True

    def stop(self):
        with self._condition:
            self._closed = True
            self._preview = None
            self._capture = None
            self._condition.notify()
        self._thread.join()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._capture is not None
                                         or self._preview is not None)
                if self._closed:
                    return
                is_capture = self._capture is not None
                if is_capture:
                    args, self._capture = self._capture, None
                else:
                    args, self._preview = self._preview, None
            try:
                result = self.capture(*args) if is_capture else self.preview(*args)
            except Exception as exc:
                self.on_error(is_capture, args, str(exc))
            else:
                if is_capture:
                    self.on_capture(result)
                else:
                    self.on_preview(result)
            finally:
                if is_capture:
                    with self._condition:
                        self._capture_busy = False
