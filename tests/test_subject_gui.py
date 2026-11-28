import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from idphoto.capture import DemoSource
from idphoto.gui import MainWindow
from idphoto.presentation_modnet import ModnetPrediction
from tests.test_matting_checker import clothing_scene


def test_ambiguous_live_frame_never_shows_raw_and_disables_capture(monkeypatch):
    from idphoto import gui, presentation_modnet

    app = QApplication.instance() or QApplication([])
    rgb, landmarks, _, _ = clothing_scene("jacket")
    started = threading.Event()
    release = threading.Event()

    class Source(DemoSource):
        def __init__(self):
            self.name = "two people"
            self.lock_state = "locked"

        def read(self):
            time.sleep(.025)
            return rgb, landmarks

    class Model:
        def __init__(self, *_args, **_kwargs):
            pass

        def prepare_input(self, *_args, **_kwargs):
            return True

        def predict_alpha(self, raw_rgb, *, capture=False):
            started.set()
            if not release.wait(5):
                raise TimeoutError("prediction was not released")
            alpha = np.full(raw_rgb.shape[:2], .9, np.float32)
            alpha[0] = 0
            return ModnetPrediction(alpha, {"inference_ms": 1})

        def apply_prediction(self, *_args):
            raise AssertionError("an ambiguous image must not be composited")

    original_pipeline = gui.Pipeline

    def create_pipeline(*args, **kwargs):
        pipeline = original_pipeline(*args, **kwargs)
        original_process = pipeline.process

        def process(*args, **kwargs):
            ctx, fired = original_process(*args, **kwargs)
            if ctx.subject_faces:
                second = ctx.subject_faces[0].copy()
                second[:, 0] += 20
                ctx.subject_faces = (ctx.subject_faces[0], second)
                ctx.subject_index = 0
            return ctx, fired

        pipeline.process = process
        return pipeline

    monkeypatch.setattr(gui, "Pipeline", create_pipeline)
    monkeypatch.setattr(gui, "open_source", lambda *_: Source())
    monkeypatch.setattr(presentation_modnet, "check_availability", lambda *_: "local")
    monkeypatch.setattr(presentation_modnet, "PresentationMODNet", Model)
    window = MainWindow("demo", "en", (640, 480), presentation_modnet=True)
    window.open_timer.stop()
    window.show()
    window.open_source()
    errors = []
    window.worker.model_failed.connect(errors.append)

    def wait_for(predicate):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert predicate()

    try:
        window.modnet_box.click()
        wait_for(started.is_set)
        wait_for(lambda: window.live_info is not None)
        window.set_before(True)
        app.processEvents()
        assert window.view.image.pixelColor(10, 10).name() == "#ffffff"
        assert not window.shoot.isEnabled()
        window.take_photo()
        assert window.worker.shutter_request is None
        release.set()
        wait_for(lambda: window.modnet_last is not None
                 and window.modnet_last.get("subject_status") == "ambiguous")
        assert window.view.image.pixelColor(10, 10).name() == "#ffffff"
        assert "Unable to separate" in window.view.message
        assert not window.shoot.isEnabled()
        assert not errors
    finally:
        release.set()
        window.close()
        app.processEvents()


def test_preview_model_error_stays_guarded_and_is_not_silently_disabled(monkeypatch):
    from idphoto import presentation_modnet

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(presentation_modnet, "check_availability", lambda *_: "local")
    monkeypatch.setattr(presentation_modnet, "PresentationMODNet", lambda *_a, **_kw: None)
    window = MainWindow("demo", "en", (640, 480), presentation_modnet=True)
    window.open_timer.stop()
    try:
        window.modnet_box.click()
        window.live_info = {"size": (160, 100), "geometry": None, "face": False}
        window.on_modnet_failed((False, window.modnet_generation, "test failure"))
        assert window.settings.modnet_demo and window.modnet_box.isChecked()
        assert not window.shoot.isEnabled()
        assert "test failure" in window.modnet_status.text()
        assert window.view.image.pixelColor(10, 10).name() == "#ffffff"
    finally:
        window.close()
        app.processEvents()
