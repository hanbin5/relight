import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from idphoto import matting
from idphoto.capture import DemoSource
from idphoto.gui import MainWindow
from tests.test_matting_checker import clothing_scene


def test_clothing_switch_updates_settings_and_language():
    app = QApplication.instance() or QApplication([])
    window = MainWindow("demo", "ko", (640, 480))
    window.open_timer.stop()
    assert window.clothing_box.isChecked()
    assert window.settings.clothing_enabled
    window.clothing_box.click()
    assert not window.settings.clothing_enabled
    assert "Main" in window.clothing_box.text()
    window.toggle_language()
    assert not window.clothing_box.isChecked()
    assert "Main" in window.clothing_box.text()
    window.clothing_box.click()
    assert window.settings.clothing_enabled
    window.close()
    app.processEvents()


def test_switch_changes_actual_worker_matting(monkeypatch):
    app = QApplication.instance() or QApplication([])
    rgb, lm, _, target = clothing_scene("jacket")

    class FixedDemo(DemoSource):
        def __init__(self):
            self.name = "fixed demo"
            self.lock_state = "locked"

        def read(self):
            time.sleep(.03)
            return rgb, lm

    import idphoto.gui as gui

    monkeypatch.setattr(gui, "open_source", lambda *_: FixedDemo())
    observed = {}
    original = matting.Matter.process

    def record(self, ctx, *args, **kwargs):
        result = original(self, ctx, *args, **kwargs)
        if ctx.landmarks is not None:
            observed[kwargs["clothing_enabled"]] = result.alpha.copy()
        return result

    monkeypatch.setattr(matting.Matter, "process", record)
    window = MainWindow("demo", "ko", (640, 480))
    window.open_timer.stop()
    window.settings.strength = 0
    window.settings.auto_shutter = False
    window.open_source()

    def wait_for(predicate):
        deadline = time.monotonic() + 15
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.02)
            app.processEvents()
        assert predicate(), (list(observed), window.view.message, window.worker.isRunning())

    try:
        wait_for(lambda: True in observed and window.view.image is not None)
        window.clothing_box.click()
        wait_for(lambda: False in observed)
        assert np.any(observed[True][target] > observed[False][target])
    finally:
        window.close()
        app.processEvents()


def test_presentation_fba_is_opt_in_and_disables_when_unavailable(monkeypatch):
    from tools import presentation_fba

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(presentation_fba, "check_availability",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("missing research weights")))
    window = MainWindow("demo", "ko", (640, 480), presentation_fba=True)
    window.open_timer.stop()
    assert not window.fba_box.isEnabled()
    assert not window.settings.fba_demo
    assert "missing research weights" in window.fba_warning.text()
    window.close()
    app.processEvents()


def test_presentation_fba_requires_clothing_and_starts_off(monkeypatch):
    from tools import presentation_fba

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(presentation_fba, "check_availability", lambda: None)
    window = MainWindow("demo", "ko", (640, 480), presentation_fba=True)
    window.open_timer.stop()
    assert window.fba_box.isEnabled() and not window.fba_box.isChecked()
    window.fba_box.click()
    assert window.settings.fba_demo and window.settings.clothing_enabled
    window.clothing_box.click()
    assert not window.settings.clothing_enabled
    assert not window.settings.fba_demo and not window.fba_box.isEnabled()
    window.close()
    app.processEvents()


def test_real_fba_demo_runs_in_gui_worker_when_assets_exist(monkeypatch):
    pytest.importorskip("torch")
    from tools.presentation_fba import PresentationFBA, check_availability

    try:
        check_availability()
    except FileNotFoundError:
        pytest.skip("local FBA research weights unavailable")
    app = QApplication.instance() or QApplication([])
    rgb, lm, _, _ = clothing_scene("jacket")

    class FixedDemo(DemoSource):
        def __init__(self):
            self.name = "fixed demo"
            self.lock_state = "locked"

        def read(self):
            time.sleep(.03)
            return rgb, lm

    import idphoto.gui as gui

    monkeypatch.setattr(gui, "open_source", lambda *_: FixedDemo())
    called = threading.Event()
    original = PresentationFBA.__call__

    def observe(self, ctx, background):
        original(self, ctx, background)
        assert np.isfinite(ctx.alpha).all()
        called.set()

    monkeypatch.setattr(PresentationFBA, "__call__", observe)
    window = MainWindow("demo", "ko", (640, 480), presentation_fba=True)
    window.open_timer.stop()
    window.settings.strength = 0
    window.settings.auto_shutter = False
    window.open_source()
    try:
        assert window.fba_box.isEnabled()
        window.fba_box.click()
        deadline = time.monotonic() + 30
        while not called.is_set() and window.worker.isRunning() and time.monotonic() < deadline:
            time.sleep(.02)
            app.processEvents()
        assert called.is_set(), window.view.message
    finally:
        window.close()
        app.processEvents()


def test_fba_wait_does_not_stop_clothing_preview_or_capture(monkeypatch):
    from tools import presentation_fba

    app = QApplication.instance() or QApplication([])
    rgb, lm, _, _ = clothing_scene("jacket")
    started, release = threading.Event(), threading.Event()
    frames, captured = [], []

    class FixedDemo(DemoSource):
        def __init__(self):
            self.name = "fixed demo"
            self.lock_state = "locked"

        def read(self):
            time.sleep(.03)
            return rgb, lm

    class SlowFBA:
        def __call__(self, ctx, background):
            started.set()
            assert release.wait(10)

    import idphoto.gui as gui

    monkeypatch.setattr(gui, "open_source", lambda *_: FixedDemo())
    monkeypatch.setattr(presentation_fba, "check_availability", lambda: None)
    monkeypatch.setattr(presentation_fba, "PresentationFBA", SlowFBA)
    window = MainWindow("demo", "ko", (640, 480), presentation_fba=True)
    window.open_timer.stop()
    window.settings.strength = 0
    window.settings.auto_shutter = False
    monkeypatch.setattr(window, "on_captured", captured.append)
    window.open_source()
    window.worker.frame_ready.connect(lambda info: frames.append(info))

    def wait_for(predicate):
        deadline = time.monotonic() + 15
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.01)
            app.processEvents()
        assert predicate()

    try:
        window.fba_box.click()
        wait_for(started.is_set)
        count = len(frames)
        window.take_photo()
        wait_for(lambda: len(frames) >= count + 3 and any(f["fba_capture_busy"] for f in frames))
        assert not captured
        release.set()
        wait_for(lambda: bool(captured) and window.fba_last is not None)
        assert captured[0]["report"]["settings"]["fba_demo"]
        assert "스냅샷" in window.fba_status.text()
        old_snapshot = window.fba_last
        window.fba_box.click()
        window.fba_box.click()
        window.on_fba_snapshot(old_snapshot)
        assert window.fba_last is None
    finally:
        release.set()
        window.close()
        app.processEvents()


def test_stale_preview_errors_are_ignored_and_current_error_disables_fba(monkeypatch):
    from tools import presentation_fba

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(presentation_fba, "check_availability", lambda: None)
    window = MainWindow("demo", "ko", (640, 480), presentation_fba=True)
    window.open_timer.stop()
    window.fba_box.click()
    stale = window.fba_generation
    window.fba_box.click()
    window.fba_box.click()
    window.on_fba_failed((False, stale, "old failure"))
    assert window.fba_box.isChecked()
    window.on_fba_failed((False, window.fba_generation, "current failure"))
    assert not window.fba_box.isEnabled()
    assert "current failure" in window.fba_status.text()
    window.clothing_box.click()
    window.clothing_box.click()
    assert not window.fba_box.isEnabled()
    window.close()
    app.processEvents()
