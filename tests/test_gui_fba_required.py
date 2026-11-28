"""The submission app always mats with FBA + Clothing and refuses to start without FBA."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox

from idphoto import gui
from idphoto.gui import MainWindow


class FakeFBA:
    """Stands in for the FBA model: leaves the Clothing result as it is."""

    calls = 0

    def __call__(self, ctx, background):
        FakeFBA.calls += 1


def _app():
    return QApplication.instance() or QApplication([])


def _wait(app, condition, seconds=15):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(.02)
        app.processEvents()
    return condition()


def test_main_refuses_to_start_without_fba(monkeypatch, capsys):
    monkeypatch.setattr(gui, "load_fba", lambda device="cpu": "FBA assets missing")
    assert gui.main(["--source", "demo", "--lang", "en"]) == 1
    err = capsys.readouterr().err
    assert "FBA assets missing" in err
    assert "tools/fba_reference.py --accept-noncommercial" in err


def test_load_fba_returns_the_reason_when_assets_are_missing(monkeypatch):
    from tools import presentation_fba

    def missing(*_args):
        raise FileNotFoundError("missing research weights")

    monkeypatch.setattr(presentation_fba, "check_availability", missing)
    assert gui.load_fba() == "missing research weights"
    assert gui.load_fba("mps") == "missing research weights"


def test_window_needs_fba_and_has_no_matting_toggles():
    app = _app()
    with pytest.raises(ValueError):
        MainWindow("demo", "en", (640, 480), None)
    window = MainWindow("demo", "ko", (640, 480), FakeFBA)
    window.open_timer.stop()
    try:
        assert window.settings.fba_demo and window.settings.clothing_enabled
        assert not window.settings.modnet_demo
        for name in ("clothing_box", "fba_box", "modnet_box", "birefnet_box", "fba_image"):
            assert not hasattr(window, name)
        boxes = window.findChildren(QCheckBox)
        assert set(boxes) == {window.auto_box, window.mirror_box}
        window.toggle_language()
    finally:
        window.close()
        app.processEvents()


def test_demo_shows_fba_snapshot_and_captures_with_fba(monkeypatch):
    app = _app()
    FakeFBA.calls = 0
    window = MainWindow("demo", "en", (640, 480), FakeFBA)
    window.open_timer.stop()
    window.auto_box.setChecked(False)
    captured = []
    monkeypatch.setattr(window, "on_captured", captured.append)
    window.open_source()
    try:
        assert _wait(app, lambda: window.fba_last is not None)
        assert window.shoot.isEnabled()
        window.take_photo()
        assert _wait(app, lambda: bool(captured))
        assert captured[0]["report"]["settings"]["fba_demo"]
        assert FakeFBA.calls >= 2  # preview and capture both ran FBA
    finally:
        window.close()
        app.processEvents()


def test_fba_preview_failure_stops_photo_taking():
    app = _app()
    window = MainWindow("demo", "en", (640, 480), FakeFBA)
    window.open_timer.stop()
    try:
        window.on_fba_failed((False, window.fba_generation + 1, "old error"))
        assert window.fba_runtime_error is None
        window.on_fba_failed((False, window.fba_generation, "boom"))
        assert window.fba_preview_failed
        assert "boom" in window.fba_runtime_error
        assert not window.shoot.isEnabled()
        assert not window.auto_box.isChecked() and not window.settings.auto_shutter
        assert window.settings.fba_demo  # no silent fallback to Clothing alone
    finally:
        window.close()
        app.processEvents()
