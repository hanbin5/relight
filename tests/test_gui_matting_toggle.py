import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from idphoto import matting
from idphoto.capture import DemoSource
from idphoto.gui import MainWindow, Worker
from tests.test_matting_checker import clothing_scene


def _alpha_with_background(shape):
    alpha = np.full(shape, .9, np.float32)
    alpha[0, :] = 0
    return alpha


def test_clothing_switch_updates_settings_and_language():
    app = QApplication.instance() or QApplication([])
    window = MainWindow("demo", "ko", (640, 480))
    window.open_timer.stop()
    assert window.clothing_box.isChecked()
    assert window.settings.clothing_enabled
    assert not window.settings.modnet_demo
    assert not window.modnet_box.isEnabled()
    assert "--presentation-modnet" in window.modnet_box.toolTip()
    window.clothing_box.click()
    assert not window.settings.clothing_enabled
    assert "Main" in window.clothing_box.text()
    window.toggle_language()
    assert not window.clothing_box.isChecked()
    assert "Main" in window.clothing_box.text()
    assert "Enable with --presentation-modnet" in window.modnet_warning.text()
    window.clothing_box.click()
    assert window.settings.clothing_enabled
    window.close()
    app.processEvents()


def test_missing_opt_in_model_disables_choice_with_reason(monkeypatch):
    from idphoto import presentation_modnet

    def missing_model(*_args):
        raise FileNotFoundError("local model missing")

    monkeypatch.setattr(presentation_modnet, "check_availability", missing_model)
    app = QApplication.instance() or QApplication([])
    window = MainWindow("demo", "en", (640, 480), presentation_modnet=True)
    window.open_timer.stop()
    try:
        assert not window.settings.modnet_demo
        assert not window.modnet_box.isEnabled()
        assert "local model missing" in window.modnet_warning.text()
        window.modnet_box.click()
        assert not window.settings.modnet_demo
    finally:
        window.close()
        app.processEvents()


def test_manual_capture_is_joined_to_the_first_source_read_after_its_click(monkeypatch):
    from idphoto.pipeline import Settings

    import idphoto.gui as gui

    worker = Worker(None, Settings(), settings_lock=threading.RLock())
    clock = [1.0]
    monkeypatch.setattr(gui.time, "monotonic", lambda: clock[0])
    worker.request_manual_capture()
    request_timestamp_ms = 1000.0

    # Frame A was already read before the button event; the request stays pending.
    frame_a = worker._take_frame_request(999.0)
    assert frame_a[3] is False
    assert worker.shutter_request == pytest.approx(request_timestamp_ms)

    # An ambiguous same-millisecond boundary is conservatively assigned to B.
    frame_a_ambiguous = worker._take_frame_request(request_timestamp_ms)
    assert frame_a_ambiguous[3] is False
    assert worker.shutter_request == pytest.approx(request_timestamp_ms)

    frame_b = worker._take_frame_request(1001.0)
    assert frame_b[3] is True
    assert frame_b[4] == request_timestamp_ms
    assert frame_b[0].auto_shutter is True
    assert worker.shutter_request is None
    assert worker._take_frame_request(1003.0)[3] is False

    clock[0] = 0.999
    worker.request_manual_capture()
    before_first_read = worker._take_frame_request(1000.0)
    assert before_first_read[3] is True
    assert before_first_read[0].auto_shutter is True


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


def test_traditional_capture_is_discarded_after_switch_to_modnet(monkeypatch):
    from idphoto import gui, presentation_modnet

    app = QApplication.instance() or QApplication([])
    rgb, lm, _, _ = clothing_scene("jacket")
    started = threading.Event()
    release = threading.Event()
    failures = []
    displayed = []

    class FixedDemo(DemoSource):
        def __init__(self):
            self.name = "fixed demo"
            self.lock_state = "locked"

        def read(self):
            time.sleep(.025)
            return rgb, lm

    original_render = gui.Pipeline.render_final

    def blocked_render(self, *args):
        started.set()
        if not release.wait(5):
            raise TimeoutError("capture was not released")
        return original_render(self, *args)

    monkeypatch.setattr(gui, "open_source", lambda *_: FixedDemo())
    monkeypatch.setattr(gui.Pipeline, "render_final", blocked_render)
    monkeypatch.setattr(presentation_modnet, "check_availability", lambda *_: "local")
    monkeypatch.setattr(presentation_modnet, "PresentationMODNet", lambda *_a, **_kw: None)
    window = MainWindow("demo", "en", (640, 480), presentation_modnet=True)
    window.open_timer.stop()
    window.settings.auto_shutter = False
    monkeypatch.setattr(window, "on_model_failed", failures.append)
    window.capture_displayed.connect(lambda result, guard: displayed.append((result, guard)))
    window.show()
    window.open_source()

    def wait_for(predicate):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert predicate()

    try:
        wait_for(lambda: window.live_info is not None and window.shoot.isEnabled())
        window.take_photo()
        wait_for(started.is_set)
        window.modnet_box.click()
        release.set()
        wait_for(lambda: any("settings changed" in error[-1] for error in failures)
                 or displayed)
        assert not displayed
        assert any("settings changed" in error[-1] for error in failures)
    finally:
        release.set()
        window.close()
        app.processEvents()


def test_modnet_worker_paints_fresh_results_and_reruns_full_capture(monkeypatch):
    from idphoto import presentation_modnet

    app = QApplication.instance() or QApplication([])
    rgb, lm, _, _ = clothing_scene("jacket")
    calls = []

    class FixedDemo(DemoSource):
        def __init__(self):
            self.name = "fixed demo"
            self.lock_state = "locked"

        def read(self):
            time.sleep(.025)
            return rgb, lm

    class FakeMODNet:
        def __init__(self, *_args, **_kwargs):
            pass

        def prepare_input(self, _raw_rgb, *, capture=False):
            return True

        def predict_alpha(self, raw_rgb, *, capture=False):
            calls.append((raw_rgb.shape, capture))
            return presentation_modnet.ModnetPrediction(
                _alpha_with_background(raw_rgb.shape[:2]),
                {"inference_ms": 1.0, "input_width": raw_rgb.shape[1],
                 "input_height": raw_rgb.shape[0], "capture": capture, "provider": "test"},
            )

        def apply_prediction(self, ctx, prediction, background):
            ctx.alpha = prediction.alpha
            ctx.wall_color = presentation_modnet.estimate_wall_color(ctx.linear_rgb, ctx.alpha)
            if background is None:
                ctx.output_frame = ctx.relit_frame
            else:
                wall = presentation_modnet.estimate_local_wall(
                    ctx.linear_rgb, ctx.alpha, ctx.wall_color)
                ctx.output_frame = matting.composite(ctx, wall, background)

    import idphoto.gui as gui

    monkeypatch.setattr(gui, "open_source", lambda *_: FixedDemo())
    monkeypatch.setattr(presentation_modnet, "check_availability", lambda *_: "local-model")
    monkeypatch.setattr(presentation_modnet, "PresentationMODNet", FakeMODNet)
    window = MainWindow("demo", "ko", (640, 480), presentation_modnet=True)
    window.open_timer.stop()
    window.settings.strength = 0
    window.settings.auto_shutter = False
    captured = []
    painted_frames = []
    window.view.modnet_frame_painted.connect(
        lambda frame_id, _painted_ms: painted_frames.append(frame_id))
    monkeypatch.setattr(window, "on_captured", captured.append)
    window.show()
    window.open_source()

    def wait_for(predicate):
        deadline = time.monotonic() + 15
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.02)
            app.processEvents()
        assert predicate(), (window.view.message, window.worker.isRunning(), calls,
                             window.dialog, window.worker.shutter_request)

    try:
        assert window.modnet_box.isEnabled()
        window.modnet_box.click()
        assert "MODNet" in window.modnet_preview_label.text()
        assert "미사용" in window.modnet_preview_label.text()
        window.clothing_box.click()
        assert "MODNet" in window.modnet_preview_label.text()
        window.clothing_box.click()
        wait_for(lambda: window.modnet_last is not None and
                 window.modnet_last["frame_id"] in painted_frames)
        first = painted_frames[0]
        assert window.view.info["modnet_display"]
        assert window.modnet_last["frame_id"] in painted_frames
        assert window.modnet_last["face_detected"]
        assert len(set(painted_frames)) == len(painted_frames)
        assert "MODNet" in window.status.text()
        assert "Worker" not in window.status.text()
        assert "Clothing" not in window.status.text()

        snapshot = window.modnet_last
        pending_info = dict(window.live_info)
        pending_info["verdicts"] = None
        window.modnet_last = None
        window.on_frame(pending_info)
        assert "미판정" in window.coach.text()
        assert "모든 점검 통과" not in window.coach.text()
        assert all(window.rule_marks[n].text() == "…" for n in range(1, 8))
        window.modnet_last = snapshot
        window.on_frame(dict(window.live_info))

        window.take_photo()
        wait_for(lambda: bool(captured))
        assert captured[0]["report"]["settings"]["modnet_demo"]
        assert (rgb.shape, True) in calls
    finally:
        window.close()
        app.processEvents()


def test_slow_capture_keeps_preview_running_and_discards_changed_settings(monkeypatch):
    from idphoto import presentation_modnet

    app = QApplication.instance() or QApplication([])
    rgb, lm, _, _ = clothing_scene("jacket")
    capture_started = threading.Event()
    release_capture = threading.Event()
    captured = []
    painted_frames = []

    class FixedDemo(DemoSource):
        def __init__(self):
            self.name = "slow capture demo"
            self.lock_state = "locked"

        def read(self):
            time.sleep(.02)
            return rgb, lm

    class SlowCaptureMODNet:
        def __init__(self, *_args, **_kwargs):
            pass

        def prepare_input(self, _raw_rgb, *, capture=False):
            return True

        def predict_alpha(self, raw_rgb, *, capture=False):
            if capture:
                capture_started.set()
                if not release_capture.wait(5):
                    raise TimeoutError("synthetic capture was not released")
            return presentation_modnet.ModnetPrediction(
                _alpha_with_background(raw_rgb.shape[:2]),
                {"inference_ms": 1.0, "input_width": raw_rgb.shape[1],
                 "input_height": raw_rgb.shape[0], "capture": capture, "provider": "test"},
            )

        def apply_prediction(self, ctx, prediction, background):
            ctx.alpha = prediction.alpha
            ctx.wall_color = presentation_modnet.estimate_wall_color(ctx.linear_rgb, ctx.alpha)
            if background is None:
                ctx.output_frame = ctx.relit_frame
            else:
                wall = presentation_modnet.estimate_local_wall(
                    ctx.linear_rgb, ctx.alpha, ctx.wall_color)
                ctx.output_frame = matting.composite(ctx, wall, background)

    import idphoto.gui as gui

    monkeypatch.setattr(gui, "open_source", lambda *_: FixedDemo())
    monkeypatch.setattr(presentation_modnet, "check_availability", lambda *_: "local-model")
    monkeypatch.setattr(presentation_modnet, "PresentationMODNet", SlowCaptureMODNet)
    window = MainWindow("demo", "ko", (640, 480), presentation_modnet=True)
    window.open_timer.stop()
    window.settings.strength = 0
    window.settings.auto_shutter = False
    window.view.modnet_frame_painted.connect(
        lambda frame_id, _painted_ms: painted_frames.append(frame_id)
    )
    monkeypatch.setattr(window, "on_captured", captured.append)
    window.show()
    window.open_source()
    model_errors = []
    window.worker.model_failed.connect(model_errors.append)

    def wait_for(predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert predicate(), (painted_frames, window.modnet_runtime_error,
                             window.worker.isRunning())

    try:
        window.modnet_box.click()
        wait_for(lambda: window.modnet_last is not None
                 and window.modnet_last["frame_id"] in painted_frames)
        window.take_photo()
        assert capture_started.wait(5)
        paints_at_capture_start = len(painted_frames)
        window.set_format("profile")
        window.set_format("resume")
        wait_for(lambda: len(painted_frames) >= paints_at_capture_start + 2)
        release_capture.set()
        wait_for(lambda: bool(model_errors))
        assert "discarded" in model_errors[0][-1]
        assert not captured
        assert window.worker is not None
        assert window.worker.isRunning()
    finally:
        release_capture.set()
        window.close()
        app.processEvents()


@pytest.mark.parametrize(("stale_change", "automatic"), [
    ("settings", True), ("auto", True), ("settings", False),
])
def test_async_capture_is_revalidated_before_display(monkeypatch, stale_change, automatic):
    from types import SimpleNamespace

    from idphoto.gui import _copy_settings

    app = QApplication.instance() or QApplication([])
    window = MainWindow("demo", "en", (640, 480))
    window.open_timer.stop()
    window.worker = SimpleNamespace(track_epoch=0 if automatic else 1)
    if not automatic:
        window.settings.modnet_demo = True
        window.settings.capture_matte = "birefnet-portrait"
    failures = []
    displayed = []
    monkeypatch.setattr(window, "on_model_failed", failures.append)
    window.capture_displayed.connect(
        lambda result, guard: displayed.append((result, guard))
    )
    request_settings = _copy_settings(window.settings)
    guard = {
        "generation": window.model_generation,
        "auto_generation": window.auto_shutter_generation,
        "automatic": automatic,
        "track_epoch": 0,
        "settings": request_settings,
        "model": "modnet" if automatic else "birefnet",
    }

    if stale_change == "settings":
        window.model_generation += 1
    else:
        window.settings.auto_shutter = False
        window.auto_shutter_generation += 1

    try:
        result = {"report": {"settings": {"auto_shutter": True}},
                  "_capture_display_guard": guard}
        window.on_captured(result)
        assert window.dialog is None
        assert len(failures) == 1
        assert "before display" in failures[0][-1]
        assert displayed == []
    finally:
        window.worker = None
        window.close()
        app.processEvents()


def test_manual_async_capture_survives_auto_and_later_track_changes(monkeypatch):
    from types import SimpleNamespace

    import idphoto.gui as gui
    from idphoto.gui import _copy_settings

    app = QApplication.instance() or QApplication([])
    window = MainWindow("demo", "en", (640, 480))
    window.open_timer.stop()
    window.worker = SimpleNamespace(track_epoch=1)
    window.settings.modnet_demo = True
    window.settings.capture_matte = "birefnet-portrait"
    window.settings.auto_shutter = False
    window.auto_shutter_generation += 1
    request_settings = _copy_settings(window.settings)

    class Finished:
        def connect(self, _callback):
            pass

    class FakeDialog:
        def __init__(self, _parent, result, _lang):
            self.result = result
            self.finished = Finished()
            self.visible = False

        def open(self):
            self.visible = True

        def isVisible(self):
            return self.visible

    monkeypatch.setattr(gui, "ResultDialog", FakeDialog)
    displayed = []
    window.capture_displayed.connect(
        lambda dialog_result, guard: displayed.append((dialog_result, guard))
    )
    guard = {
        "generation": window.model_generation,
        "auto_generation": window.auto_shutter_generation - 1,
        "automatic": False,
        "track_epoch": 0,
        "settings": request_settings,
        "model": "birefnet",
    }
    result = {
        "report": {"settings": {"auto_shutter": False}},
        "_capture_display_guard": guard,
    }
    try:
        window.on_captured(result)
        assert window.dialog is not None
        assert "_capture_display_guard" not in window.dialog.result
        assert window.dialog.result["report"]["settings"]["auto_shutter"] is False
        assert displayed == [(window.dialog.result, guard)]
        assert guard["result_dialog_is_expected_type"] is True
        assert guard["result_dialog_visible"] is True
    finally:
        window.worker = None
        window.close()
        app.processEvents()


@pytest.mark.parametrize("outcome", [
    "display", "manual-and-auto", "settings-change", "ambiguous",
])
def test_frozen_manual_birefnet_frame_a_survives_live_track_change_to_b(
    monkeypatch, tmp_path, outcome
):
    from types import SimpleNamespace

    from idphoto import presentation_birefnet, presentation_modnet
    from idphoto.context import FrameContext

    app = QApplication.instance() or QApplication([])
    rgb_a, lm, _, _ = clothing_scene("jacket")
    rgb_a = rgb_a.copy()
    rgb_a[:12, :12] = [20, 40, 60]
    rgb_a[:12, -12:] = [20, 40, 60]
    rgb_b = rgb_a.copy()
    rgb_b[:12, :12] = [200, 30, 40]
    rgb_b[:12, -12:] = [200, 30, 40]
    source_phase = {"value": "A"}
    b_track_break_seen = {"value": False}
    capture_started = threading.Event()
    release_capture = threading.Event()
    auto_on_manual_frame = threading.Event()
    frozen = {}
    manual = {}

    class FixedDemo(DemoSource):
        def __init__(self):
            self.name = "A-to-B demo"
            self.lock_state = "locked"

        def read(self):
            time.sleep(.025)
            return (rgb_a if source_phase["value"] == "A" else rgb_b), lm

    class FakeMODNet:
        def __init__(self, *_args, **_kwargs):
            pass

        def prepare_input(self, _raw_rgb, *, capture=False):
            return True

        def predict_alpha(self, raw_rgb, *, capture=False):
            return presentation_modnet.ModnetPrediction(
                _alpha_with_background(raw_rgb.shape[:2]),
                {"inference_ms": 1.0, "input_width": raw_rgb.shape[1],
                 "input_height": raw_rgb.shape[0], "capture": capture, "provider": "test"},
            )

        def apply_prediction(self, ctx, prediction, background):
            ctx.alpha = prediction.alpha
            ctx.wall_color = presentation_modnet.estimate_wall_color(
                ctx.linear_rgb, ctx.alpha)
            if background is None:
                ctx.output_frame = ctx.relit_frame
            else:
                wall = presentation_modnet.estimate_local_wall(
                    ctx.linear_rgb, ctx.alpha, ctx.wall_color)
                ctx.output_frame = matting.composite(ctx, wall, background)

    import idphoto.gui as gui

    source = FixedDemo()
    monkeypatch.setattr(gui, "open_source", lambda *_: source)
    monkeypatch.setattr(presentation_modnet, "check_availability", lambda *_: "local-model")
    monkeypatch.setattr(presentation_modnet, "PresentationMODNet", FakeMODNet)
    monkeypatch.setattr(
        presentation_birefnet, "check_availability",
        lambda path, **_kwargs: path,
    )

    original_pipeline = gui.Pipeline
    original_process = original_pipeline.process
    original_finalize = original_pipeline.finalize_deferred_model
    original_manual = gui.Worker.request_manual_capture
    original_take_frame_request = gui.Worker._take_frame_request

    def request_manual(worker):
        original_manual(worker)
        manual["timestamp_ms"] = worker.shutter_request

    monkeypatch.setattr(gui.Worker, "request_manual_capture", request_manual)

    def take_frame_request(worker, timestamp_ms):
        request = original_take_frame_request(worker, timestamp_ms)
        if request[3]:
            manual["frame_timestamp_ms"] = timestamp_ms
        return request

    monkeypatch.setattr(gui.Worker, "_take_frame_request", take_frame_request)

    def create_pipeline(*args, **kwargs):
        pipeline = original_pipeline(*args, **kwargs)

        def process(raw, *process_args, **process_kwargs):
            ctx, fired = original_process(pipeline, raw, *process_args, **process_kwargs)
            is_b_marker = any(
                raw[0, x, 0] > 150 and raw[0, x, 1] < 60 and raw[0, x, 2] < 70
                for x in (0, -1)
            )
            if is_b_marker and not b_track_break_seen["value"]:
                b_track_break_seen["value"] = True
                ctx.stats["landmark_source"] = "native"
                ctx.stats["track_break"] = True
            return ctx, fired

        pipeline.process = process
        if outcome == "manual-and-auto":
            def finalize(ctx, settings):
                result, fired = original_finalize(pipeline, ctx, settings)
                if ctx.timestamp_ms == manual.get("frame_timestamp_ms"):
                    fired = True
                    auto_on_manual_frame.set()
                return result, fired
            pipeline.finalize_deferred_model = finalize
        return pipeline

    monkeypatch.setattr(gui, "Pipeline", create_pipeline)

    def delayed_render_final(_self, frame, live, _settings):
        frozen["frame"] = frame.copy()
        frozen["timestamp_ms"] = live.timestamp_ms
        frozen["subject_faces"] = live.subject_faces
        frozen["subject_index"] = live.subject_index
        source_phase["value"] = "B"
        capture_started.set()
        if not release_capture.wait(5):
            raise TimeoutError("test capture was not released")
        if outcome == "ambiguous":
            return {"subject_blocked": "ambiguous"}
        ctx = FrameContext(raw_frame=frame.copy(), timestamp_ms=live.timestamp_ms)
        ctx.stats = {}
        return {"ctx": ctx, "report": {"settings": {"auto_shutter": False}}}

    monkeypatch.setattr(original_pipeline, "render_final", delayed_render_final)

    class FakeDialog:
        def __init__(self, _parent, result, _lang):
            self.result = result
            self.visible = False
            self.finished = SimpleNamespace(connect=lambda _callback: None)

        def open(self):
            self.visible = True

        def isVisible(self):
            return self.visible

        def close(self):
            self.visible = False

    monkeypatch.setattr(gui, "ResultDialog", FakeDialog)
    model_path = tmp_path / "portrait.onnx"
    model_path.write_bytes(b"synthetic path only")
    window = MainWindow(
        "demo", "en", (640, 480), presentation_modnet=True,
        birefnet_model=model_path,
    )
    window.open_timer.stop()
    window.settings.strength = 0
    window.settings.auto_shutter = outcome == "manual-and-auto"
    displayed = []
    failures = []
    window.capture_displayed.connect(
        lambda result, guard: displayed.append((result, guard))
    )
    window.show()
    window.open_source()
    window.worker.model_failed.connect(failures.append)
    abstained = []
    window.worker.subject_abstained.connect(abstained.append)

    def wait_for(predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert predicate(), (
            window.view.message, window.worker.isRunning(), failures,
            window.settings.modnet_demo, window.settings.capture_matte,
            window.worker.modnet_result_drops,
        )

    try:
        window.modnet_box.click()
        window.birefnet_box.click()
        wait_for(lambda: window.modnet_last is not None and window.shoot.isEnabled())
        starting_epoch = window.worker.track_epoch
        assert starting_epoch == 0
        window.take_photo()
        wait_for(capture_started.is_set)
        if outcome == "manual-and-auto":
            assert auto_on_manual_frame.is_set()
        assert np.array_equal(frozen["frame"], rgb_a)
        assert frozen["subject_index"] == 0
        np.testing.assert_array_equal(frozen["subject_faces"][0], lm)
        wait_for(lambda: window.worker.track_epoch > starting_epoch)
        assert window.worker.track_epoch == starting_epoch + 1
        assert source_phase["value"] == "B"
        if outcome == "settings-change":
            window.set_background("mint")
        release_capture.set()
        if outcome in ("display", "manual-and-auto"):
            wait_for(lambda: bool(displayed))
            assert len(displayed) == 1
            assert window.dialog is not None
            np.testing.assert_array_equal(displayed[0][0]["ctx"].raw_frame, rgb_a)
            assert displayed[0][0]["ctx"].timestamp_ms == frozen["timestamp_ms"]
            assert displayed[0][1]["automatic"] is False
            assert displayed[0][1]["timestamp_ms"] == frozen["timestamp_ms"]
            assert displayed[0][1]["capture_worker_input"]["timestamp_ms"] == frozen["timestamp_ms"]
            assert "changed" not in abstained
            assert not failures
        elif outcome == "settings-change":
            wait_for(lambda: bool(failures))
            assert not displayed and window.dialog is None
            assert "settings changed" in failures[-1][3]
        else:
            wait_for(lambda: "ambiguous" in abstained)
            assert not displayed and window.dialog is None
            assert not failures
    finally:
        release_capture.set()
        window.close()
        app.processEvents()


def test_post_freeze_track_exemption_is_birefnet_manual_only():
    from idphoto.pipeline import Settings
    from idphoto.gui import _capture_stale_reasons, _copy_settings

    settings = Settings()
    request_settings = _copy_settings(settings)
    request = {
        "generation": 2,
        "auto_generation": 3,
        "automatic": False,
        "track_epoch": 4,
        "model": "birefnet",
        "settings": request_settings,
    }

    assert _capture_stale_reasons(request, settings, 2, 3, 5) == []

    settings.modnet_demo = True
    request_settings.modnet_demo = True
    assert _capture_stale_reasons(request, settings, 2, 3, 5) == []
    request["automatic"] = True
    assert _capture_stale_reasons(request, settings, 2, 3, 5) == [
        "face track changed"
    ]
    request["automatic"] = False
    request["model"] = "modnet"
    assert _capture_stale_reasons(request, settings, 2, 3, 5) == [
        "face track changed"
    ]


def test_birefnet_capture_choice_is_explicit_and_does_not_change_live_label(monkeypatch, tmp_path):
    from idphoto import presentation_birefnet, presentation_modnet

    app = QApplication.instance() or QApplication([])
    model_path = tmp_path / "portrait.onnx"
    model_path.write_bytes(b"synthetic path only")
    monkeypatch.setattr(presentation_modnet, "check_availability", lambda *_: "local-modnet")
    monkeypatch.setattr(presentation_modnet, "PresentationMODNet", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        presentation_birefnet, "check_availability",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        presentation_birefnet, "BirefNetPortrait",
        lambda *_args, **_kwargs: None,
    )
    window = MainWindow(
        "demo", "en", (640, 480), presentation_modnet=True,
        birefnet_model=model_path,
    )
    window.open_timer.stop()
    try:
        assert window.birefnet_box.isEnabled()
        assert not window.birefnet_box.isChecked()
        assert window.settings.capture_matte == "modnet"
        window.modnet_box.click()
        assert "MODNet" in window.modnet_preview_label.text()
        window.birefnet_box.click()
        assert window.settings.capture_matte == "birefnet-portrait"
        assert "BiRefNet" in window.birefnet_status.text()
        assert "MODNet" in window.modnet_preview_label.text()
    finally:
        window.close()
        app.processEvents()


@pytest.mark.parametrize(
    "scenario", [
        "auto-off", "settings-round-trip", "mode-round-trip", "head-share-change",
        "submit-guard", "manual-auto-off", "track-break-before-freeze",
    ])
def test_deferred_modnet_does_not_capture_stale_result(monkeypatch, scenario):
    from types import SimpleNamespace

    from idphoto.checker import Checker
    from idphoto import presentation_modnet

    app = QApplication.instance() or QApplication([])
    rgb, lm, _, _ = clothing_scene("jacket")
    rgb_a = rgb.copy()
    rgb_a[:12, :12] = [20, 40, 60]
    rgb_a[:12, -12:] = [20, 40, 60]
    rgb_b = rgb_a.copy()
    rgb_b[:12, :12] = [200, 30, 40]
    rgb_b[:12, -12:] = [200, 30, 40]
    source_phase = {"value": "A"}
    b_track_break_seen = {"value": False}
    prediction_started = threading.Event()
    release_prediction = threading.Event()
    finalize_paused = threading.Event()
    release_finalize = threading.Event()
    finalized = []
    captured = []
    capture_finished = threading.Event()

    class FixedDemo(DemoSource):
        def __init__(self):
            self.name = "slow fixed demo"
            self.lock_state = "locked"

        def read(self):
            time.sleep(.5)
            return (rgb_a if source_phase["value"] == "A" else rgb_b), lm

    class BlockingMODNet:
        def __init__(self, *_args, **_kwargs):
            self.calls = 0

        def prepare_input(self, _raw_rgb, *, capture=False):
            return True

        def predict_alpha(self, raw_rgb, *, capture=False):
            self.calls += 1
            if self.calls == 1:
                prediction_started.set()
                if not release_prediction.wait(5):
                    raise TimeoutError("test prediction was not released")
            return presentation_modnet.ModnetPrediction(
                _alpha_with_background(raw_rgb.shape[:2]),
                {"inference_ms": 1.0, "input_width": raw_rgb.shape[1],
                 "input_height": raw_rgb.shape[0], "capture": capture, "provider": "test"},
            )

        def apply_prediction(self, ctx, prediction, background):
            ctx.alpha = prediction.alpha
            ctx.wall_color = presentation_modnet.estimate_wall_color(ctx.linear_rgb, ctx.alpha)
            if background is None:
                ctx.output_frame = ctx.relit_frame
            else:
                wall = presentation_modnet.estimate_local_wall(
                    ctx.linear_rgb, ctx.alpha, ctx.wall_color)
                ctx.output_frame = matting.composite(ctx, wall, background)

    import idphoto.gui as gui

    original_pipeline = gui.Pipeline
    original_finalize = original_pipeline.finalize_deferred_model

    def create_pipeline(*args, **kwargs):
        pipeline = original_pipeline(*args, **kwargs)
        original_process = pipeline.process

        def process(raw, *process_args, **process_kwargs):
            ctx, fired = original_process(raw, *process_args, **process_kwargs)
            is_b_marker = any(
                raw[0, x, 0] > 150 and raw[0, x, 1] < 60 and raw[0, x, 2] < 70
                for x in (0, -1)
            )
            if (
                scenario == "track-break-before-freeze"
                and is_b_marker
                and not b_track_break_seen["value"]
            ):
                b_track_break_seen["value"] = True
                ctx.stats["landmark_source"] = "native"
                ctx.stats["track_break"] = True
            return ctx, fired

        pipeline.process = process
        pipeline.shutter.frames = 1
        return pipeline

    def record_finalize(self, ctx, settings):
        finalized.append(settings.auto_shutter)
        result = original_finalize(self, ctx, settings)
        if scenario == "submit-guard":
            finalize_paused.set()
            if not release_finalize.wait(5):
                raise TimeoutError("test finalization was not released")
        return result

    monkeypatch.setattr(gui, "Pipeline", create_pipeline)
    monkeypatch.setattr(original_pipeline, "finalize_deferred_model", record_finalize)
    monkeypatch.setattr(
        Checker, "evaluate",
        lambda _self, _ctx: {
            n: SimpleNamespace(
                rule=n, key="even_light", value=None, passed=True,
                message=None, bound="", detail={},
            )
            for n in range(1, 8)
        },
    )
    monkeypatch.setattr(gui, "open_source", lambda *_: FixedDemo())
    monkeypatch.setattr(presentation_modnet, "check_availability", lambda *_: "local-model")
    monkeypatch.setattr(presentation_modnet, "PresentationMODNet", BlockingMODNet)
    window = MainWindow("demo", "ko", (640, 480), presentation_modnet=True)
    window.open_timer.stop()
    window.settings.strength = 0

    def on_captured(result):
        captured.append(result)
        capture_finished.set()

    monkeypatch.setattr(window, "on_captured", on_captured)
    window.show()
    window.open_source()
    errors = []
    window.worker.model_failed.connect(errors.append)

    def wait_for(predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert predicate(), (finalized, captured, errors, window.worker.isRunning())

    try:
        window.modnet_box.click()
        wait_for(prediction_started.is_set)
        if scenario == "settings-round-trip":
            window.set_format("profile")
            window.set_format("resume")
        elif scenario == "mode-round-trip":
            window.modnet_box.setChecked(False)
            window.modnet_box.setChecked(True)
        elif scenario == "head-share-change":
            window.set_head_share(70)
        elif scenario == "auto-off":
            window.auto_box.setChecked(False)
        elif scenario == "manual-auto-off":
            window.worker.request_manual_capture()
            window.auto_box.setChecked(False)
        elif scenario == "track-break-before-freeze":
            window.worker.request_manual_capture()
            source_phase["value"] = "B"
            wait_for(lambda: window.worker.track_epoch > 0)
        release_prediction.set()
        if scenario in ("settings-round-trip", "mode-round-trip", "head-share-change"):
            wait_for(lambda: window.worker.modnet_result_drops > 0)
            assert not finalized
            assert not captured
        elif scenario == "auto-off":
            wait_for(lambda: bool(finalized))
            assert finalized[0] is False
            deadline = time.monotonic() + .2
            while time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            assert not capture_finished.is_set()
            assert not captured
        elif scenario == "manual-auto-off":
            wait_for(capture_finished.is_set)
            assert len(captured) == 1
            assert finalized[-1] is False
            assert captured[0]["_capture_display_guard"]["automatic"] is False
        elif scenario == "track-break-before-freeze":
            wait_for(lambda: window.worker.modnet_result_drops > 0)
            assert not captured
            assert not capture_finished.is_set()
        else:
            wait_for(finalize_paused.is_set)
            window.set_format("profile")
            window.set_format("resume")
            release_finalize.set()
            wait_for(lambda: bool(finalized))
            assert finalized[0] is True
            deadline = time.monotonic() + .2
            while time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            assert not capture_finished.is_set()
            assert not captured
    finally:
        release_prediction.set()
        release_finalize.set()
        window.close()
        app.processEvents()


def test_default_fba_disables_with_explicit_reason_when_unavailable(monkeypatch):
    from tools import presentation_fba

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(presentation_fba, "check_availability",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("missing research weights")))
    window = MainWindow("demo", "ko", (640, 480), presentation_fba=True, start_fba=True)
    window.open_timer.stop()
    window.settings.auto_shutter = False
    captured = []
    monkeypatch.setattr(window, "on_captured", captured.append)
    window.open_source()
    try:
        assert not window.fba_box.isEnabled()
        assert not window.settings.fba_demo
        assert window.settings.clothing_enabled
        assert "missing research weights" in window.fba_warning.text()
        assert "전통 Clothing으로 시작" in window.fba_warning.text()
        assert "--accept-noncommercial" in window.fba_warning.text()
        assert not window.worker.settings.fba_demo
        deadline = time.monotonic() + 15
        while (window.live_info is None or not window.shoot.isEnabled()) and time.monotonic() < deadline:
            time.sleep(.02)
            app.processEvents()
        assert window.live_info is not None and window.shoot.isEnabled()
        window.take_photo()
        deadline = time.monotonic() + 15
        while not captured and time.monotonic() < deadline:
            time.sleep(.02)
            app.processEvents()
        assert captured and not captured[0]["report"]["settings"]["fba_demo"]
        assert captured[0]["report"]["settings"]["clothing_enabled"]
        window.toggle_language()
        assert "Started in traditional Clothing mode" in window.fba_warning.text()
        window.clothing_box.click()
        assert not window.settings.clothing_enabled
    finally:
        window.close()
        app.processEvents()


def test_unavailable_mps_disables_fba_without_silent_cpu_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    window = MainWindow("demo", "ko", (640, 480), presentation_fba=True, fba_device="mps")
    window.open_timer.stop()
    assert not window.fba_box.isEnabled()
    assert "MPS acceleration is unavailable" in window.fba_warning.text()
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


def test_cli_selects_prepared_fba_on_startup(monkeypatch):
    import idphoto.gui as gui

    calls = {}

    class App:
        def installEventFilter(self, _keys):
            pass

        def exec(self):
            return 0

    class Window:
        def __init__(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs

        def show(self):
            pass

    monkeypatch.setattr(gui, "QApplication", lambda *_args: App())
    monkeypatch.setattr(gui, "MainWindow", Window)
    monkeypatch.setattr(gui, "KeyFilter", lambda _window: None)
    assert gui.main(["--source", "demo", "--lang", "en"]) == 0
    assert calls["kwargs"]["presentation_fba"]
    assert calls["kwargs"]["start_fba"]
    assert calls["kwargs"]["fba_device"] == "cpu"


def test_prepared_default_can_switch_to_traditional_and_back(monkeypatch):
    from tools import presentation_fba

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(presentation_fba, "check_availability", lambda: None)
    window = MainWindow("demo", "en", (640, 480), presentation_fba=True, start_fba=True)
    window.open_timer.stop()
    try:
        assert window.fba_box.isChecked()
        assert window.settings.fba_demo and window.settings.clothing_enabled
        window.fba_box.click()
        assert not window.settings.fba_demo and window.settings.clothing_enabled
        window.fba_box.click()
        assert window.settings.fba_demo
        window.clothing_box.click()
        assert not window.settings.fba_demo and not window.settings.clothing_enabled
        assert not window.fba_box.isEnabled()
    finally:
        window.close()
        app.processEvents()


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_real_fba_demo_runs_in_gui_worker_when_assets_exist(monkeypatch, device):
    torch = pytest.importorskip("torch")
    from tools.presentation_fba import PresentationFBA, check_availability

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    try:
        check_availability(device)
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
    inferred_contexts = []
    original = PresentationFBA.__call__

    def observe(self, ctx, background):
        original(self, ctx, background)
        assert np.isfinite(ctx.alpha).all()
        inferred_contexts.append(ctx)
        called.set()

    monkeypatch.setattr(PresentationFBA, "__call__", observe)
    window = MainWindow("demo", "ko", (640, 480), presentation_fba=True,
                        fba_device=device, start_fba=True)
    window.open_timer.stop()
    window.settings.strength = 0
    window.settings.auto_shutter = False
    captured = []
    monkeypatch.setattr(window, "on_captured", captured.append)
    window.open_source()
    try:
        assert window.fba_box.isEnabled() and window.fba_box.isChecked()
        assert window.worker.settings.fba_demo and window.worker.settings.clothing_enabled
        deadline = time.monotonic() + 30
        while (not called.is_set() or not window.view.info or not window.view.info.get("fba_display")) \
                and window.worker.isRunning() and time.monotonic() < deadline:
            time.sleep(.02)
            app.processEvents()
        assert called.is_set(), window.view.message
        assert window.view.info["fba_display"]
        assert window.view.image == window.fba_last["image"].flipped(Qt.Orientation.Horizontal)
        window.take_photo()
        deadline = time.monotonic() + 30
        while not captured and window.worker.isRunning() and time.monotonic() < deadline:
            time.sleep(.02)
            app.processEvents()
        assert captured and captured[0]["report"]["settings"]["fba_demo"]
        assert any(ctx is captured[0]["ctx"] for ctx in inferred_contexts)
        assert np.isfinite(captured[0]["ctx"].alpha).all()
    finally:
        window.close()
        app.processEvents()


def test_fba_toggle_displays_own_snapshot_in_main_view_and_clears_on_face_loss(monkeypatch):
        from tools import presentation_fba

        app = QApplication.instance() or QApplication([])
        rgb, lm, _, _ = clothing_scene("jacket")
        face_present = threading.Event()
        face_present.set()

        class FixedDemo(DemoSource):
            def __init__(self):
                self.name = "fixed demo"
                self.lock_state = "locked"

            def read(self):
                time.sleep(.03)
                return rgb, lm if face_present.is_set() else None

        class RedFBA:
            def __call__(self, ctx, background):
                ctx.output_frame = np.zeros_like(ctx.output_frame)
                ctx.output_frame[:, :, 0] = 1

        import idphoto.gui as gui

        monkeypatch.setattr(gui, "open_source", lambda *_: FixedDemo())
        monkeypatch.setattr(presentation_fba, "check_availability", lambda: None)
        monkeypatch.setattr(presentation_fba, "PresentationFBA", RedFBA)
        window = MainWindow("demo", "ko", (640, 480), presentation_fba=True)
        window.open_timer.stop()
        window.settings.strength = 0
        window.settings.auto_shutter = False
        window.open_source()

        def wait_for(predicate):
            deadline = time.monotonic() + 15
            while not predicate() and time.monotonic() < deadline:
                time.sleep(.01)
                app.processEvents()
            assert predicate()

        try:
            wait_for(lambda: window.view.image is not None)
            assert window.view.image.pixelColor(0, 0).getRgb()[:3] != (255, 0, 0)
            window.fba_box.click()
            wait_for(lambda: window.view.info.get("fba_display"))
            assert window.view.image.pixelColor(0, 0).getRgb()[:3] == (255, 0, 0)
            assert window.fba_image.pixmap() is not None
            old_snapshot = window.fba_last

            window.set_before(True)
            wait_for(lambda: window.view.image.pixelColor(0, 0).getRgb()[:3] != (255, 0, 0))
            window.set_before(False)
            wait_for(lambda: window.view.image.pixelColor(0, 0).getRgb()[:3] == (255, 0, 0))

            face_present.clear()
            wait_for(lambda: not window.live_info["fba_eligible"])
            assert window.fba_last is None
            assert not window.view.info.get("fba_display")
            window.on_fba_snapshot(old_snapshot)
            assert window.fba_last is None

            face_present.set()
            wait_for(lambda: window.view.info.get("fba_display"))
            window.fba_box.click()
            wait_for(lambda: not window.view.info.get("fba_display"))
            assert window.view.image.pixelColor(0, 0).getRgb()[:3] != (255, 0, 0)
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
        wait_for(lambda: "스냅샷" in window.fba_status.text())
        assert captured[0]["report"]["settings"]["fba_demo"]
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


@pytest.mark.parametrize("setting", ["background", "strength"])
def test_fba_snapshot_before_setting_roundtrip_is_not_reused(monkeypatch, setting):
    from tools import presentation_fba

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(presentation_fba, "check_availability", lambda: None)
    window = MainWindow("demo", "ko", (640, 480), presentation_fba=True)
    window.open_timer.stop()
    monkeypatch.setattr("idphoto.gui.time.monotonic", lambda: 10.0)
    window.fba_box.click()
    snapshot = {"generation": window.fba_generation, "background": window.settings.background,
                "strength": window.settings.strength, "timestamp_ms": 9999}
    if setting == "background":
        window.set_background("mint")
        window.set_background(snapshot["background"])
    else:
        original = round(snapshot["strength"] * 100)
        window.set_strength(0 if original else 100)
        window.set_strength(original)
    window.on_fba_snapshot(snapshot)
    assert window.fba_last is None
    assert window.fba_generation != snapshot["generation"]
    fresh = dict(snapshot, generation=window.fba_generation, timestamp_ms=10001)
    window.on_fba_snapshot(fresh)
    assert window.fba_last == fresh
    newer = dict(fresh, timestamp_ms=10002)
    window.on_fba_snapshot(newer)
    window.on_fba_snapshot(fresh)
    assert window.fba_last == newer
    window.close()
    app.processEvents()
