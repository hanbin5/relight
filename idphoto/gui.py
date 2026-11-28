"""Module C: the desktop app (PySide6), in Korean and English (plan Figure 4).

The live view shows the relit, background-replaced frame with the crop guide. The side
panel holds the format, background and strength choices, the seven checks with the
coaching message for the first failing rule, the auto shutter and the notice that the
photos are not for passports or government IDs.

    uv run idphoto                     # webcam 0
    uv run idphoto --source demo       # synthetic sitter, no camera needed
    uv run idphoto --source photo.jpg --lang en

Keys: hold B to see the frame before processing, Space to take a photo now.
"""

import argparse
import copy
import functools
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import color, config, geometry
from .capture import DemoSource, FrameGrabber, LockMonitor, camera_permission, is_camera, open_source
from .fba_jobs import ModelJobs
from .i18n import t
from .pipeline import BACKGROUNDS, Pipeline, Settings, to_live

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)  # some OpenCV builds point Qt at their own plugins

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QThread, QTimer, QUrl, Signal  # noqa: E402
from PySide6.QtGui import QColor, QDesktopServices, QFont, QImage, QPainter, QPen, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QVBoxLayout, QWidget,
)

ACCENT = "#2f6fdf"
FALLBACK_LOCKS = ("unsupported", "failed", "drifting")


def to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _copy_settings(settings: Settings) -> Settings:
    snapshot = copy.copy(settings)
    snapshot.head_share = settings.head_share.copy()
    return snapshot


def _same_modnet_settings(left: Settings, right: Settings) -> bool:
    return (
        left.format == right.format
        and left.head_share == right.head_share
        and left.background == right.background
        and left.strength == right.strength
        and left.capture_matte == right.capture_matte
        and left.modnet_demo == right.modnet_demo
        and left.fba_demo == right.fba_demo
    )


def _same_capture_settings(left: Settings, right: Settings) -> bool:
    return (
        _same_modnet_settings(left, right)
        and left.clothing_enabled == right.clothing_enabled
    )


def _capture_stale_reasons(request, current_settings, current_generation,
                           current_auto_generation, current_track_epoch):
    reasons = []
    if request["generation"] != current_generation:
        reasons.append("settings changed")
    if not _same_capture_settings(request["settings"], current_settings):
        reasons.append("capture settings changed")
    if ((request["automatic"] or request["model"] != "birefnet")
            and request["track_epoch"] != current_track_epoch):
        reasons.append("face track changed")
    if request["automatic"] and request["auto_generation"] != current_auto_generation:
        reasons.append("AUTO setting changed")
    return reasons


def _run_serialized_fba(lock, operation, *args, **kwargs):
    with lock:
        return operation(*args, **kwargs)


def _valid_modnet_capture(ctx, capture_frame, result: dict, frame_id: int) -> bool:
    def valid_image(value):
        return (
            isinstance(value, np.ndarray)
            and value.ndim == 3
            and value.shape[0] > 0
            and value.shape[1] > 0
            and value.shape[2] == 3
            and value.dtype == np.uint8
        )

    raw = ctx.raw_frame
    alpha = ctx.alpha
    landmarks = ctx.landmarks
    light = ctx.light_coeffs
    face_mask = ctx.face_mask
    output = ctx.output_frame
    try:
        return (
            result.get("frame_id") == frame_id
            and result.get("timestamp_ms") == ctx.timestamp_ms
            and valid_image(capture_frame)
            and valid_image(raw)
            and ctx.subject_status == "ready"
            and isinstance(alpha, np.ndarray)
            and alpha.shape == raw.shape[:2]
            and np.isfinite(alpha).all()
            and landmarks is not None
            and np.asarray(landmarks).size > 0
            and np.isfinite(landmarks).all()
            and light is not None
            and np.isfinite(np.asarray(light)).all()
            and face_mask is not None
            and bool(np.asarray(face_mask).any())
            and isinstance(output, np.ndarray)
            and output.shape == raw.shape
            and np.isfinite(output).all()
        )
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------------------
# Processing thread


class Worker(QThread):
    frame_ready = Signal(object)
    captured = Signal(object)
    failed = Signal(str)
    model_snapshot = Signal(object)
    modnet_preparing = Signal(int)
    model_failed = Signal(object)
    subject_abstained = Signal(str)

    def __init__(self, source, settings: Settings, fba_factory=None, modnet_factory=None,
                 birefnet_factory=None, face_model_path=None, settings_lock=None):
        super().__init__()
        self.source = source  # opened on the main thread (see MainWindow.open_source)
        self.settings = settings
        self.settings_lock = settings_lock or threading.RLock()
        self.fba_factory = fba_factory
        self.modnet_factory = modnet_factory
        self.birefnet_factory = birefnet_factory
        self.face_model_path = face_model_path
        self.running = True
        self.view_before = False
        self.mirror = True
        self.shutter_request = None
        self.rearm_request = False
        self.model_generation = 0
        self.auto_shutter_generation = 0
        self.track_epoch = 0
        self.modnet_result_drops = 0
        self.modnet_failed = False

    def _take_frame_request(self, source_read_timestamp_ms):
        with self.settings_lock:
            frame_settings = _copy_settings(self.settings)
            generation = self.model_generation
            auto_generation = self.auto_shutter_generation
            manual_capture_timestamp_ms = self.shutter_request
            if (manual_capture_timestamp_ms is not None
                    and manual_capture_timestamp_ms < source_read_timestamp_ms):
                self.shutter_request = None
            else:
                manual_capture_timestamp_ms = None
        return (
            frame_settings,
            generation,
            auto_generation,
            manual_capture_timestamp_ms is not None,
            manual_capture_timestamp_ms,
        )

    def request_manual_capture(self):
        with self.settings_lock:
            if self.shutter_request is None:
                self.shutter_request = time.monotonic() * 1000

    def run(self):
        source = self.source
        try:
            detector = None
            if not isinstance(source, DemoSource):
                from .landmarks import FaceLandmarkDetector

                detector = FaceLandmarkDetector(model_path=self.face_model_path)
        except Exception as exc:
            source.release()
            self.failed.emit(str(exc))
            return
        grabber = FrameGrabber(source)
        grabber.start()
        pipeline = Pipeline(detector, fba_factory=self.fba_factory,
                            modnet_factory=self.modnet_factory,
                            birefnet_factory=self.birefnet_factory)
        reacquiring_detector = None
        using_reacquirer = False

        def select_detector(use_reacquirer):
            nonlocal detector, reacquiring_detector, using_reacquirer
            if detector is None:
                return
            if use_reacquirer and not using_reacquirer:
                from .center_face import CenterFaceDetector

                reacquiring_detector = CenterFaceDetector(
                    model_path=self.face_model_path
                )
                pipeline.detector = reacquiring_detector
                using_reacquirer = True
            elif not use_reacquirer and using_reacquirer:
                detector.reset()
                reacquiring_detector.close()
                reacquiring_detector = None
                pipeline.detector = detector
                using_reacquirer = False
                pipeline.relighter.reset()
                pipeline.tracker.reset()
                pipeline.shutter.rearm()
                pipeline.checker.sharpness_seen.clear()

        jobs = None
        completed_modnet = queue.Queue()
        modnet_ready = threading.Event()
        pending_modnet = {}
        manual_capture_target = {
            "frame_id": None,
            "generation": None,
            "request_timestamp_ms": None,
            "read_timestamp_ms": None,
        }

        def clear_manual_capture_target():
            manual_capture_target.update(
                frame_id=None,
                generation=None,
                request_timestamp_ms=None,
                read_timestamp_ms=None,
            )
        track_epoch = {"value": 0}
        if self.fba_factory is not None or self.modnet_factory is not None:
            fba_lock = threading.Lock()

            def preview(ctx, settings, generation, frame_id, timestamp_ms=None):
                if settings.modnet_demo:
                    if not pipeline.prepare_modnet_input(ctx):
                        return {
                            "model": "modnet-preparing",
                            "generation": generation,
                        }
                    prediction = pipeline.predict_modnet_alpha(ctx)
                    return {
                        "model": "modnet-alpha",
                        "frame_id": frame_id,
                        "timestamp_ms": timestamp_ms,
                        "generation": generation,
                        "prediction": prediction,
                    }
                snapshot = copy.copy(ctx)
                _run_serialized_fba(
                    fba_lock, pipeline._apply_fba, snapshot, settings,
                    settings.background_linear(),
                )
                model = "fba"
                model_stats = snapshot.stats.get(model, {})
                result = {"image": to_qimage(color.encode_u8(snapshot.output_frame)),
                          "timestamp_ms": ctx.timestamp_ms, "generation": generation, "model": model,
                          "frame_id": frame_id, "face_detected": ctx.landmarks is not None,
                          "inference_ms": model_stats.get("inference_ms"),
                          "inference_started_ms": model_stats.get("inference_started_ms"),
                          "inference_finished_ms": model_stats.get("inference_finished_ms"),
                          "wall_estimate_ms": model_stats.get("wall_estimate_ms"),
                          "composite_ms": model_stats.get("composite_ms"),
                          "execution_provider": model_stats.get("provider"),
                          "provider_backends": model_stats.get("provider_backends"),
                          "prepare_finished_ms": ctx.stats.get("prepare_finished_ms"),
                          "pipeline_timings_ms": ctx.stats.get("timings_ms"),
                          "input_width": model_stats.get("input_width"),
                          "input_height": model_stats.get("input_height"),
                          "background": settings.background, "strength": settings.strength,
                          "geometry": ctx.geometry}
                return result

            def preview_done(result):
                if result["model"] == "modnet-preparing":
                    self.modnet_preparing.emit(result["generation"])
                    return
                if result["model"] == "modnet-alpha":
                    completed_modnet.put(result)
                    modnet_ready.set()
                    return
                self.model_snapshot.emit(result)

            capture_pipeline = Pipeline(
                fba_factory=self.fba_factory,
                modnet_factory=self.modnet_factory,
                birefnet_factory=self.birefnet_factory,
            )

            def render_capture(frame, ctx, settings, request):
                if settings.modnet_demo:
                    result = capture_pipeline.render_final(frame, ctx, settings)
                else:
                    result = _run_serialized_fba(
                        fba_lock, pipeline.render_final, frame, ctx, settings
                    )
                capture_ctx = result.get("ctx")
                if capture_ctx is not None:
                    capture_ctx.stats["capture_source"] = {
                        "frame_id": request["frame_id"],
                        "timestamp_ms": request["timestamp_ms"],
                        "manual_request_timestamp_ms": request["manual_request_timestamp_ms"],
                        "source_read_timestamp_ms": request["source_read_timestamp_ms"],
                    }
                    model_stats = capture_ctx.stats.get("birefnet_portrait")
                    if isinstance(model_stats, dict):
                        model_stats.update({
                            "source_frame_id": request["frame_id"],
                            "source_timestamp_ms": request["timestamp_ms"],
                            "source_read_timestamp_ms": request["source_read_timestamp_ms"],
                            "manual_request_timestamp_ms":
                                request["manual_request_timestamp_ms"],
                        })
                result["_capture_request"] = request
                return result

            def capture_done(result):
                request = result.pop("_capture_request")
                capture_worker_input = result.pop("capture_worker_input", None)
                capture_timing = result.setdefault("_capture_event_timing_ms", {})
                capture_timing["worker_result_callback_ms"] = time.monotonic() * 1000
                if not self.running:
                    return
                if self.modnet_failed and request["settings"].modnet_demo:
                    return
                with self.settings_lock:
                    current_settings = _copy_settings(self.settings)
                    current_generation = self.model_generation
                    current_auto_generation = self.auto_shutter_generation
                stale_reasons = _capture_stale_reasons(
                    request, current_settings, current_generation,
                    current_auto_generation, track_epoch["value"],
                )
                capture_timing["worker_guard_checked_ms"] = time.monotonic() * 1000
                if stale_reasons:
                    if "face track changed" in stale_reasons and request["settings"].modnet_demo:
                        self.subject_abstained.emit("changed")
                        return
                    self.model_failed.emit((
                        True, request["model"], None,
                        "Capture result discarded because "
                        + ", ".join(stale_reasons) + ".",
                    ))
                    return
                if result.get("subject_blocked"):
                    self.subject_abstained.emit(result["subject_blocked"])
                    return
                result["_capture_display_guard"] = {
                    "generation": request["generation"],
                    "auto_generation": request["auto_generation"],
                    "automatic": request["automatic"],
                    "track_epoch": request["track_epoch"],
                    "settings": request["settings"],
                    "model": request["model"],
                    "frame_id": request["frame_id"],
                    "timestamp_ms": request["timestamp_ms"],
                    "manual_request_timestamp_ms": request["manual_request_timestamp_ms"],
                    "source_read_timestamp_ms": request["source_read_timestamp_ms"],
                    "capture_worker_input": capture_worker_input,
                    "model_job_timing_ms": result.get("model_job_timing_ms"),
                    "capture_timing_ms": capture_timing,
                }
                capture_timing["worker_signal_emitted_ms"] = time.monotonic() * 1000
                self.captured.emit(result)

            def job_error(capture, args, error):
                settings = args[2] if capture else args[1]
                generation = None if capture else args[2]
                if (not capture and settings.modnet_demo
                        and generation == self.model_generation and self.settings.modnet_demo):
                    self.modnet_failed = True
                model = (
                    "birefnet" if capture and settings.capture_matte == "birefnet-portrait"
                    else "modnet" if settings.modnet_demo else "fba"
                )
                self.model_failed.emit((capture, model, generation, error))

            jobs = ModelJobs(preview, render_capture, preview_done, capture_done, job_error)

            def submit_capture(frame, ctx, settings, frame_id, generation,
                               auto_generation, automatic, manual_request_timestamp_ms=None):
                snapshot = _copy_settings(settings)
                request = {
                    "frame_id": frame_id,
                    "timestamp_ms": ctx.timestamp_ms,
                    "source_read_timestamp_ms": ctx.timestamp_ms,
                    "manual_request_timestamp_ms": manual_request_timestamp_ms,
                    "generation": generation,
                    "auto_generation": auto_generation,
                    "automatic": automatic,
                    "track_epoch": track_epoch["value"],
                    "settings": snapshot,
                    "model": (
                        "birefnet" if snapshot.capture_matte == "birefnet-portrait"
                        else "modnet" if snapshot.modnet_demo else "fba"
                    ),
                }
                return jobs.submit_capture(frame.copy(), ctx, snapshot, request)

            def apply_ready_modnet():
                while True:
                    try:
                        result = completed_modnet.get_nowait()
                    except queue.Empty:
                        modnet_ready.clear()
                        return
                    frame_id = result["frame_id"]
                    pending = pending_modnet.pop(frame_id, None)
                    if self.modnet_failed:
                        continue
                    if pending is None:
                        self.modnet_result_drops += 1
                        if manual_capture_target["frame_id"] == frame_id:
                            clear_manual_capture_target()
                            self.model_failed.emit((
                                True, "modnet", None,
                                "Requested MODNet frame was unavailable; no photo was captured",
                            ))
                        continue
                    ctx, result_settings, generation, result_epoch, auto_generation = pending
                    if result_epoch != track_epoch["value"]:
                        self.modnet_result_drops += 1
                        if manual_capture_target["frame_id"] == frame_id:
                            clear_manual_capture_target()
                            self.model_failed.emit((
                                True, "modnet", None,
                                "Requested matte belongs to an earlier face track",
                            ))
                        continue
                    with self.settings_lock:
                        current_settings = _copy_settings(self.settings)
                        current_generation = self.model_generation
                        current_auto_generation = self.auto_shutter_generation
                    if (generation != current_generation or not result_settings.modnet_demo
                            or not _same_modnet_settings(result_settings, current_settings)):
                        self.modnet_result_drops += 1
                        if manual_capture_target["frame_id"] == frame_id:
                            clear_manual_capture_target()
                            self.model_failed.emit((
                                True, "modnet", None,
                                "Settings changed before the requested frame was ready",
                            ))
                        continue
                    try:
                        prediction = result.pop("prediction")
                        pipeline.apply_modnet_prediction(
                            ctx, prediction, result_settings.background_linear())
                        finalize_settings = _copy_settings(result_settings)
                        auto_generation_matches = auto_generation == current_auto_generation
                        finalize_settings.auto_shutter = bool(
                            result_settings.auto_shutter
                            and current_settings.auto_shutter
                            and auto_generation_matches
                        )
                        result_ctx, model_fired = pipeline.finalize_deferred_model(
                            ctx, finalize_settings)
                        if not auto_generation_matches:
                            pipeline.shutter.rearm()
                    except Exception as exc:
                        self.modnet_failed = True
                        self.model_failed.emit((False, "modnet", generation,
                                                f"MODNet checks failed: {exc}"))
                        continue
                    finalized_ms = time.monotonic() * 1000
                    model_stats = result_ctx.stats.get("modnet", {})
                    face_mask_valid = (
                        result_ctx.face_mask is not None and bool(result_ctx.face_mask.any())
                    )
                    rule_count = len(result_ctx.rule_verdicts or {})
                    fully_processed = bool(
                        result_ctx.light_coeffs is not None
                        and face_mask_valid
                        and result_ctx.geometry is not None
                        and "crop_box" in result_ctx.geometry
                        and rule_count == 7
                    )
                    result.update({
                        "image": to_qimage(color.encode_u8(result_ctx.output_frame)),
                        "timestamp_ms": result_ctx.timestamp_ms,
                        "generation": generation,
                        "auto_shutter_generation": auto_generation,
                        "model": "modnet",
                        "track_epoch": result_epoch,
                        "subject_status": result_ctx.subject_status,
                        "face_detected": result_ctx.landmarks is not None,
                        "fully_processed": fully_processed,
                        "light_valid": result_ctx.light_coeffs is not None,
                        "face_mask_valid": face_mask_valid,
                        "geometry_valid": result_ctx.geometry is not None
                            and "crop_box" in result_ctx.geometry,
                        "rule_count": rule_count,
                        "landmark_source": result_ctx.stats.get("landmark_source", "none"),
                        "track_break": result_ctx.stats.get("track_break", False),
                        "landmark_detector_calls": result_ctx.stats.get("landmark_detector_calls", 0),
                        "inference_ms": model_stats.get("inference_ms"),
                        "inference_started_ms": model_stats.get("inference_started_ms"),
                        "inference_finished_ms": model_stats.get("inference_finished_ms"),
                        "wall_estimate_ms": model_stats.get("wall_estimate_ms"),
                        "composite_ms": model_stats.get("composite_ms"),
                        "execution_provider": model_stats.get("provider"),
                        "provider_backends": model_stats.get("provider_backends"),
                        "prepare_finished_ms": result_ctx.stats.get("prepare_finished_ms"),
                        "pipeline_timings_ms": result_ctx.stats.get("timings_ms"),
                        "input_width": model_stats.get("input_width"),
                        "input_height": model_stats.get("input_height"),
                        "background": result_settings.background,
                        "strength": result_settings.strength,
                        "geometry": result_ctx.geometry,
                        "verdicts": result_ctx.rule_verdicts,
                        "shutter": (pipeline.shutter.count, pipeline.shutter.frames,
                                    pipeline.shutter.armed),
                        "stats": {key: result_ctx.stats.get(key)
                                  for key in ("r_lr_before", "r_lr_after")},
                        "finalized_ms": finalized_ms,
                    })
                    manual_target = manual_capture_target["frame_id"] == frame_id \
                        and manual_capture_target["generation"] == generation
                    manual_request_timestamp_ms = (
                        manual_capture_target["request_timestamp_ms"] if manual_target else None
                    )
                    manual_read_timestamp_ms = (
                        manual_capture_target["read_timestamp_ms"] if manual_target else None
                    )
                    safe_track = result_ctx.stats.get("landmark_source") in ("native", "provided") \
                        and not result_ctx.stats.get("track_break", False)
                    if manual_target:
                        clear_manual_capture_target()
                        result_ctx.stats["manual_capture_source_boundary_ms"] = \
                            manual_read_timestamp_ms
                        result_ctx.stats["manual_capture_request_ms"] = \
                            manual_request_timestamp_ms
                        if result_ctx.subject_status != "ready":
                            self.subject_abstained.emit(result_ctx.subject_status)
                            manual_target = False
                        elif not safe_track:
                            self.model_failed.emit((
                                True, "modnet", None,
                                "Requested frame is from an uncertain or changing face track",
                            ))
                    capture_frame = result_ctx.full_frame \
                        if result_ctx.full_frame is not None else result_ctx.raw_frame
                    with self.settings_lock:
                        current_settings = _copy_settings(self.settings)
                        current_generation = self.model_generation
                        current_auto_generation = self.auto_shutter_generation
                        current_settings_match = (
                            generation == current_generation
                            and _same_modnet_settings(result_settings, current_settings)
                        )
                        auto_capture_current = (
                            auto_generation == current_auto_generation
                            and current_settings.auto_shutter
                        )
                        should_capture = (
                            (model_fired and auto_capture_current)
                            or (manual_target and safe_track)
                        )
                        if (not current_settings_match
                                or auto_generation != current_auto_generation):
                            pipeline.shutter.rearm()
                        if should_capture and (
                            not current_settings_match
                            or not safe_track
                            or result_ctx.geometry is None
                            or "crop_box" not in result_ctx.geometry
                            or not _valid_modnet_capture(
                                result_ctx, capture_frame, result, frame_id)
                        ):
                            pipeline.shutter.rearm()
                            if manual_target:
                                self.model_failed.emit((
                                    True, "modnet", None,
                                    "Requested frame failed landmark, lighting, geometry, or settings checks",
                                ))
                            should_capture = False
                        if should_capture:
                            pipeline.shutter.armed = False
                            if not submit_capture(
                                capture_frame, result_ctx, current_settings, frame_id,
                                generation, auto_generation,
                                automatic=bool(model_fired and not manual_target),
                                manual_request_timestamp_ms=manual_request_timestamp_ms,
                            ):
                                capture_model = (
                                    "birefnet" if current_settings.capture_matte == "birefnet-portrait"
                                    else "modnet"
                                )
                                self.model_failed.emit((True, capture_model, None,
                                                        "A model capture is already in progress"))
                    self.model_snapshot.emit(result)
        monitor = LockMonitor()
        seq, fps, last = 0, 0.0, time.perf_counter()
        while self.running:
            if self.modnet_factory is not None and self.settings.modnet_demo:
                apply_ready_modnet()
            item = grabber.latest(seq)
            if item is None:
                if grabber.error is not None:
                    self.failed.emit(str(grabber.error))
                    break
                if self.modnet_factory is not None and self.settings.modnet_demo:
                    modnet_ready.wait(0.003)
                    modnet_ready.clear()
                else:
                    self.msleep(3)
                continue
            seq, frame, lm_full, ts = item
            live, lm_live = to_live(frame, lm_full)
            if self.rearm_request:
                pipeline.shutter.rearm()
                self.rearm_request = False
            (
                settings,
                generation,
                auto_generation,
                manual_capture_requested,
                manual_capture_request_timestamp_ms,
            ) = self._take_frame_request(ts)
            try:
                if not (settings.modnet_demo and self.modnet_failed):
                    select_detector(settings.modnet_demo)
            except Exception as exc:
                if settings.modnet_demo:
                    self.modnet_failed = True
                    self.model_failed.emit((False, "modnet", generation,
                                            f"Landmark reacquisition unavailable: {exc}"))
                else:
                    self.failed.emit(f"Unable to restore the native landmark detector: {exc}")
                    break
            if (not settings.modnet_demo
                    and manual_capture_target["frame_id"] is not None):
                clear_manual_capture_target()
                self.model_failed.emit((
                    True, "modnet", None,
                    "MODNet mode changed before the requested frame was ready",
                ))
            if settings.modnet_demo and jobs is not None and not self.modnet_failed:
                if manual_capture_target["frame_id"] is None:
                    jobs.submit_preview(live, settings, generation, seq, ts)
            try:
                ctx, fired = pipeline.process(live, settings, landmarks=lm_live, timestamp_ms=ts,
                                              full_frame=frame,
                                              defer_fba=settings.fba_demo and jobs is not None,
                                              defer_modnet=settings.modnet_demo and jobs is not None)
                ctx.stats["prepare_finished_ms"] = time.monotonic() * 1000
            except Exception as exc:
                self.failed.emit(f"processing failed: {exc}")
                break
            fba_eligible = bool(ctx.landmarks is not None and ctx.face_mask is not None and ctx.face_mask.any())
            if jobs is not None:
                if settings.fba_demo and fba_eligible:
                    jobs.submit_preview(ctx, settings, generation, seq)
                elif settings.modnet_demo and not self.modnet_failed:
                    if ctx.stats.get("track_break", False):
                        track_epoch["value"] += 1
                        self.track_epoch = track_epoch["value"]
                        pending_modnet.clear()
                        if manual_capture_target["frame_id"] is not None:
                            clear_manual_capture_target()
                            self.model_failed.emit((
                                True, "modnet", None,
                                "Face track changed before the requested matte was ready",
                            ))
                    pending_modnet[seq] = (
                        ctx, settings, generation, track_epoch["value"], auto_generation)
                    while len(pending_modnet) > 4:
                        protected_id = manual_capture_target["frame_id"]
                        if manual_capture_requested and protected_id is None:
                            protected_id = seq
                        expired_id = next(
                            (frame_id for frame_id in pending_modnet
                             if frame_id != protected_id),
                            None,
                        )
                        if expired_id is None:
                            break
                        pending_modnet.pop(expired_id)
                        self.modnet_result_drops += 1
                        if manual_capture_target["frame_id"] == expired_id:
                            clear_manual_capture_target()
                            self.model_failed.emit((
                                True, "modnet", None,
                                "Requested MODNet frame expired before matte inference",
                            ))
                else:
                    jobs.clear_preview()

            if settings.modnet_demo and jobs is not None and not self.modnet_failed:
                if manual_capture_requested:
                    if manual_capture_target["frame_id"] is not None:
                        self.model_failed.emit((
                            True, "modnet", None,
                            "A manual capture is already waiting for a MODNet matte",
                        ))
                    elif (ctx.landmarks is not None and ctx.light_coeffs is not None
                            and ctx.stats.get("landmark_source") in ("native", "provided")
                            and not ctx.stats.get("track_break", False)):
                        manual_capture_target.update(
                            frame_id=seq,
                            generation=generation,
                            request_timestamp_ms=manual_capture_request_timestamp_ms,
                            read_timestamp_ms=ts,
                        )
                    else:
                        self.model_failed.emit((
                            True, "modnet", None,
                            "No stable native face track and lighting estimate for requested photo",
                        ))
            if settings.modnet_demo and jobs is not None:
                apply_ready_modnet()

            lock = getattr(source, "lock_state", "locked")
            if lock in ("locked", "partial") and ctx.face_mask is not None and ctx.face_mask.any():
                if monitor.update(float(ctx.Y[ctx.face_mask].mean()), ctx.landmarks, ts) == "drifting":
                    lock = "drifting"
            frames = config.SMOOTH_FRAMES_FALLBACK if lock in FALLBACK_LOCKS else config.SMOOTH_FRAMES
            if pipeline.relighter.history.maxlen != frames:
                pipeline.set_light_smoothing(frames)

            if (not (settings.modnet_demo and jobs is not None)
                    and (fired or manual_capture_requested)
                    and ctx.landmarks is not None and ctx.light_coeffs is not None):
                pipeline.shutter.armed = False
                if settings.fba_demo and jobs is not None:
                    if not submit_capture(
                        frame, ctx, settings, seq, generation, auto_generation,
                        automatic=bool(fired),
                    ):
                        model = "FBA"
                        self.model_failed.emit((True, model.lower(), None,
                                                f"{model} capture is already in progress"))
                else:
                    try:
                        result = pipeline.render_final(frame, ctx, settings)
                        result["_capture_display_guard"] = {
                            "generation": generation,
                            "auto_generation": auto_generation,
                            "automatic": bool(fired and not manual_capture_requested),
                            "track_epoch": track_epoch["value"],
                            "settings": _copy_settings(settings),
                            "model": "fba" if settings.fba_demo else "traditional",
                            "frame_id": seq,
                            "timestamp_ms": ts,
                        }
                        self.captured.emit(result)
                    except Exception as exc:  # keep the live view running
                        self.failed.emit(f"capture failed: {exc}")
            now = time.perf_counter()
            dt = max(now - last, 1e-3)
            fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
            last = now
            shown = (ctx.raw_frame if self.view_before and not settings.modnet_demo
                     else color.encode_u8(ctx.output_frame))
            if self.mirror:
                shown = shown[:, ::-1]
            self.frame_ready.emit({
                "image": to_qimage(shown),
                "size": ctx.size,
                "geometry": ctx.geometry,
                "verdicts": ctx.rule_verdicts,
                "shutter": (pipeline.shutter.count, pipeline.shutter.frames, pipeline.shutter.armed),
                "fps": fps,
                "lock": lock,
                "stats": {k: ctx.stats.get(k) for k in ("r_lr_before", "r_lr_after")},
                "face": ctx.landmarks is not None,
                "fba_eligible": fba_eligible,
                "modnet_eligible": settings.modnet_demo,
                "timestamp_ms": ctx.timestamp_ms,
                "source_frame_id": seq,
                "source": getattr(source, "name", ""),
                "fba_capture_busy": bool(jobs and jobs.capture_busy),
                "model_capture_busy": bool(jobs and jobs.capture_busy),
                "modnet_result_drops": self.modnet_result_drops,
                "subject_status": ctx.subject_status,
                "track_epoch": track_epoch["value"],
            })
        grabber.stop()
        if jobs is not None:
            jobs.stop()
        if reacquiring_detector is not None:
            reacquiring_detector.close()
        if detector is not None:
            detector.close()


# ---------------------------------------------------------------------------------------
# Live view


class LiveView(QWidget):
    modnet_frame_painted = Signal(int, float)

    def __init__(self):
        super().__init__()
        self.image = None
        self.info = None
        self.lang = "ko"
        self.before = False
        self.mirror = True
        self.message = None
        self.flash = 0.0
        self._last_modnet_painted_frame_id = None
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, info: dict):
        self.image, self.info = info["image"], info
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#ecebe8"))
        if self.image is None:
            p.setPen(QColor("#555"))
            flags = Qt.AlignmentFlag.AlignCenter.value | Qt.TextFlag.TextWordWrap.value
            p.drawText(QRectF(self.rect()).adjusted(60, 0, -60, 0), flags, self.message or t("lock_warming", self.lang))
            return
        iw, ih = self.image.width(), self.image.height()
        s = min(self.width() / iw, self.height() / ih)
        ox, oy = (self.width() - iw * s) / 2, (self.height() - ih * s) / 2
        p.drawImage(QRectF(ox, oy, iw * s, ih * s), self.image)

        def to_view(x, y):
            return QPointF(ox + ((iw - x) if self.mirror else x) * s, oy + y * s)

        before = self.before and not self.info.get("modnet_guarded")
        geo = self.info.get("geometry")
        if geo and not before:
            self._draw_guide(p, geo, to_view)
        if not self.info.get("face") and not self.info.get("modnet_guarded"):
            self._banner(p, t("no_face", self.lang) + " · " + t("msg_no_face", self.lang))
        if before:
            label = t("pill_before", self.lang)
        elif self.info.get("modnet_display"):
            age = max(0.0, time.monotonic() - self.info["modnet_timestamp_ms"] / 1000)
            label = t("pill_modnet", self.lang, age=age)
        elif self.info.get("modnet_waiting"):
            label = t("pill_modnet_waiting", self.lang)
        elif self.info.get("fba_display"):
            age = max(0.0, time.monotonic() - self.info["fba_timestamp_ms"] / 1000)
            label = t("pill_fba", self.lang, age=age)
        elif self.info.get("fba_waiting"):
            label = t("pill_clothing", self.lang)
        else:
            label = t("pill_relit", self.lang)
        self._pill(p, QPointF(18, 16), label, dark=not before)
        if not self.info.get("modnet_guarded"):
            self._pill(p, QPointF(self.width() - 18, 16), t("hold_b", self.lang), dark=False, right=True)
        if self.message:
            self._banner(p, self.message, bottom=True)
        if self.flash > 0:
            p.fillRect(self.rect(), QColor(255, 255, 255, int(220 * self.flash)))
        if self.info.get("modnet_display") and not before:
            frame_id = self.info.get("modnet_frame_id")
            if frame_id is not None and frame_id != self._last_modnet_painted_frame_id:
                self._last_modnet_painted_frame_id = frame_id
                self.modnet_frame_painted.emit(frame_id, time.monotonic() * 1000)

    def _draw_guide(self, p: QPainter, geo: dict, to_view):
        x0, y0, x1, y1 = geo["crop_box"]
        a, b = to_view(x0, y0), to_view(x1, y1)
        rect = QRectF(a, b).normalized()
        fits = geo.get("crop_fits", True)
        col = QColor(ACCENT if fits else "#e0712c")
        p.setPen(QPen(col, 2, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(rect)
        p.setPen(QPen(col, 1, Qt.PenStyle.DotLine))
        font = QFont(self.font())
        font.setPointSizeF(max(9.0, font.pointSizeF()))
        p.setFont(font)
        for key, y in (("crown", float(geo["crown"][1])), ("chin", float(geo["chin"][1]))):
            yv = to_view(x0, y).y()
            p.drawLine(QPointF(rect.left(), yv), QPointF(rect.right() + 14, yv))
            p.drawText(QPointF(rect.right() + 18, yv + 4), t(key, self.lang))
        fmt = self.info.get("format", "resume")
        label = t("crop_label", self.lang, fmt=t(f"fmt_{fmt}_long", self.lang), pct=self.info.get("head_pct", 75))
        metrics = p.fontMetrics()
        box = QRectF(rect.left(), rect.bottom() + 6, metrics.horizontalAdvance(label) + 14, metrics.height() + 6)
        p.fillRect(box, QColor(255, 255, 255, 225))
        p.setPen(col)
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, label)

    def _pill(self, p: QPainter, at: QPointF, text: str, dark: bool, right: bool = False):
        font = QFont(self.font())
        font.setBold(True)
        p.setFont(font)
        m = p.fontMetrics()
        w, h = m.horizontalAdvance(text) + 28, m.height() + 12
        rect = QRectF(at.x() - w if right else at.x(), at.y(), w, h)
        p.setPen(QPen(QColor("#111"), 1.2))
        p.setBrush(QColor("#111") if dark else QColor(255, 255, 255, 235))
        p.drawRoundedRect(rect, h / 2, h / 2)
        p.setPen(QColor("white") if dark else QColor("#111"))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        p.setBrush(Qt.BrushStyle.NoBrush)

    def _banner(self, p: QPainter, text: str, bottom: bool = False):
        m = p.fontMetrics()
        w, h = m.horizontalAdvance(text) + 32, m.height() + 16
        y = self.height() - h - 20 if bottom else self.height() / 2 - h / 2
        rect = QRectF((self.width() - w) / 2, y, w, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRoundedRect(rect, 8, 8)
        p.setPen(QColor("white"))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


# ---------------------------------------------------------------------------------------
# Result dialog


class ResultDialog(QDialog):
    def __init__(self, parent, result: dict, lang: str):
        super().__init__(parent)
        self.result, self.lang = result, lang
        self.saved_dir = None
        self.setWindowTitle(t("result_title", lang))
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        for key in ("resume", "id", "profile"):
            photo = result["photos"][key]
            col = QVBoxLayout()
            pic = QLabel()
            pix = QPixmap.fromImage(to_qimage(photo))
            pic.setPixmap(pix.scaledToHeight(260, Qt.TransformationMode.SmoothTransformation))
            pic.setStyleSheet("border: 1px solid #ccc; background: white;")
            col.addWidget(pic, alignment=Qt.AlignmentFlag.AlignHCenter)
            fmt = geometry.FORMATS[key]
            cap = QLabel(f"{t('fmt_' + key, lang)} · {fmt.size_px[0]}×{fmt.size_px[1]} px")
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(cap)
            row.addLayout(col)
        layout.addLayout(row)
        notice = QLabel(t("notice", lang))
        notice.setWordWrap(True)
        notice.setStyleSheet("border: 1px dashed #999; border-radius: 6px; padding: 8px; color: #333;")
        layout.addWidget(notice)
        note = QLabel(t("sheet_note", lang))
        note.setWordWrap(True)
        layout.addWidget(note)
        if result.get("kept_wall"):
            kept = QLabel(t("kept_wall", lang))
            kept.setWordWrap(True)
            kept.setStyleSheet("color: #a5501e;")
            layout.addWidget(kept)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItems(["JPG", "PNG"])
        save = QPushButton(t("save", lang))
        save.setDefault(True)
        save.clicked.connect(self.save)
        self.open_btn = QPushButton(t("open_folder", lang))
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.saved_dir))))
        retake = QPushButton(t("retake", lang))
        retake.clicked.connect(self.reject)
        buttons.addWidget(self.kind)
        buttons.addStretch(1)
        buttons.addWidget(self.open_btn)
        buttons.addWidget(retake)
        buttons.addWidget(save)
        layout.addLayout(buttons)

    def save(self):
        start = Path.home() / "Pictures"
        folder = QFileDialog.getExistingDirectory(self, t("choose_folder", self.lang), str(start if start.exists() else Path.home()))
        if folder:
            self.saved_dir, files = geometry.save_outputs(self.result, Path(folder), self.kind.currentText().lower())
            self.status.setText(t("saved_to", self.lang, n=len(files), path=str(self.saved_dir)))
            self.open_btn.setEnabled(True)


# ---------------------------------------------------------------------------------------
# Main window


class KeyFilter(QObject):
    """Hold B for the unprocessed frame, Space for a photo, wherever the focus is."""

    def __init__(self, window):
        super().__init__()
        self.window = window

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and QApplication.activeModalWidget() is None:
            if event.key() == Qt.Key.Key_B and not event.isAutoRepeat():
                self.window.set_before(kind == QEvent.Type.KeyPress)
                return True
            if event.key() == Qt.Key.Key_Space and kind == QEvent.Type.KeyPress and not event.isAutoRepeat():
                self.window.take_photo()
                return True
        return False


class MainWindow(QMainWindow):
    capture_displayed = Signal(object, object)

    def __init__(self, source, lang: str, camera_size, presentation_fba=False, fba_device="cpu",
                 presentation_modnet=False, modnet_model="models/modnet.onnx",
                 modnet_provider="cpu", birefnet_model=None, face_model=None, start_fba=False):
        super().__init__()
        self.lang = lang
        self.settings = Settings()
        self.settings_lock = threading.RLock()
        self.fba_factory = None
        self.fba_error = None
        self.modnet_factory = None
        self.modnet_error = None
        self.birefnet_factory = None
        self.birefnet_error = None
        self.birefnet_runtime_error = None
        self.presentation_fba = presentation_fba
        self.modnet_opted_in = presentation_modnet
        self.face_model_path = Path(face_model).expanduser() if face_model else None
        self.model_generation = 0
        self.auto_shutter_generation = 0
        self.fba_generation = 0
        self.fba_last = None
        self.fba_valid_after_ms = -1
        self.live_info = None
        self.fba_runtime_error = None
        self.fba_preview_failed = False
        self.modnet_generation = 0
        self.modnet_last = None
        self.modnet_valid_after_ms = -1
        self.modnet_runtime_error = None
        self.modnet_preview_failed = False
        self.worker = None
        if presentation_fba:
            try:
                from tools.presentation_fba import PresentationFBA, check_availability

                if fba_device == "cpu":
                    check_availability()
                    self.fba_factory = PresentationFBA
                else:
                    check_availability(fba_device)
                    self.fba_factory = functools.partial(PresentationFBA, device=fba_device)
            except (ImportError, OSError, ValueError, subprocess.SubprocessError) as exc:
                self.fba_error = str(exc)
        if start_fba and self.fba_factory is not None:
            self.settings.fba_demo = True
        if presentation_modnet:
            try:
                from .presentation_modnet import PresentationMODNet, check_availability

                model_path = check_availability(modnet_model, modnet_provider)
                self.modnet_factory = functools.partial(
                    PresentationMODNet, model_path, provider=modnet_provider
                )
            except (ImportError, OSError, ValueError, subprocess.SubprocessError) as exc:
                self.modnet_error = str(exc)
            if birefnet_model:
                try:
                    from .presentation_birefnet import (
                        BirefNetPortrait,
                        check_availability,
                    )

                    model_path = check_availability(birefnet_model, verify_hash=False)
                    self.birefnet_factory = functools.partial(BirefNetPortrait, model_path)
                except (ImportError, OSError, ValueError, subprocess.SubprocessError) as exc:
                    self.birefnet_error = str(exc)
            else:
                self.birefnet_error = (
                    "Provide --birefnet-model with a verified local BiRefNet portrait ONNX file."
                )
            if self.birefnet_factory is not None and self.modnet_factory is None:
                self.birefnet_factory = None
                self.birefnet_error = (
                    "BiRefNet portrait capture requires an available MODNet preview: "
                    + (self.modnet_error or "MODNet is unavailable.")
                )
        else:
            self.modnet_error = t("modnet_enable_hint", lang)
            self.birefnet_error = t("modnet_enable_hint", lang)
        self.dialog = None
        self.closing = False
        self.view = LiveView()
        self.view.lang = lang
        self._build_panel()
        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.scroll)
        self.setCentralWidget(central)
        self.resize(1280, 800)

        self.source_spec, self.camera_size = source, camera_size
        self.open_attempts = 0
        self.open_timer = QTimer(self)
        self.open_timer.setSingleShot(True)
        self.open_timer.timeout.connect(self.open_source)
        self.open_timer.start(0)  # once the event loop runs
        self.flash_timer = QTimer(self)
        self.flash_timer.timeout.connect(self._fade_flash)
        self.retranslate()

    # -- source --------------------------------------------------------------------------
    def open_source(self):
        """Open the frame source on the main thread, then start processing.

        macOS asks for camera access only from the main thread. The first attempt shows the
        request and fails, so we poll the permission until the user answers it.
        """
        self.open_attempts += 1
        camera = is_camera(self.source_spec)
        permission = camera_permission() if camera else "unknown"
        if permission in ("denied", "restricted"):
            return self._show_message(t("camera_denied", self.lang))
        if permission == "not_determined" and self.open_attempts > 1:
            return self._wait_for_camera()  # the request is on screen; don't ask again
        try:
            source = open_source(self.source_spec, self.camera_size)
        except Exception as exc:
            if camera and camera_permission() == "not_determined":
                return self._wait_for_camera()
            if camera and permission == "unknown" and self.open_attempts < 10:
                return self._wait_for_camera()  # permission state unreadable: retry a few times
            return self.on_failed(str(exc))
        self.view.message = None
        self.worker = Worker(
            source, self.settings, self.fba_factory, self.modnet_factory,
            self.birefnet_factory, face_model_path=self.face_model_path,
            settings_lock=self.settings_lock)
        self.worker.model_generation = self.model_generation
        self.worker.auto_shutter_generation = self.auto_shutter_generation
        self.worker.mirror = self.view.mirror
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.captured.connect(self.on_captured)
        self.worker.failed.connect(self.on_failed)
        self.worker.model_snapshot.connect(self.on_model_snapshot)
        self.worker.modnet_preparing.connect(self.on_modnet_preparing)
        self.worker.model_failed.connect(self.on_model_failed)
        self.worker.subject_abstained.connect(self.on_subject_abstained)
        self.worker.start()

    def _wait_for_camera(self):
        self._show_message(t("camera_allow", self.lang))
        self.open_timer.start(1000)

    def _show_message(self, text: str):
        self.view.message = text
        self.view.update()

    # -- panel ---------------------------------------------------------------------------
    def _section(self, layout):
        label = QLabel()
        label.setObjectName("section")
        layout.addWidget(label)
        return label

    def _build_panel(self):
        panel = QWidget()
        panel.setObjectName("panel")
        v = QVBoxLayout(panel)
        v.setContentsMargins(20, 12, 20, 12)
        v.setSpacing(6)

        head = QHBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("title")
        self.lang_btn = QPushButton()
        self.lang_btn.setObjectName("lang")
        self.lang_btn.clicked.connect(self.toggle_language)
        head.addWidget(self.title, 1)
        head.addWidget(self.lang_btn)
        v.addLayout(head)

        self.format_label = self._section(v)
        seg = QHBoxLayout()
        seg.setSpacing(0)
        self.format_group = QButtonGroup(self)
        self.format_buttons = {}
        for i, key in enumerate(geometry.FORMATS):
            b = QPushButton()
            b.setCheckable(True)
            b.setObjectName("segment")
            b.setProperty("pos", "first" if i == 0 else ("last" if i == 2 else "mid"))
            b.clicked.connect(lambda _=False, k=key: self.set_format(k))
            self.format_group.addButton(b)
            self.format_buttons[key] = b
            seg.addWidget(b)
        self.format_buttons["resume"].setChecked(True)
        v.addLayout(seg)

        self.head_label = self._section(v)
        self.head_slider = QSlider(Qt.Orientation.Horizontal)
        self.head_slider.setRange(40, 90)
        self.head_slider.valueChanged.connect(self.set_head_share)
        self.head_value = QLabel()
        hs = QHBoxLayout()
        hs.addWidget(self.head_slider, 1)
        hs.addWidget(self.head_value)
        v.addLayout(hs)

        self.bg_label = self._section(v)
        sw = QHBoxLayout()
        sw.setSpacing(8)
        self.bg_group = QButtonGroup(self)
        self.bg_buttons = {}
        for key, hex_color in BACKGROUNDS.items():
            b = QPushButton()
            b.setCheckable(True)
            b.setFixedSize(38, 38)
            fill = hex_color or "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f2f2f2, stop:0.49 #f2f2f2, stop:0.5 #cfcfcf, stop:1 #cfcfcf)"
            b.setStyleSheet(
                f"QPushButton {{ background: {fill}; border: 1px solid #b8b8b8; border-radius: 8px; }}"
                f"QPushButton:checked {{ border: 3px solid {ACCENT}; }}")
            b.clicked.connect(lambda _=False, k=key: self.set_background(k))
            self.bg_group.addButton(b)
            self.bg_buttons[key] = b
            sw.addWidget(b)
        self.bg_name = QLabel()
        self.bg_name.setObjectName("muted")
        sw.addWidget(self.bg_name, 1)
        self.bg_buttons["white"].setChecked(True)
        v.addLayout(sw)

        self.model_label = self._section(v)
        self.clothing_box = QCheckBox()
        self.clothing_box.setChecked(self.settings.clothing_enabled)
        self.clothing_box.toggled.connect(self.set_clothing)
        v.addWidget(self.clothing_box)
        self.fba_box = None
        if self.presentation_fba:
            self.fba_box = QCheckBox()
            self.fba_box.setEnabled(self.fba_factory is not None)
            self.fba_box.setChecked(self.settings.fba_demo)
            self.fba_box.toggled.connect(self.set_fba)
            v.addWidget(self.fba_box)
            self.fba_status = QLabel()
            self.fba_status.setWordWrap(True)
            v.addWidget(self.fba_status)
            self.fba_preview_label = QLabel()
            v.addWidget(self.fba_preview_label)
            self.fba_image = QLabel()
            self.fba_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.addWidget(self.fba_image)
            if self.fba_error:
                self.fba_warning = QLabel(t("fba_unavailable", self.lang, err=self.fba_error))
                self.fba_warning.setWordWrap(True)
                v.addWidget(self.fba_warning)
        self.modnet_box = None
        self.modnet_preview_label = None
        self.birefnet_box = None
        self.birefnet_status = None
        self.modnet_box = QCheckBox()
        self.modnet_box.setEnabled(self.modnet_factory is not None)
        self.modnet_box.toggled.connect(self.set_modnet)
        v.addWidget(self.modnet_box)
        self.modnet_status = QLabel()
        self.modnet_status.setWordWrap(True)
        v.addWidget(self.modnet_status)
        self.modnet_preview_label = QLabel()
        v.addWidget(self.modnet_preview_label)
        self.modnet_image = QLabel()
        self.modnet_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.modnet_image)
        if self.modnet_error:
            self.modnet_warning = QLabel(self.modnet_error)
            self.modnet_warning.setWordWrap(True)
            v.addWidget(self.modnet_warning)
        if self.modnet_opted_in:
            self.birefnet_box = QCheckBox()
            self.birefnet_box.setEnabled(
                self.birefnet_factory is not None and self.modnet_factory is not None
            )
            self.birefnet_box.setToolTip(
                self.birefnet_error or t("birefnet_capture_tip", self.lang)
            )
            self.birefnet_box.toggled.connect(self.set_birefnet_capture)
            v.addWidget(self.birefnet_box)
            self.birefnet_status = QLabel()
            self.birefnet_status.setWordWrap(True)
            v.addWidget(self.birefnet_status)

        self.strength_label = self._section(v)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(int(round(self.settings.strength * 100)))
        self.strength_slider.valueChanged.connect(self.set_strength)
        self.strength_value = QLabel()
        ss = QHBoxLayout()
        ss.addWidget(self.strength_slider, 1)
        ss.addWidget(self.strength_value)
        v.addLayout(ss)

        self.checks_label = self._section(v)
        box = QFrame()
        box.setObjectName("checks")
        grid = QGridLayout(box)
        grid.setContentsMargins(14, 8, 14, 8)
        grid.setVerticalSpacing(6)
        self.rule_names, self.rule_marks = {}, {}
        for n in range(1, 8):
            name, mark = QLabel(), QLabel("·")
            mark.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(name, n, 0)
            grid.addWidget(mark, n, 1)
            self.rule_names[n], self.rule_marks[n] = name, mark
        v.addWidget(box)

        self.coach = QLabel()
        self.coach.setObjectName("coach")
        self.coach.setWordWrap(True)
        v.addWidget(self.coach)

        self.shutter_label = self._section(v)
        self.shutter_text = QLabel()
        v.addWidget(self.shutter_text)
        self.progress = QProgressBar()
        self.progress.setRange(0, config.SHUTTER_FRAMES)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(10)
        v.addWidget(self.progress)
        row = QHBoxLayout()
        self.auto_box = QCheckBox()
        self.auto_box.setChecked(True)
        self.auto_box.toggled.connect(self.set_auto)
        self.mirror_box = QCheckBox()
        self.mirror_box.setChecked(True)
        self.mirror_box.toggled.connect(self.set_mirror)
        self.shoot = QPushButton()
        self.shoot.setObjectName("shoot")
        self.shoot.clicked.connect(self.take_photo)
        row.addWidget(self.auto_box)
        row.addWidget(self.mirror_box)
        row.addStretch(1)
        row.addWidget(self.shoot)
        v.addLayout(row)

        self.notice = QLabel()
        self.notice.setObjectName("notice")
        self.notice.setWordWrap(True)
        v.addWidget(self.notice)
        self.status = QLabel()
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        v.addStretch(1)

        for w in panel.findChildren(QPushButton) + panel.findChildren(QSlider) + panel.findChildren(QCheckBox):
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        panel.setStyleSheet(PANEL_STYLE)
        self.scroll = QScrollArea()
        self.scroll.setWidget(panel)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedWidth(400)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)

    # -- settings ------------------------------------------------------------------------
    def set_format(self, key: str):
        with self.settings_lock:
            changed = self.settings.format != key
            self.settings.format = key
            if changed:
                self._advance_model_generation()
        self.head_slider.blockSignals(True)
        self.head_slider.setValue(int(round(self.settings.head_share[key] * 100)))
        self.head_slider.blockSignals(False)
        self._update_head_label()
        if changed:
            self._clear_active_model_snapshots()
        self.update_view()

    def set_head_share(self, value: int):
        with self.settings_lock:
            fmt = self.settings.format
            share = value / 100
            changed = self.settings.head_share[fmt] != share
            self.settings.head_share[fmt] = share
            if changed:
                self._advance_model_generation()
        self._update_head_label()
        if changed:
            self._clear_active_model_snapshots()
        self.update_view()

    def _update_head_label(self):
        pct = int(round(self.settings.head_share[self.settings.format] * 100))
        self.head_value.setText(f"{pct}%")

    def _advance_model_generation(self):
        self.model_generation += 1
        if self.settings.fba_demo:
            self.fba_generation = self.model_generation
        if self.settings.modnet_demo:
            self.modnet_generation = self.model_generation
        if self.worker:
            self.worker.model_generation = self.model_generation

    def _clear_active_model_snapshots(self):
        if self.settings.fba_demo:
            self.clear_fba_snapshot()
        if self.settings.modnet_demo:
            self.clear_modnet_snapshot()

    def set_background(self, key: str):
        with self.settings_lock:
            changed = self.settings.background != key
            if changed:
                self.settings.background = key
                self._advance_model_generation()
        if changed:
            self.clear_fba_snapshot()
            self.clear_modnet_snapshot()
        self.bg_name.setText(t(f"bg_{key}", self.lang))

    def set_strength(self, value: int):
        with self.settings_lock:
            strength = value / 100
            changed = self.settings.strength != strength
            if changed:
                self.settings.strength = strength
                self._advance_model_generation()
        if changed:
            self.clear_fba_snapshot()
            self.clear_modnet_snapshot()

    def clear_fba_snapshot(self):
        self.fba_valid_after_ms = max(self.fba_valid_after_ms, int(time.monotonic() * 1000))
        self.fba_last = None
        self.update_fba_status()
        self.update_view()

    def set_auto(self, on: bool):
        with self.settings_lock:
            changed = self.settings.auto_shutter != on
            self.settings.auto_shutter = on
            if changed:
                self.auto_shutter_generation += 1
                if self.worker:
                    self.worker.auto_shutter_generation = self.auto_shutter_generation
        if on and self.worker:
            self.worker.rearm_request = True
        if changed and self.settings.modnet_demo:
            self.clear_modnet_snapshot()

    def set_clothing(self, on: bool):
        if not on and self.fba_box is not None:
            self.fba_box.setChecked(False)
        with self.settings_lock:
            changed = self.settings.clothing_enabled != on
            self.settings.clothing_enabled = on
            if changed and not self.settings.modnet_demo:
                self._advance_model_generation()
        if changed and not self.settings.modnet_demo:
            self._clear_active_model_snapshots()
        if self.fba_box is not None:
            self.fba_box.setEnabled(on and self.fba_factory is not None and not self.fba_preview_failed)
        if self.modnet_preview_label is not None:
            label = ("modnet_output_preview" if self.settings.modnet_demo
                     else "modnet_clothing_preview" if on else "modnet_main_preview")
            self.modnet_preview_label.setText(t(label, self.lang))

    def set_fba(self, on: bool):
        if on and self.modnet_box is not None and self.modnet_box.isChecked():
            self.modnet_box.setChecked(False)
        with self.settings_lock:
            changed = self.settings.fba_demo != on
            self.settings.fba_demo = on
            if changed:
                self._advance_model_generation()
            self.fba_generation = self.model_generation
        self.fba_valid_after_ms = max(self.fba_valid_after_ms, int(time.monotonic() * 1000))
        self.fba_last = None
        self.fba_runtime_error = None
        self.fba_image.clear()
        self.fba_preview_label.setVisible(on)
        self.fba_image.setVisible(on)
        self.update_fba_status()
        self.update_view()

    def clear_modnet_snapshot(self):
        self.modnet_valid_after_ms = max(self.modnet_valid_after_ms, int(time.monotonic() * 1000))
        self.modnet_last = None
        self.update_modnet_status()
        self.update_view()

    def set_modnet(self, on: bool):
        if on and self.fba_box is not None and self.fba_box.isChecked():
            self.fba_box.setChecked(False)
        if not on and self.birefnet_box is not None and self.birefnet_box.isChecked():
            self.birefnet_box.setChecked(False)
        with self.settings_lock:
            changed = self.settings.modnet_demo != on
            self.settings.modnet_demo = on
            if changed:
                self._advance_model_generation()
            self.modnet_generation = self.model_generation
        self.modnet_valid_after_ms = max(self.modnet_valid_after_ms, int(time.monotonic() * 1000))
        self.modnet_last = None
        self.modnet_runtime_error = None
        self.shoot.setEnabled(False)
        if not on and self.modnet_preview_failed and self.modnet_box is not None:
            self.modnet_box.setEnabled(False)
        self.modnet_image.clear()
        preview_label = ("modnet_output_preview" if on else
                         "modnet_clothing_preview" if self.settings.clothing_enabled
                         else "modnet_main_preview")
        self.modnet_preview_label.setText(t(preview_label, self.lang))
        self.modnet_preview_label.setVisible(on)
        self.modnet_image.setVisible(on)
        self.update_modnet_status()
        self.update_view()

    def set_birefnet_capture(self, on: bool):
        with self.settings_lock:
            capture_matte = "birefnet-portrait" if on else "modnet"
            changed = self.settings.capture_matte != capture_matte
            self.settings.capture_matte = capture_matte
            if changed:
                self._advance_model_generation()
        self.birefnet_runtime_error = None
        if changed:
            self.clear_modnet_snapshot()
            if self.worker:
                self.worker.rearm_request = True
        self.update_birefnet_status()
        self.update_view()

    def update_modnet_status(self, busy=False):
        if self.modnet_box is None:
            return
        if self.modnet_runtime_error:
            self.modnet_status.setText(t("modnet_error", self.lang, err=self.modnet_runtime_error))
        elif not self.settings.modnet_demo:
            self.modnet_status.setText("")
        elif busy and self.settings.capture_matte == "modnet":
            self.modnet_status.setText(t("modnet_capturing", self.lang))
        elif self.modnet_last is None:
            self.modnet_status.setText(t("modnet_waiting", self.lang))
        else:
            age = max(0.0, time.monotonic() - self.modnet_last["timestamp_ms"] / 1000)
            self.modnet_status.setText(t(
                "modnet_snapshot_age", self.lang, age=age,
                inference_ms=self.modnet_last.get("inference_ms") or 0,
                frame_id=self.modnet_last.get("frame_id", -1),
                provider=self.modnet_last.get("execution_provider", "unknown")))

    def update_birefnet_status(self, busy=False):
        if self.birefnet_status is None:
            return
        if self.birefnet_runtime_error:
            text = t("birefnet_capture_error", self.lang, err=self.birefnet_runtime_error)
        elif self.birefnet_error:
            text = t("birefnet_capture_unavailable", self.lang, err=self.birefnet_error)
        elif self.settings.capture_matte == "birefnet-portrait" and busy:
            text = t("birefnet_capture_busy", self.lang)
        elif self.settings.capture_matte == "birefnet-portrait":
            text = t("birefnet_capture_selected", self.lang)
        else:
            text = t("birefnet_capture_off", self.lang)
        self.birefnet_status.setText(text)

    def on_modnet_snapshot(self, snapshot):
        if (not self.settings.modnet_demo or self.modnet_preview_failed
                or snapshot["generation"] != self.modnet_generation
                or snapshot.get("auto_shutter_generation") != self.auto_shutter_generation
                or snapshot["background"] != self.settings.background
                or snapshot["strength"] != self.settings.strength
                or (self.live_info is not None
                    and snapshot.get("track_epoch") is not None
                    and snapshot["track_epoch"] != self.live_info.get("track_epoch"))
                or snapshot["timestamp_ms"] <= self.modnet_valid_after_ms
                or (self.modnet_last is not None
                    and snapshot["timestamp_ms"] <= self.modnet_last["timestamp_ms"])):
            return
        self.modnet_last = snapshot
        self.update_view()
        if self.live_info is not None:
            self.on_frame(self.live_info)
        self.update_modnet_status()

    def on_subject_abstained(self, status):
        if not self.settings.modnet_demo:
            return
        self.modnet_last = None
        if self.live_info is not None:
            self.live_info = dict(self.live_info, subject_status=status)
            self.on_frame(self.live_info)
        else:
            self.update_view()

    def on_modnet_preparing(self, generation):
        if self.settings.modnet_demo and generation == self.modnet_generation:
            self.modnet_last = None
            self.update_view()
            self.update_modnet_status()

    def on_modnet_failed(self, error):
        capture, generation, message = error
        if not capture and (not self.settings.modnet_demo or generation != self.modnet_generation):
            return
        self.modnet_runtime_error = ("capture: " if capture else "preview: ") + message
        if not capture and self.modnet_box is not None:
            self.modnet_preview_failed = True
            self.modnet_last = None
            self.shoot.setEnabled(False)
            if self.worker:
                self.worker.rearm_request = True
            self.modnet_runtime_error = "preview: " + message
        elif capture and self.worker:
            self.worker.rearm_request = True
        self.update_modnet_status()
        self.update_view()

    def on_model_snapshot(self, snapshot):
        if snapshot.get("model") == "modnet":
            self.on_modnet_snapshot(snapshot)
        else:
            self.on_fba_snapshot(snapshot)

    def on_model_failed(self, error):
        capture, model, generation, message = error
        if model == "birefnet":
            self.birefnet_runtime_error = message
            if capture and self.worker:
                self.worker.rearm_request = True
            self.update_birefnet_status()
        elif model == "modnet":
            self.on_modnet_failed((capture, generation, message))
        elif model == "traditional":
            if self.worker:
                self.worker.rearm_request = True
            self._show_message(message)
        else:
            self.on_fba_failed((capture, generation, message))

    def update_fba_status(self, busy=False):
        if self.fba_box is None:
            return
        if self.fba_runtime_error:
            self.fba_status.setText(t("fba_error", self.lang, err=self.fba_runtime_error))
        elif not self.settings.fba_demo:
            self.fba_status.setText("")
        elif busy:
            self.fba_status.setText(t("fba_capturing", self.lang))
        elif self.fba_last is None:
            self.fba_status.setText(t("fba_waiting", self.lang))
        else:
            age = max(0.0, time.monotonic() - self.fba_last["timestamp_ms"] / 1000)
            self.fba_status.setText(t("fba_snapshot_age", self.lang, age=age))

    def on_fba_snapshot(self, snapshot):
        if (not self.settings.fba_demo or snapshot["generation"] != self.fba_generation
                or snapshot["background"] != self.settings.background
                or snapshot["strength"] != self.settings.strength
                or snapshot["timestamp_ms"] <= self.fba_valid_after_ms
                or (self.fba_last is not None
                    and snapshot["timestamp_ms"] <= self.fba_last["timestamp_ms"])):
            return
        self.fba_last = snapshot
        self.update_view()
        self.update_fba_status()

    def on_fba_failed(self, error):
        capture, generation, message = error
        if not capture and (not self.settings.fba_demo or generation != self.fba_generation):
            return
        self.fba_runtime_error = ("capture: " if capture else "preview: ") + message
        if not capture and self.fba_box is not None:
            self.fba_preview_failed = True
            self.fba_box.setChecked(False)
            self.fba_box.setEnabled(False)
            self.fba_runtime_error = "preview: " + message
        elif capture and self.worker:
            self.worker.rearm_request = True
        self.update_fba_status()

    def set_mirror(self, on: bool):
        if self.worker:
            self.worker.mirror = on
        self.view.mirror = on
        self.update_view()

    def set_before(self, on: bool):
        if self.worker:
            self.worker.view_before = on
        self.view.before = on
        self.update_view()

    def take_photo(self):
        if self.dialog is None and self.worker and self.shoot.isEnabled():
            self.worker.request_manual_capture()

    def toggle_language(self):
        self.lang = "en" if self.lang == "ko" else "ko"
        self.view.lang = self.lang
        self.retranslate()

    def retranslate(self):
        L = self.lang
        if not self.modnet_opted_in:
            self.modnet_error = t("modnet_enable_hint", L)
            self.birefnet_error = self.modnet_error
            self.modnet_warning.setText(self.modnet_error)
        self.setWindowTitle(t("title", L))
        self.title.setText(t("title", L))
        self.lang_btn.setText(t("language", L))
        self.format_label.setText(t("format", L))
        for key, b in self.format_buttons.items():
            b.setText(t(f"fmt_{key}", L))
        self.head_label.setText(t("head_size", L))
        self.bg_label.setText(t("background", L))
        self.bg_name.setText(t(f"bg_{self.settings.background}", L))
        for key, b in self.bg_buttons.items():
            b.setToolTip(t(f"bg_{key}", L))
        self.model_label.setText(t("matting_model", L))
        self.clothing_box.setText(t("clothing_toggle", L))
        self.clothing_box.setToolTip(t("clothing_toggle_tip", L))
        if self.fba_box is not None:
            self.fba_box.setText(t("fba_demo", L))
            self.fba_box.setToolTip(self.fba_error or t("fba_demo_tip", L))
            if self.fba_error:
                self.fba_warning.setText(t("fba_unavailable", L, err=self.fba_error))
            self.fba_preview_label.setText(t("fba_clothing_preview", L))
            self.fba_preview_label.setVisible(self.settings.fba_demo)
            self.fba_image.setVisible(self.settings.fba_demo)
            self.update_fba_status()
        if self.modnet_box is not None:
            self.modnet_box.setText(t("modnet_demo", L))
            self.modnet_box.setToolTip(self.modnet_error or t("modnet_demo_tip", L))
            label = ("modnet_output_preview" if self.settings.modnet_demo else
                     "modnet_clothing_preview" if self.settings.clothing_enabled
                     else "modnet_main_preview")
            self.modnet_preview_label.setText(t(label, L))
            self.modnet_preview_label.setVisible(self.settings.modnet_demo)
            self.modnet_image.setVisible(self.settings.modnet_demo)
            self.update_modnet_status()
        if self.birefnet_box is not None:
            self.birefnet_box.setText(t("birefnet_capture", L))
            self.birefnet_box.setToolTip(
                self.birefnet_error or t("birefnet_capture_tip", L)
            )
            self.update_birefnet_status()
        self.strength_label.setText(t("strength", L))
        self.checks_label.setText(t("checks", L))
        self.shutter_label.setText(t("auto_shutter", L))
        self.auto_box.setText(t("auto", L))
        self.mirror_box.setText(t("mirror", L))
        self.shoot.setText(t("take_photo", L))
        self.notice.setText(t("notice", L))
        self.set_format(self.settings.format)
        self.view.update()

    # -- frames --------------------------------------------------------------------------
    def update_view(self):
        if self.live_info is None:
            self.view.update()
            return
        info = dict(self.live_info)
        info["format"] = self.settings.format
        info["head_pct"] = int(round(self.settings.head_share[self.settings.format] * 100))
        snapshot = None
        model = None
        if self.settings.modnet_demo:
            snapshot, model = self.modnet_last, "modnet"
            info["modnet_guarded"] = True
            width, height = info["size"]
            placeholder = QImage(width, height, QImage.Format.Format_RGB888)
            placeholder.fill(QColor(BACKGROUNDS.get(self.settings.background) or "#ffffff"))
            info["image"] = placeholder
            info["geometry"] = None
            info["face"] = False
        elif not self.view.before:
            if self.settings.fba_demo:
                snapshot, model = self.fba_last, "fba"
        if snapshot is not None:
            image = snapshot["image"]
            info["image"] = image.flipped(Qt.Orientation.Horizontal) if self.view.mirror else image
            geo = dict(snapshot["geometry"]) if snapshot["geometry"] is not None else None
            if geo is not None:
                geo["crop_box"] = geometry.crop_box(
                    geo, geometry.FORMATS[self.settings.format],
                    self.settings.head_share[self.settings.format])
                geo["crop_fits"] = geometry.crop_fits(geo["crop_box"], (image.width(), image.height()))
            info["geometry"] = geo
            info["face"] = geo is not None
            info[f"{model}_display"] = True
            info[f"{model}_timestamp_ms"] = snapshot["timestamp_ms"]
            if model == "modnet":
                info["modnet_frame_id"] = snapshot["frame_id"]
                info["subject_status"] = snapshot.get("subject_status", "ready")
        elif ((self.settings.fba_demo or self.settings.modnet_demo) and not self.view.before):
            info["fba_waiting" if self.settings.fba_demo else "modnet_waiting"] = True
        self.view.message = None
        if self.settings.modnet_demo and self.modnet_preview_failed:
            self.view.message = t("modnet_error", self.lang, err=self.modnet_runtime_error)
        elif self.settings.modnet_demo and info.get("subject_status") in ("no_face", "ambiguous", "changed"):
            self.view.message = t("subject_" + info["subject_status"], self.lang)
        geo = info.get("geometry")
        if geo is not None and not geo.get("crop_fits", True):
            self.view.message = t("msg_move_back", self.lang)
        self.view.set_frame(info)

    def on_frame(self, info: dict):
        L = self.lang
        if (self.settings.modnet_demo and self.modnet_last is not None
                and self.modnet_last.get("track_epoch") is not None
                and self.modnet_last["track_epoch"] != info.get("track_epoch")):
            self.modnet_last = None
        if self.settings.modnet_demo and self.modnet_last is not None:
            info = dict(info)
            info["verdicts"] = self.modnet_last["verdicts"]
            info["shutter"] = self.modnet_last["shutter"]
            info["stats"] = self.modnet_last["stats"]
        self.live_info = info
        if self.settings.fba_demo and not info["fba_eligible"]:
            self.fba_valid_after_ms = max(self.fba_valid_after_ms, info["timestamp_ms"])
            self.fba_last = None
        if self.fba_last is not None and (self.fba_last["background"] != self.settings.background
                                         or self.fba_last["strength"] != self.settings.strength):
            self.clear_fba_snapshot()
        if self.modnet_last is not None and (self.modnet_last["background"] != self.settings.background
                                             or self.modnet_last["strength"] != self.settings.strength):
            self.clear_modnet_snapshot()
        if self.settings.fba_demo:
            if self.view.before:
                self.fba_image.clear()
            else:
                self.fba_image.setPixmap(QPixmap.fromImage(info["image"]).scaled(
                    220, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        if self.settings.modnet_demo:
            if self.view.before:
                self.modnet_image.clear()
            else:
                self.modnet_image.setPixmap(QPixmap.fromImage(info["image"]).scaled(
                    220, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.update_view()
        self.shoot.setEnabled(
            not self.settings.modnet_demo
            or (self.settings.modnet_demo and not self.modnet_preview_failed and self.modnet_last is not None
                and self.modnet_last.get("subject_status", "ready") == "ready"))
        self.strength_value.setText(f"{self.strength_slider.value()}%")

        verdicts = info.get("verdicts")
        verdicts = verdicts or {}
        first_fail = None
        checks_complete = True
        for n in range(1, 8):
            v = verdicts.get(n)
            if v is None:
                checks_complete = False
                name = t("rule_pending", L)
                self.rule_names[n].setText(f"{n} · {name}")
                self.rule_marks[n].setText("…")
                self.rule_marks[n].setStyleSheet("color: #777; font-weight: 600;")
                continue
            name = t(f"rule_{v.key}", L)
            if v and n == 1 and v.value is not None and math.isfinite(v.value):
                name += f" (R_LR {v.value:.2f})"
            self.rule_names[n].setText(f"{n} · {name}")
            ok = bool(v.passed)
            self.rule_marks[n].setText("✓" if ok else "✕")
            self.rule_marks[n].setStyleSheet(f"color: {'#2a7a3b' if ok else '#c0392b'}; font-weight: 600;")
            if not v.passed and first_fail is None:
                first_fail = v
        if first_fail is not None:
            name = t(f"rule_{first_fail.key}", L)
            self.coach.setText(f"<b>{name}:</b> {t('msg_' + (first_fail.message or 'no_face'), L)}")
            self.coach.setProperty("ok", False)
            self.coach.setProperty("pending", False)
        elif checks_complete and not (info.get("geometry") or {}).get("crop_fits", True):
            # All checks pass, but the auto shutter waits until the crop fits in the frame.
            self.coach.setText(f"<b>{t('framing', L)}:</b> {t('msg_move_back', L)}")
            self.coach.setProperty("ok", False)
            self.coach.setProperty("pending", False)
        elif checks_complete:
            self.coach.setText(f"<b>{t('all_pass', L)}</b>")
            self.coach.setProperty("ok", True)
            self.coach.setProperty("pending", False)
        else:
            self.coach.setText(f"<b>{t('checks_pending', L)}</b>")
            self.coach.setProperty("ok", False)
            self.coach.setProperty("pending", True)
        status = (self.modnet_last.get("subject_status") if self.modnet_last is not None
                  else info.get("subject_status"))
        if self.settings.modnet_demo and status in ("no_face", "ambiguous", "changed"):
            self.coach.setText(f"<b>{t('subject_' + status, L)}</b>")
        self.coach.style().unpolish(self.coach)
        self.coach.style().polish(self.coach)

        count, frames, armed = info["shutter"]
        self.progress.setValue(min(count, frames))
        self.shutter_text.setText(t("passing", L, n=min(count, frames), m=frames))
        lock_key = {"warming": "lock_warming", "locked": "lock_locked"}.get(info["lock"], f"lock_{info['lock']}")
        st = info["stats"]
        rlr = ""
        if st.get("r_lr_before") is not None and math.isfinite(st["r_lr_before"]):
            rlr = f" · R_LR {st['r_lr_before']:.2f} → {st['r_lr_after']:.2f}"
        rate_label = ("Clothing " if self.settings.fba_demo
                      else "MODNet " if self.settings.modnet_demo else "")
        self.status.setText(f"{info['source']} · {rate_label}{info['fps']:.0f} fps{rlr}\n{t(lock_key, L)}")
        self.update_fba_status(info.get("fba_capture_busy", False))
        capture_busy = info.get("model_capture_busy", False)
        self.update_modnet_status(capture_busy)
        self.update_birefnet_status(capture_busy)

    def on_captured(self, result: dict):
        guard = result.pop("_capture_display_guard", None)
        result.pop("_capture_event_timing_ms", None)
        if self.closing:
            return
        if guard is not None:
            capture_timing = guard.setdefault("capture_timing_ms", {})
            capture_timing["gui_result_received_ms"] = time.monotonic() * 1000
            stale_reasons = _capture_stale_reasons(
                guard, self.settings, self.model_generation,
                self.auto_shutter_generation, getattr(self.worker, "track_epoch", 0),
            )
            if guard["automatic"] and not self.settings.auto_shutter:
                stale_reasons.append("AUTO setting changed")
            if stale_reasons:
                self.on_model_failed((
                    True, guard["model"], None,
                    "Capture result discarded before display because "
                    + ", ".join(stale_reasons) + ".",
                ))
                return
            capture_timing["gui_guard_passed_ms"] = time.monotonic() * 1000
        self.view.flash = 1.0
        self.flash_timer.start(30)
        self.dialog = ResultDialog(self, result, self.lang)
        self.dialog.finished.connect(self._dialog_closed)
        self.dialog.open()
        if guard is not None:
            capture_timing["result_dialog_opened_ms"] = time.monotonic() * 1000
            guard["result_dialog_is_expected_type"] = isinstance(
                self.dialog, ResultDialog
            )
            guard["result_dialog_visible"] = bool(self.dialog.isVisible())
        self.capture_displayed.emit(result, guard)

    def _dialog_closed(self, *_):
        self.dialog = None
        self.worker.rearm_request = True

    def _fade_flash(self):
        self.view.flash = max(0.0, self.view.flash - 0.12)
        self.view.update()
        if self.view.flash <= 0:
            self.flash_timer.stop()

    def on_failed(self, message: str):
        self.view.message = t("camera_error", self.lang, err=message)
        self.view.update()

    def closeEvent(self, event):
        self.open_timer.stop()
        if self.worker:
            self.worker.running = False
            if not self.worker.wait(3000):
                if not self.closing:
                    self.closing = True
                    self.worker.finished.connect(self.close)
                self._show_message(t("finishing_processing", self.lang))
                event.ignore()
                return
        super().closeEvent(event)


PANEL_STYLE = f"""
QWidget#panel {{ background: #fbfbfa; }}
QLabel {{ color: #1d1d1f; font-size: 14px; }}
QLabel#title {{ font-size: 18px; font-weight: 600; }}
QLabel#section {{ color: #6b6b6b; font-size: 12px; font-weight: 600; letter-spacing: 1px; margin-top: 5px; }}
QLabel#muted {{ color: #6b6b6b; font-size: 12px; }}
QPushButton#lang {{ border: 1px solid #c8c8c8; border-radius: 12px; padding: 4px 12px; background: white; }}
QPushButton#segment {{ border: 1px solid #1d1d1f; padding: 8px 4px; background: white; font-size: 13px; }}
QPushButton#segment[pos="first"] {{ border-top-left-radius: 8px; border-bottom-left-radius: 8px; }}
QPushButton#segment[pos="last"] {{ border-top-right-radius: 8px; border-bottom-right-radius: 8px; }}
QPushButton#segment:checked {{ background: #111; color: white; font-weight: 600; }}
QFrame#checks {{ background: white; border: 1px solid #dedede; border-radius: 10px; }}
QLabel#coach {{ background: #fbeeea; border: 1px solid #efc4b8; border-radius: 8px; padding: 10px; }}
QLabel#coach[ok="true"] {{ background: #eaf6ec; border: 1px solid #b9dfc1; }}
QLabel#coach[pending="true"] {{ background: #f5f2e8; border: 1px solid #ded5b5; }}
QLabel#notice {{ border: 1px dashed #a0a0a0; border-radius: 8px; padding: 10px; color: #333; background: #f6f6f4; font-size: 13px; }}
QProgressBar {{ background: #e6e6e6; border: none; border-radius: 5px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QPushButton#shoot {{ background: {ACCENT}; color: white; border: none; border-radius: 8px; padding: 8px 16px; font-weight: 600; }}
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Webcam ID photo app")
    parser.add_argument("--source", default=None, help="camera index (default 0), 'demo', or an image/video file")
    parser.add_argument("--lang", choices=["ko", "en"], default=None, help="interface language")
    parser.add_argument("--camera-size", default=f"{config.CAMERA_SIZE[0]}x{config.CAMERA_SIZE[1]}",
                        help="requested camera resolution, e.g. 1920x1080")
    parser.add_argument("--face-model", default=None,
                        help="local MediaPipe face_landmarker.task path; required outside a source checkout")
    parser.add_argument("--presentation-fba", action="store_true",
                        help="compatibility flag; prepared FBA is already selected by default")
    parser.add_argument("--fba-device", choices=("cpu", "mps"), default="cpu",
                        help="FBA presentation inference device (default: cpu; mps: Apple GPU)")
    parser.add_argument("--presentation-modnet", action="store_true",
                        help="enable the opt-in local MODNet experimental preview")
    parser.add_argument("--modnet-model", default="models/modnet.onnx",
                        help="local official MODNet ONNX model path (never downloaded automatically)")
    parser.add_argument("--modnet-provider", choices=("cpu", "coreml-static"), default="cpu",
                        help="MODNet provider; coreml-static pins the stride-aligned input grid")
    parser.add_argument("--birefnet-model", default=None,
                        help="optional local BiRefNet portrait ONNX for shutter captures only; never downloaded")
    parser.add_argument("--snapshot", default=None, help=argparse.SUPPRESS)  # save a window screenshot and quit
    parser.add_argument("--snapshot-after", type=float, default=6.0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.modnet_provider != "cpu" and not args.presentation_modnet:
        parser.error("--modnet-provider requires --presentation-modnet")
    if args.birefnet_model and not args.presentation_modnet:
        parser.error("--birefnet-model requires --presentation-modnet")

    app = QApplication(sys.argv[:1])
    lang = args.lang or ("ko" if QApplication.instance() and _system_is_korean() else "en")
    size = tuple(int(x) for x in args.camera_size.lower().split("x"))
    window = MainWindow(args.source, lang, size, presentation_fba=True, start_fba=True,
                        fba_device=args.fba_device, presentation_modnet=args.presentation_modnet,
                        modnet_model=args.modnet_model, modnet_provider=args.modnet_provider,
                        birefnet_model=args.birefnet_model, face_model=args.face_model)
    keys = KeyFilter(window)
    app.installEventFilter(keys)
    window.show()
    if args.snapshot:
        QTimer.singleShot(int(args.snapshot_after * 1000), lambda: _snapshot(window, args.snapshot, app))
    return app.exec()


def _system_is_korean() -> bool:
    from PySide6.QtCore import QLocale

    return QLocale.system().language() == QLocale.Language.Korean


def _snapshot(window: MainWindow, path: str, app: QApplication):
    window.grab().save(path)
    if window.dialog is not None:
        window.dialog.grab().save(str(Path(path).with_name(Path(path).stem + "_result.png")))
    window.close()
    app.quit()


if __name__ == "__main__":
    sys.exit(main())
