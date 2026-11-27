"""Measure fresh MODNet results through the application's selected Qt main view.

Example (public VideoMatte240K test-split clips, extracted locally):
    uv run --extra parsing python tools/bench_modnet_live.py \
        --model /path/to/modnet.onnx --video fgr-0000.mp4 fgr-0001.mp4 \
        --out out/modnet-live.json

The runner stores numeric events only. It never writes video frames or captures.
"""

import argparse
import csv
import hashlib
import json
import math
import numbers
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Slot
from PySide6.QtWidgets import QApplication

from idphoto import config
from idphoto.gui import MainWindow


class VideoReplaySource:
    """Pace a sequence of public video files as a camera-like RGB source."""

    def __init__(self, videos, width=1280, fps=30):
        self.videos = [Path(path) for path in videos]
        self.width = width
        self.period = 1.0 / fps
        self.captures = []
        for path in self.videos:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                for opened in self.captures:
                    opened.release()
                raise FileNotFoundError(f"Unable to open replay video: {path}")
            self.captures.append(cap)
        self.index = 0
        self.next_deadline = time.monotonic()
        self.read_count = 0
        self.read_events = []
        self.eof = False
        self.eof_timestamp_ms = None
        self.lock_state = "locked"
        self.name = "public motion replay"

    def read(self):
        while self.index < len(self.captures):
            ok, bgr = self.captures[self.index].read()
            if ok:
                break
            self.index += 1
        else:
            self.eof = True
            if self.eof_timestamp_ms is None:
                self.eof_timestamp_ms = time.monotonic() * 1000
            return None, None

        clip_index = self.index
        height, width = bgr.shape[:2]
        target_height = max(1, round(height * self.width / width))
        if width != self.width:
            bgr = cv2.resize(bgr, (self.width, target_height), interpolation=cv2.INTER_AREA)
        delay = self.next_deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next_deadline = max(self.next_deadline + self.period, time.monotonic())
        timestamp_ms = time.monotonic() * 1000
        self.read_count += 1
        self.read_events.append({
            "frame_id": self.read_count,
            "timestamp_ms": timestamp_ms,
            "clip_name": self.videos[clip_index].name,
        })
        return np.ascontiguousarray(bgr[:, :, ::-1]), None, timestamp_ms

    def release(self):
        for cap in self.captures:
            cap.release()


class ReplayMetrics(QObject):
    def __init__(self, window, source, app, started_ms, warmup_seconds, duration_seconds,
                 capture_offset_seconds=None):
        super().__init__(window)
        self.window = window
        self.source = source
        self.app = app
        self.started_ms = started_ms
        self.warmup_seconds = warmup_seconds
        self.duration_seconds = duration_seconds
        self.capture_offset_seconds = capture_offset_seconds
        self.capture_deadline_ms = None
        self.model_events = []
        self.paint_events = []
        self.processed_events = []
        self.capture_requested_ms = None
        self.capture_completed_ms = None
        self.capture_signal_received_ms = None
        self.capture_requested_utc = None
        self.capture_completed_utc = None
        self.measurement_started_utc = None
        self.measurement_ended_utc = None
        self.process_clock_samples = []
        self.capture_result_count = 0
        self.capture_display_count = 0
        self.capture_results = []
        self.capture_error = None
        self.start_ms = None
        self.end_ms = None
        self.startup_timeout = False
        self.warmup_scheduled = False

    def _maybe_schedule_warmup(self):
        if self.warmup_scheduled:
            return
        face_result_ids = {event["frame_id"] for event in self.model_events if event["face_detected"]}
        if face_result_ids.intersection(event["frame_id"] for event in self.paint_events):
            self.warmup_scheduled = True
            QTimer.singleShot(round(self.warmup_seconds * 1000), self.begin_measurement)

    @Slot(object)
    def on_model_result(self, result):
        if result.get("model") != "modnet":
            return
        self.model_events.append({
            "frame_id": result["frame_id"],
            "source_timestamp_ms": result["timestamp_ms"],
            "received_ms": time.monotonic() * 1000,
            "face_detected": bool(result["face_detected"]),
            "fully_processed": bool(result.get("fully_processed", False)),
            "light_valid": bool(result.get("light_valid", False)),
            "face_mask_valid": bool(result.get("face_mask_valid", False)),
            "geometry_valid": bool(result.get("geometry_valid", False)),
            "rule_count": result.get("rule_count", 0),
            "landmark_source": result.get("landmark_source", "unknown"),
            "track_break": bool(result.get("track_break", False)),
            "landmark_detector_calls": result.get("landmark_detector_calls", 0),
            "inference_ms": result.get("inference_ms"),
            "inference_started_ms": result.get("inference_started_ms"),
            "inference_finished_ms": result.get("inference_finished_ms"),
            "prepare_finished_ms": result.get("prepare_finished_ms"),
            "model_job_timing_ms": result.get("model_job_timing_ms"),
            "finalized_ms": result.get("finalized_ms"),
            "pipeline_timings_ms": result.get("pipeline_timings_ms"),
            "wall_estimate_ms": result.get("wall_estimate_ms"),
            "composite_ms": result.get("composite_ms"),
            "execution_provider": result.get("execution_provider"),
            "provider_backends": result.get("provider_backends"),
            "input_width": result.get("input_width"),
            "input_height": result.get("input_height"),
        })
        self._maybe_schedule_warmup()

    @Slot(int, float)
    def on_painted(self, frame_id, painted_ms):
        self.paint_events.append({"frame_id": frame_id, "painted_ms": painted_ms})
        self._maybe_schedule_warmup()

    @Slot(object)
    def on_processed(self, info):
        self.processed_events.append({
            "frame_id": info["source_frame_id"],
            "source_timestamp_ms": info["timestamp_ms"],
            "received_ms": time.monotonic() * 1000,
        })

    def _record_process_clock(self, event, target_offset_s=None):
        observed_ms = time.monotonic() * 1000
        dialog = getattr(self.window, "dialog", None)
        self.process_clock_samples.append({
            "event": event,
            "target_offset_s": target_offset_s,
            "observed_offset_s": (
                (observed_ms - self.start_ms) / 1000
                if self.start_ms is not None else None
            ),
            "observed_monotonic_ms": observed_ms,
            "process_cpu_seconds": time.process_time(),
            "dialog_visible": bool(dialog is not None and dialog.isVisible()),
        })

    def _record_window_boundary(self, offset_s):
        deadline_ms = self.start_ms + offset_s * 1000
        delay_ms = _remaining_delay_ms(deadline_ms, time.monotonic() * 1000)
        if delay_ms:
            QTimer.singleShot(
                delay_ms, lambda boundary=offset_s: self._record_window_boundary(boundary)
            )
            return
        if self.end_ms is None:
            self._record_process_clock("window_boundary", offset_s)

    @Slot()
    def begin_measurement(self):
        if self.start_ms is not None:
            return
        self.start_ms = time.monotonic() * 1000
        self.measurement_started_utc = datetime.now(timezone.utc).isoformat()
        self._record_process_clock("measurement_start", 0)
        for offset_s in range(10, int(self.duration_seconds), 10):
            QTimer.singleShot(
                math.ceil(offset_s * 1000),
                lambda boundary=offset_s: self._record_window_boundary(boundary),
            )
        if self.capture_offset_seconds is not None:
            self.capture_deadline_ms = (
                self.start_ms + self.capture_offset_seconds * 1000
            )
            QTimer.singleShot(
                math.ceil(self.capture_offset_seconds * 1000), self.request_capture
            )
        QTimer.singleShot(round(self.duration_seconds * 1000), self.finish_measurement)

    @Slot()
    def request_capture(self):
        if self.capture_requested_ms is not None:
            return
        if self.capture_deadline_ms is not None:
            delay_ms = _remaining_delay_ms(
                self.capture_deadline_ms, time.monotonic() * 1000
            )
            if delay_ms:
                QTimer.singleShot(delay_ms, self.request_capture)
                return
        self.capture_requested_ms = time.monotonic() * 1000
        self.capture_requested_utc = datetime.now(timezone.utc).isoformat()
        self._record_process_clock(
            "shutter_requested",
            (self.capture_requested_ms - self.start_ms) / 1000,
        )
        self.window.shoot.click()

    @Slot(object)
    def on_capture_emitted(self, _result):
        self.capture_result_count += 1
        self.capture_signal_received_ms = time.monotonic() * 1000

    @Slot(object, object)
    def on_capture_displayed(self, result, guard):
        self.capture_display_count += 1
        self.capture_completed_ms = time.monotonic() * 1000
        self.capture_completed_utc = datetime.now(timezone.utc).isoformat()
        self._record_process_clock(
            "result_dialog_opened",
            (self.capture_completed_ms - self.start_ms) / 1000,
        )
        report = result.get("report", {})
        ctx = result.get("ctx")
        ctx_stats = getattr(ctx, "stats", {})
        model_stats = ctx_stats.get("birefnet_portrait", {})
        source = ctx_stats.get("capture_source", {})
        guard_frame_id = guard.get("frame_id") if guard else None
        source_frame_id = source.get("frame_id") if isinstance(source, dict) else None
        model_frame_id = model_stats.get("source_frame_id") if isinstance(model_stats, dict) else None
        guard_timestamp_ms = guard.get("timestamp_ms") if guard else None
        guard_read_timestamp_ms = guard.get("source_read_timestamp_ms") if guard else None
        guard_manual_request_ms = (
            guard.get("manual_request_timestamp_ms") if guard else None
        )
        source_timestamp_ms = source.get("timestamp_ms") if isinstance(source, dict) else None
        model_timestamp_ms = (
            model_stats.get("source_timestamp_ms") if isinstance(model_stats, dict) else None
        )
        source_read_timestamp_ms = (
            source.get("source_read_timestamp_ms") if isinstance(source, dict) else None
        )
        manual_request_timestamp_ms = (
            source.get("manual_request_timestamp_ms") if isinstance(source, dict) else None
        )
        context_timestamp_ms = getattr(ctx, "timestamp_ms", None)
        source_read = _unique_event_for_id(
            getattr(self.source, "read_events", []), guard_frame_id, "frame_id"
        )
        modnet_model_event = _unique_event_for_id(
            self.model_events, guard_frame_id, "frame_id"
        )
        modnet_worker_event = _unique_event_for_id(
            self.processed_events, guard_frame_id, "frame_id"
        )
        capture_worker_input = (
            guard.get("capture_worker_input") if isinstance(guard, dict) else None
        )
        capture_timing = guard.get("capture_timing_ms") if isinstance(guard, dict) else {}
        capture_result = {
            "model": guard.get("model") if guard else None,
            "capture_matte": report.get("settings", {}).get("capture_matte"),
            "original_wall_kept": report.get("original_wall_kept"),
            "report_rules": sorted(int(rule) for rule in report.get("rules", {})),
            "source_frame_id": guard_frame_id,
            "source_timestamp_ms": guard_timestamp_ms,
            "model_source_frame_id": model_frame_id,
            "model_source_timestamp_ms": model_timestamp_ms,
            "returned_ctx_source_frame_id": source_frame_id,
            "returned_ctx_source_timestamp_ms": source_timestamp_ms,
            "returned_ctx_timestamp_ms": context_timestamp_ms,
            "capture_requested_timestamp_ms": self.capture_requested_ms,
            "manual_request_timestamp_ms": manual_request_timestamp_ms,
            "source_read_timestamp_ms": source_read_timestamp_ms,
            "worker_guard_manual_request_timestamp_ms": guard_manual_request_ms,
            "worker_guard_source_read_timestamp_ms": guard_read_timestamp_ms,
            "capture_worker_input": (
                dict(capture_worker_input)
                if isinstance(capture_worker_input, dict) else None
            ),
            "source_read_event": source_read,
            "modnet_model_snapshot_event": modnet_model_event,
            "modnet_worker_frame_ready_event": modnet_worker_event,
            "source_frame_shape": (
                list(ctx.raw_frame.shape) if ctx is not None
                and getattr(ctx, "raw_frame", None) is not None else None
            ),
            "crop_boxes": report.get("geometry", {}).get("crop_boxes"),
            "model_stats": dict(model_stats) if isinstance(model_stats, dict) else None,
            "model_job_timing_ms": guard.get("model_job_timing_ms") if guard else None,
            "capture_timing_ms": dict(capture_timing or {}),
            "pipeline_capture_timing_ms": (
                dict(ctx_stats.get("capture_timing_ms") or {})
                if isinstance(ctx_stats.get("capture_timing_ms"), dict) else None
            ),
            "result_dialog_is_expected_type": (
                guard.get("result_dialog_is_expected_type") is True
                if isinstance(guard, dict) else False
            ),
            "result_dialog_visible": (
                guard.get("result_dialog_visible") is True
                if isinstance(guard, dict) else False
            ),
        }
        capture_result["dialog_opened"] = (
            capture_result["result_dialog_is_expected_type"]
            and capture_result["result_dialog_visible"]
        )
        capture_result["source_linked"] = _capture_lineage_is_valid(capture_result)
        self.capture_results.append(capture_result)

    @Slot(object)
    def on_model_failed(self, error):
        capture, _model, _generation, message = error
        if capture and self.capture_requested_ms is not None:
            self.capture_error = message

    @Slot()
    def finish_measurement(self):
        now_ms = time.monotonic() * 1000
        remaining_ms = self.duration_seconds * 1000 - (now_ms - self.start_ms)
        if remaining_ms > 0:
            QTimer.singleShot(max(1, math.ceil(remaining_ms)), self.finish_measurement)
            return
        self.end_ms = now_ms
        self.measurement_ended_utc = datetime.now(timezone.utc).isoformat()
        self._record_process_clock(
            "measurement_end", (self.end_ms - self.start_ms) / 1000
        )
        self._shutdown()

    @Slot()
    def startup_timeout_reached(self):
        if self.start_ms is None:
            self.startup_timeout = True
            self.end_ms = time.monotonic() * 1000
            self._shutdown()

    def _shutdown(self):
        if self.window.worker is not None:
            self.window.worker.running = False

        def wait_for_worker():
            if self.window.worker is not None and self.window.worker.isRunning():
                QTimer.singleShot(50, wait_for_worker)
                return
            self.window.close()
            self.app.quit()

        QTimer.singleShot(0, wait_for_worker)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values):
    if not values:
        return {"p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(max(values)),
    }


def _in_window(timestamp_ms, start_ms, end_ms):
    return start_ms <= timestamp_ms < end_ms


def _longest_gap_ms(timestamps):
    ordered = sorted(timestamps)
    return max((right - left for left, right in zip(ordered, ordered[1:])), default=None)


def _remaining_delay_ms(deadline_ms, now_ms):
    remaining = deadline_ms - now_ms
    return max(0, math.ceil(remaining))


def _process_clock_report(samples):
    ordered = sorted(samples, key=lambda sample: sample["observed_monotonic_ms"])
    intervals = []
    for before, after in zip(ordered, ordered[1:]):
        wall_seconds = (
            after["observed_monotonic_ms"] - before["observed_monotonic_ms"]
        ) / 1000
        cpu_seconds = after["process_cpu_seconds"] - before["process_cpu_seconds"]
        intervals.append({
            "from_event": before["event"],
            "to_event": after["event"],
            "wall_seconds": wall_seconds,
            "process_cpu_seconds": cpu_seconds,
            "process_cpu_to_wall_ratio": (
                cpu_seconds / wall_seconds if wall_seconds > 0 else None
            ),
            "dialog_visible_at_start": before["dialog_visible"],
            "dialog_visible_at_end": after["dialog_visible"],
        })
    opened = next(
        (sample for sample in ordered if sample["event"] == "result_dialog_opened"),
        None,
    )
    after_open = [
        sample for sample in ordered
        if opened is not None
        and sample["observed_monotonic_ms"] >= opened["observed_monotonic_ms"]
        and sample["event"] in ("window_boundary", "measurement_end")
    ]
    return {
        "samples": ordered,
        "intervals": intervals,
        "dialog_visible_at_all_post_open_samples": (
            all(sample["dialog_visible"] for sample in after_open)
            if opened is not None and after_open else None
        ),
        "post_open_sample_count": len(after_open),
    }


def _summarize_gates(measured_seconds, windows, average_rates, latency_ages,
                     source_eof_before_end=False):
    complete_window_count = int(measured_seconds // 10)
    enough_windows = measured_seconds >= 60 and complete_window_count >= 6 \
        and len(windows) == complete_window_count
    average_rate_gates = {
        key: measured_seconds >= 60 and rate >= 24
        for key, rate in average_rates.items()
    }
    window_rate_gates = {
        key: enough_windows and all(window[f"{key}_rate_pass"] for window in windows)
        for key in average_rates
    }
    latency_gates = {
        key: bool(ages) and float(np.percentile(ages, 95)) <= 100
        for key, ages in latency_ages.items()
    }
    overall_pass = enough_windows and all(average_rate_gates.values()) \
        and all(window_rate_gates.values()) and all(latency_gates.values()) \
        and not source_eof_before_end
    return {
        "duration_at_least_60s_with_complete_10s_windows": enough_windows,
        "source_eof_before_measurement_end": bool(source_eof_before_end),
        "average_rate_gates": average_rate_gates,
        "every_complete_10s_window_rate_gates": window_rate_gates,
        "average_at_least_24_fresh_face_results_per_second":
            average_rate_gates.get("fresh_face_results", False),
        "at_least_24_fresh_face_results_per_second":
            average_rate_gates.get("fresh_face_results", False)
            and window_rate_gates.get("fresh_face_results", False),
        "average_at_least_24_fresh_face_matched_main_view_paints_per_second":
            average_rate_gates.get("fresh_face_paints", False),
        "at_least_24_fresh_face_matched_main_view_paints_per_second":
            average_rate_gates.get("fresh_face_paints", False)
            and window_rate_gates.get("fresh_face_paints", False),
        "average_at_least_24_fully_processed_results_per_second":
            average_rate_gates.get("fully_processed_results", False),
        "average_at_least_24_fully_processed_main_view_paints_per_second":
            average_rate_gates.get("fully_processed_paints", False),
        "at_least_24_fully_processed_results_per_second":
            average_rate_gates.get("fully_processed_results", False)
            and window_rate_gates.get("fully_processed_results", False),
        "at_least_24_fully_processed_main_view_paints_per_second":
            average_rate_gates.get("fully_processed_paints", False)
            and window_rate_gates.get("fully_processed_paints", False),
        "p95_any_source_read_to_paint_at_most_100ms":
            latency_gates.get("any_paint", False),
        "p95_face_matched_source_read_to_paint_at_most_100ms":
            latency_gates.get("face_paint", False),
        "p95_fully_processed_source_read_to_paint_at_most_100ms":
            latency_gates.get("fully_processed_paint", False),
        "overall_pass": overall_pass,
    }


def _cli_exit_code(report):
    if report.get("gate", {}).get("overall_pass") is not True:
        return 1
    capture = report.get("capture_validation") or {}
    if (report.get("capture_requested") or report.get("capture_validation") is not None) and (
        capture.get("gate", {}).get("overall_pass") is not True
    ):
        return 1
    return 0


def _is_integer_id(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_finite_number(value):
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _same_frame_id(*values):
    return (
        bool(values)
        and all(_is_integer_id(value) for value in values)
        and all(int(value) == int(values[0]) for value in values[1:])
    )


def _same_timestamp(*values):
    return (
        bool(values)
        and all(_is_finite_number(value) for value in values)
        and all(
        math.isclose(float(value), float(values[0]), rel_tol=0.0, abs_tol=0.01)
            for value in values[1:]
        )
    )


def _unique_event_for_id(events, frame_id, id_field):
    if not _is_integer_id(frame_id):
        return None
    matches = [
        event for event in events
        if isinstance(event, dict)
        and _is_integer_id(event.get(id_field))
        and int(event[id_field]) == int(frame_id)
    ]
    return matches[0] if len(matches) == 1 else None


def _capture_lineage_is_valid(result):
    read_event = result.get("source_read_event")
    model_event = result.get("modnet_model_snapshot_event")
    worker_event = result.get("modnet_worker_frame_ready_event")
    worker_input = result.get("capture_worker_input")
    model_stats = result.get("model_stats") or {}
    job = result.get("model_job_timing_ms") or {}
    capture_timing = result.get("capture_timing_ms") or {}
    if not all(isinstance(item, dict) for item in (
        read_event, model_event, worker_event, worker_input,
        model_stats, job, capture_timing,
    )):
        return False

    frame_ids = (
        result.get("source_frame_id"),
        result.get("model_source_frame_id"),
        result.get("returned_ctx_source_frame_id"),
        read_event.get("frame_id"),
        model_event.get("frame_id"),
        worker_event.get("frame_id"),
        worker_input.get("frame_id"),
        model_stats.get("source_frame_id"),
    )
    read_timestamp = read_event.get("timestamp_ms")
    request_timestamp = result.get("capture_requested_timestamp_ms")
    manual_timestamp = result.get("manual_request_timestamp_ms")
    worker_started = job.get("started")
    worker_input_started = worker_input.get("started_ms")
    job_submitted = job.get("submitted")
    job_finished = job.get("finished")
    dialog_opened = capture_timing.get("result_dialog_opened_ms")
    linked_timestamps = (
        result.get("source_timestamp_ms"),
        result.get("model_source_timestamp_ms"),
        result.get("returned_ctx_source_timestamp_ms"),
        result.get("returned_ctx_timestamp_ms"),
        read_timestamp,
        model_event.get("source_timestamp_ms"),
        worker_event.get("source_timestamp_ms"),
        worker_input.get("timestamp_ms"),
        result.get("source_read_timestamp_ms"),
        result.get("worker_guard_source_read_timestamp_ms"),
        worker_input.get("source_read_timestamp_ms"),
        model_stats.get("source_timestamp_ms"),
        model_stats.get("source_read_timestamp_ms"),
    )
    same_manual_timestamps = _same_timestamp(
        manual_timestamp,
        result.get("worker_guard_manual_request_timestamp_ms"),
        worker_input.get("manual_request_timestamp_ms"),
        model_stats.get("manual_request_timestamp_ms"),
    )
    model_event_received = model_event.get("received_ms")
    worker_event_received = worker_event.get("received_ms")
    return (
        result.get("source_linked") is not False
        and result.get("result_dialog_is_expected_type") is True
        and result.get("result_dialog_visible") is True
        and result.get("dialog_opened") is True
        and _same_frame_id(*frame_ids)
        and _same_timestamp(*linked_timestamps)
        and same_manual_timestamps
        and _is_finite_number(request_timestamp)
        and _is_finite_number(manual_timestamp)
        and float(request_timestamp) <= float(manual_timestamp)
        and float(manual_timestamp) < float(read_timestamp)
        and _is_finite_number(worker_started)
        and _is_finite_number(worker_input_started)
        and _is_finite_number(job_submitted)
        and _is_finite_number(job_finished)
        and _same_timestamp(worker_started, worker_input_started)
        and float(read_timestamp) <= float(job_submitted) <= float(worker_started)
        <= float(job_finished)
        and _is_finite_number(dialog_opened)
        and float(job_finished) <= float(dialog_opened)
        and _is_finite_number(model_event_received)
        and _is_finite_number(worker_event_received)
        and float(read_timestamp) <= float(model_event_received) <= float(dialog_opened)
        and float(read_timestamp) <= float(worker_event_received) <= float(dialog_opened)
    )


def _capture_timing_is_recorded(result):
    model = result.get("model_stats") or {}
    pipeline = result.get("pipeline_capture_timing_ms") or {}
    job = result.get("model_job_timing_ms") or {}
    events = result.get("capture_timing_ms") or {}
    model_durations = (
        "session_create_ms", "preprocess_ms", "inference_ms",
        "alpha_postprocess_ms", "total_predict_ms",
    )
    pipeline_durations = (
        "frame_decode_relight_ms", "matte_apply_call_ms",
        "checker_wall_estimate_ms", "checker_ms", "composite_ms",
        "crop_photo_sheet_render_ms", "render_final_total_ms",
    )
    event_times = (
        "worker_result_callback_ms", "worker_guard_checked_ms",
        "worker_signal_emitted_ms", "gui_result_received_ms",
        "gui_guard_passed_ms", "result_dialog_opened_ms",
    )
    return (
        model.get("inference_count") == 1
        and all(_is_finite_number(model.get(key)) for key in model_durations)
        and all(_is_finite_number(pipeline.get(key)) for key in pipeline_durations)
        and _is_finite_number(pipeline.get("render_final_started_monotonic_ms"))
        and _is_finite_number(pipeline.get("render_final_finished_monotonic_ms"))
        and all(_is_finite_number(job.get(key)) for key in ("submitted", "started", "finished"))
        and all(_is_finite_number(events.get(key)) for key in event_times)
        and job["submitted"] <= job["started"] <= pipeline[
            "render_final_started_monotonic_ms"
        ] <= pipeline["render_final_finished_monotonic_ms"] <= job["finished"]
        <= events["worker_result_callback_ms"] <= events["worker_guard_checked_ms"]
        <= events["worker_signal_emitted_ms"] <= events["gui_result_received_ms"]
        <= events["gui_guard_passed_ms"] <= events["result_dialog_opened_ms"]
    )


def _capture_interval_summary(
    start_ms, end_ms, model_events, paint_events,
    source_read_events=None, request_ms=None,
):
    source_read_gate_applied = (
        source_read_events is not None and _is_finite_number(request_ms)
    )
    source_reads = {}
    if source_read_events is not None:
        for event in source_read_events:
            if (
                isinstance(event, dict)
                and _is_integer_id(event.get("frame_id"))
                and event["frame_id"] not in source_reads
            ):
                source_reads[event["frame_id"]] = event

    missing_source_reads = 0
    pre_request_source_reads = 0
    fully_processed = {}
    for event in model_events:
        if not (
            event["fully_processed"]
            and start_ms <= event["received_ms"] <= end_ms
        ):
            continue
        if source_read_gate_applied:
            read_event = _unique_event_for_id(
                source_read_events, event.get("frame_id"), "frame_id"
            )
            if read_event is None or not _is_finite_number(read_event.get("timestamp_ms")):
                missing_source_reads += 1
                continue
            if float(read_event["timestamp_ms"]) <= float(request_ms):
                pre_request_source_reads += 1
                continue
        fully_processed.setdefault(event["frame_id"], event)

    painted = {}
    for event in paint_events:
        if (event["frame_id"] in fully_processed
                and start_ms <= event["painted_ms"] <= end_ms):
            painted.setdefault(event["frame_id"], event)
    ages = []
    for frame_id, event in fully_processed.items():
        if frame_id not in painted:
            continue
        source_timestamp = (
            source_reads[frame_id]["timestamp_ms"]
            if source_read_gate_applied
            else event["source_timestamp_ms"]
        )
        ages.append(painted[frame_id]["painted_ms"] - source_timestamp)
    paint_times = sorted(event["painted_ms"] for event in painted.values())
    duration_ms = end_ms - start_ms
    paint_rate = len(painted) * 1000 / duration_ms if duration_ms > 0 else 0.0
    source_to_paint = _quantiles(ages)
    return {
        "duration_ms": duration_ms,
        "fully_processed_results": len(fully_processed),
        "unique_fully_processed_paints": len(painted),
        "fully_processed_paint_rate_hz": paint_rate,
        "fully_processed_paint_rate_pass": paint_rate >= 24,
        "source_read_to_paint": source_to_paint,
        "source_read_to_paint_sample_count": len(ages),
        "source_read_to_paint_p95_pass": (
            source_to_paint["p95_ms"] is not None
            and source_to_paint["p95_ms"] <= 100
        ),
        "source_read_timestamp_gate_applied": source_read_gate_applied,
        "source_read_event_missing_count": missing_source_reads,
        "pre_request_source_read_excluded_count": pre_request_source_reads,
        "source_read_events_linked": (
            source_read_gate_applied and missing_source_reads == 0
        ),
        "longest_fresh_paint_gap_ms_including_request_and_result": _longest_gap_ms(
            [start_ms, *paint_times, end_ms]
        ),
    }


def _pipeline_timing_columns(event):
    timings = event.get("pipeline_timings_ms") or {}
    return {
        f"pipeline_{name}_ms": timings.get(name, "")
        for name in ("landmarks", "relight", "matting", "geometry", "checker")
    }


def _capture_gate(capture):
    results = capture.get("capture_results", [])
    result = results[0] if len(results) == 1 else {}
    stats = result.get("model_stats") or {}
    actual_birefnet_cpu = (
        result.get("model") == "birefnet"
        and result.get("capture_matte") == "birefnet-portrait"
        and stats.get("provider") == "CPUExecutionProvider"
        and _is_finite_number(stats.get("inference_ms"))
        and stats["inference_ms"] > 0
    )
    source_linked = (
        result.get("source_linked") is True
        and _capture_lineage_is_valid(result)
        and _is_integer_id(result.get("source_frame_id"))
        and _is_finite_number(result.get("source_timestamp_ms"))
        and bool(result.get("crop_boxes"))
    )
    checks = {
        "exactly_one_capture_result": capture.get("capture_result_count") == 1,
        "result_dialog_opened": capture.get("capture_display_count") == 1
        and result.get("dialog_opened") is True,
        "actual_birefnet_cpu_inference": actual_birefnet_cpu,
        "exactly_one_birefnet_inference": stats.get("inference_count") == 1,
        "source_frame_and_crop_recorded": source_linked,
        "capture_interval_source_reads_verified":
            capture.get("source_read_timestamp_gate_applied") is True
            and capture.get("source_read_event_missing_count") == 0
            and capture.get("source_read_events_linked") is True,
        "capture_stage_timings_recorded": _capture_timing_is_recorded(result),
        "no_capture_error": capture.get("capture_error") is None,
        "at_least_24_fully_processed_paints_per_second":
            capture.get("fully_processed_paint_rate_pass") is True,
        "source_read_to_paint_p95_at_most_100ms":
            capture.get("source_read_to_paint_p95_pass") is True,
    }
    return {**checks, "overall_pass": all(checks.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="local official MODNet ONNX path")
    parser.add_argument("--video", nargs="+", required=True, help="ordered public motion clips")
    parser.add_argument("--provider", choices=("cpu", "coreml-static"), default="cpu")
    parser.add_argument("--input-width", type=int, default=1280,
                        help="paced replay width before the production 640px live resize")
    parser.add_argument("--warmup-seconds", type=float, default=3)
    parser.add_argument("--duration-seconds", type=float, default=60)
    parser.add_argument("--startup-timeout-seconds", type=float, default=30)
    parser.add_argument("--birefnet-model", default=None,
                        help="local BiRefNet portrait model for one real GUI capture request")
    parser.add_argument("--capture-offset-seconds", type=float, default=None,
                        help="trigger the GUI shutter this many seconds into the measured interval")
    parser.add_argument("--out", type=Path, required=True, help="numeric JSON report path")
    args = parser.parse_args(argv)
    if args.duration_seconds < 60:
        parser.error("--duration-seconds must be at least 60 for the sustained gate")
    if args.capture_offset_seconds is not None and not args.birefnet_model:
        parser.error("--capture-offset-seconds requires --birefnet-model")
    if args.capture_offset_seconds is not None and not 0 <= args.capture_offset_seconds < args.duration_seconds:
        parser.error("--capture-offset-seconds must be inside the measured interval")

    source = VideoReplaySource(args.video, width=args.input_width)
    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)

    import idphoto.gui as gui

    gui.open_source = lambda _spec, _size: source
    window = MainWindow("public-motion-replay", "en", (args.input_width, 720),
                        presentation_modnet=True, modnet_model=args.model,
                        modnet_provider=args.provider,
                        birefnet_model=args.birefnet_model)
    window.open_timer.stop()
    window.settings.auto_shutter = False
    if window.modnet_box is None or not window.modnet_box.isEnabled():
        source.release()
        raise RuntimeError(window.modnet_error or "MODNet preview is unavailable.")

    started_ms = time.monotonic() * 1000
    metrics = ReplayMetrics(window, source, app, started_ms, args.warmup_seconds,
                            args.duration_seconds, args.capture_offset_seconds)
    window.view.modnet_frame_painted.connect(metrics.on_painted)
    window.show()
    window.modnet_box.setChecked(True)
    if args.birefnet_model:
        if window.birefnet_box is None or not window.birefnet_box.isEnabled():
            source.release()
            raise RuntimeError(window.birefnet_error or "BiRefNet capture is unavailable.")
        window.birefnet_box.setChecked(True)

    window.capture_displayed.connect(metrics.on_capture_displayed)
    window.open_source()
    window.worker.model_snapshot.connect(metrics.on_model_result)
    window.worker.frame_ready.connect(metrics.on_processed)
    window.worker.captured.connect(metrics.on_capture_emitted)
    window.worker.model_failed.connect(metrics.on_model_failed)
    QTimer.singleShot(round(args.startup_timeout_seconds * 1000),
                      metrics.startup_timeout_reached)
    app.exec()

    end_ms = metrics.end_ms or time.monotonic() * 1000
    start_ms = metrics.start_ms
    if start_ms is None:
        start_ms = end_ms
    measured_seconds = max(0.0, (end_ms - start_ms) / 1000)
    all_results = {
        event["frame_id"]: event for event in metrics.model_events
        if _in_window(event["received_ms"], start_ms, end_ms)
    }
    face_results = {
        event["frame_id"]: event for event in metrics.model_events
        if event["face_detected"] and _in_window(event["received_ms"], start_ms, end_ms)
    }
    fully_processed_results = {
        event["frame_id"]: event for event in metrics.model_events
        if event["fully_processed"] and _in_window(event["received_ms"], start_ms, end_ms)
    }
    paint_by_id = {}
    for event in metrics.paint_events:
        if event["frame_id"] in all_results and _in_window(
                event["painted_ms"], start_ms, end_ms):
            paint_by_id.setdefault(event["frame_id"], event)
    face_paint_by_id = {
        frame_id: paint for frame_id, paint in paint_by_id.items() if frame_id in face_results
    }
    fully_processed_paint_by_id = {
        frame_id: paint for frame_id, paint in paint_by_id.items()
        if frame_id in fully_processed_results
    }
    source_to_paint_ages = [
        event["painted_ms"] - all_results[frame_id]["source_timestamp_ms"]
        for frame_id, event in paint_by_id.items()
    ]
    face_source_to_paint_ages = [
        event["painted_ms"] - face_results[frame_id]["source_timestamp_ms"]
        for frame_id, event in face_paint_by_id.items()
    ]
    fully_processed_source_to_paint_ages = [
        event["painted_ms"] - fully_processed_results[frame_id]["source_timestamp_ms"]
        for frame_id, event in fully_processed_paint_by_id.items()
    ]
    result_to_paint_ages = [
        event["painted_ms"] - all_results[frame_id]["received_ms"]
        for frame_id, event in paint_by_id.items()
    ]
    source_to_result_ages = [
        event["received_ms"] - event["source_timestamp_ms"] for event in all_results.values()
    ]
    non_inference_ages = [
        event["received_ms"] - event["source_timestamp_ms"] - event["inference_ms"]
        for event in all_results.values() if event["inference_ms"] is not None
    ]
    stage_ages = {
        "source_read_to_prepare": [
            event["prepare_finished_ms"] - event["source_timestamp_ms"]
            for event in all_results.values() if event["prepare_finished_ms"] is not None
        ],
        "prepare_to_submit": [
            event["model_job_timing_ms"]["submitted"] - event["prepare_finished_ms"]
            for event in all_results.values()
            if event["prepare_finished_ms"] is not None and event["model_job_timing_ms"]
        ],
        "model_queue_wait": [
            event["model_job_timing_ms"]["started"] - event["model_job_timing_ms"]["submitted"]
            for event in all_results.values() if event["model_job_timing_ms"]
        ],
        "inference": [
            event["inference_finished_ms"] - event["inference_started_ms"]
            for event in all_results.values()
            if event["inference_started_ms"] is not None
            and event["inference_finished_ms"] is not None
        ],
        "inference_to_job_finish": [
            event["model_job_timing_ms"]["finished"] - event["inference_finished_ms"]
            for event in all_results.values()
            if event["inference_finished_ms"] is not None and event["model_job_timing_ms"]
        ],
        "job_finish_to_finalize": [
            event["finalized_ms"] - event["model_job_timing_ms"]["finished"]
            for event in all_results.values()
            if event["finalized_ms"] is not None and event["model_job_timing_ms"]
        ],
        "finalize_to_paint": [
            paint_by_id[frame_id]["painted_ms"] - event["finalized_ms"]
            for frame_id, event in all_results.items()
            if frame_id in paint_by_id and event["finalized_ms"] is not None
        ],
    }
    read_ids = {
        event["frame_id"] for event in source.read_events
        if _in_window(event["timestamp_ms"], start_ms, end_ms)
    }
    processed_ids = {
        event["frame_id"] for event in metrics.processed_events
        if _in_window(event["source_timestamp_ms"], start_ms, end_ms)
    }
    all_result_ids = set(all_results)
    face_result_ids = set(face_results)
    fully_processed_ids = set(fully_processed_results)
    all_painted_ids = set(paint_by_id)
    face_painted_ids = set(face_paint_by_id)
    fully_processed_painted_ids = set(fully_processed_paint_by_id)
    results_per_window = []
    complete_windows = int(measured_seconds // 10)
    for offset in range(0, complete_windows * 10, 10):
        a = start_ms + offset * 1000
        b = a + 10_000
        fresh_all = {
            event["frame_id"] for event in metrics.model_events
            if _in_window(event["received_ms"], a, b)
        }
        fresh_face = {
            event["frame_id"] for event in metrics.model_events
            if event["face_detected"] and _in_window(event["received_ms"], a, b)
        }
        fresh_fully_processed = {
            event["frame_id"] for event in metrics.model_events
            if event["fully_processed"] and _in_window(event["received_ms"], a, b)
        }
        painted_all = {
            event["frame_id"] for event in metrics.paint_events
            if event["frame_id"] in fresh_all and _in_window(event["painted_ms"], a, b)
        }
        painted_face = {
            frame_id for frame_id in painted_all if frame_id in fresh_face
        }
        painted_fully_processed = {
            frame_id for frame_id in painted_all if frame_id in fresh_fully_processed
        }
        results_per_window.append({
            "start_offset_s": offset,
            "duration_s": 10,
            "fresh_results_hz": len(fresh_all) / 10,
            "fresh_results_with_face_hz": len(fresh_face) / 10,
            "unique_fresh_main_view_paints_hz": len(painted_all) / 10,
            "fresh_face_matched_paints_hz": len(painted_face) / 10,
            "fully_processed_results_hz": len(fresh_fully_processed) / 10,
            "fully_processed_main_view_paints_hz": len(painted_fully_processed) / 10,
            "fresh_results_rate_pass": len(fresh_all) / 10 >= 24,
            "fresh_face_results_rate_pass": len(fresh_face) / 10 >= 24,
            "fresh_paints_rate_pass": len(painted_all) / 10 >= 24,
            "fresh_face_paints_rate_pass": len(painted_face) / 10 >= 24,
            "fully_processed_results_rate_pass": len(fresh_fully_processed) / 10 >= 24,
            "fully_processed_paints_rate_pass": len(painted_fully_processed) / 10 >= 24,
        })

    all_results_in_interval = list(all_results.values())
    first_face_result = next((event for event in metrics.model_events
                              if event["face_detected"]), None)
    first_face_paint = next((event for event in metrics.paint_events
                             if event["frame_id"] in {
                                 result["frame_id"] for result in metrics.model_events
                                 if result["face_detected"]
                             }), None)
    observed_widths = sorted({event["input_width"] for event in all_results.values()
                              if event["input_width"] is not None})
    observed_heights = sorted({event["input_height"] for event in all_results.values()
                               if event["input_height"] is not None})
    clip_by_id = {event["frame_id"]: event["clip_name"] for event in source.read_events}
    per_clip = []
    for path in args.video:
        clip_name = Path(path).name
        clip_ids = {frame_id for frame_id, name in clip_by_id.items() if name == clip_name}
        processed_clip_ids = clip_ids & processed_ids
        result_clip_ids = clip_ids & all_result_ids
        face_clip_ids = clip_ids & face_result_ids
        fully_processed_clip_ids = clip_ids & fully_processed_ids
        painted_clip_ids = clip_ids & all_painted_ids
        face_painted_clip_ids = clip_ids & face_painted_ids
        fully_processed_clip_painted_ids = clip_ids & fully_processed_painted_ids
        per_clip.append({
            "clip": clip_name,
            "source_frames_read_in_measurement": len(clip_ids & read_ids),
            "worker_frames_processed": len(processed_clip_ids),
            "unique_fresh_results": len(result_clip_ids),
            "fresh_results_with_face": len(face_clip_ids),
            "fully_processed_results": len(fully_processed_clip_ids),
            "unique_fresh_main_view_paints": len(painted_clip_ids),
            "fresh_face_matched_paints": len(face_painted_clip_ids),
            "fully_processed_main_view_paints": len(fully_processed_clip_painted_ids),
            "longest_fresh_paint_gap_ms": _longest_gap_ms(
                paint["painted_ms"] for frame_id, paint in paint_by_id.items()
                if clip_by_id.get(frame_id) == clip_name
            ),
        })
    gate_summary = _summarize_gates(
        measured_seconds,
        results_per_window,
        {
            "fresh_results": len(all_result_ids) / max(measured_seconds, 1e-9),
            "fresh_face_results": len(face_result_ids) / max(measured_seconds, 1e-9),
            "fresh_paints": len(all_painted_ids) / max(measured_seconds, 1e-9),
            "fresh_face_paints": len(face_painted_ids) / max(measured_seconds, 1e-9),
            "fully_processed_results": len(fully_processed_ids) / max(measured_seconds, 1e-9),
            "fully_processed_paints":
                len(fully_processed_painted_ids) / max(measured_seconds, 1e-9),
        },
        {
            "any_paint": source_to_paint_ages,
            "face_paint": face_source_to_paint_ages,
            "fully_processed_paint": fully_processed_source_to_paint_ages,
        },
        source_eof_before_end=(
            source.eof_timestamp_ms is not None
            and metrics.end_ms is not None
            and source.eof_timestamp_ms < metrics.end_ms
        ),
    )
    capture_validation = None
    if args.capture_offset_seconds is not None:
        capture_start_ms = metrics.capture_requested_ms
        capture_end_ms = metrics.capture_completed_ms or end_ms
        source_read_request_ms = (
            metrics.capture_results[0].get("manual_request_timestamp_ms")
            if len(metrics.capture_results) == 1 else None
        )
        if capture_start_ms is None:
            interval = {
                "duration_ms": None,
                "fully_processed_results": 0,
                "unique_fully_processed_paints": 0,
                "fully_processed_paint_rate_hz": None,
                "fully_processed_paint_rate_pass": False,
                "source_read_to_paint": _quantiles([]),
                "source_read_to_paint_sample_count": 0,
                "source_read_to_paint_p95_pass": False,
                "source_read_timestamp_gate_applied": False,
                "source_read_event_missing_count": 0,
                "pre_request_source_read_excluded_count": 0,
                "source_read_events_linked": False,
                "longest_fresh_paint_gap_ms_including_request_and_result": None,
            }
        else:
            interval = _capture_interval_summary(
                capture_start_ms, capture_end_ms,
                metrics.model_events, metrics.paint_events,
                source_read_events=getattr(metrics.source, "read_events", None),
                request_ms=source_read_request_ms,
            )
        capture_validation = {
            "requested": True,
            "trigger_offset_s": (
                capture_start_ms - start_ms
            ) / 1000 if capture_start_ms is not None and start_ms is not None else None,
            "capture_result_count": metrics.capture_result_count,
            "capture_display_count": metrics.capture_display_count,
            "capture_result_received": metrics.capture_completed_ms is not None,
            "capture_signal_received_ms": metrics.capture_signal_received_ms,
            "capture_result_latency_ms": (
                metrics.capture_completed_ms - metrics.capture_requested_ms
                if metrics.capture_completed_ms is not None
                and metrics.capture_requested_ms is not None else None
            ),
            "capture_interval_absolute_monotonic_ms": {
                "measurement_start": metrics.start_ms,
                "shutter_button_requested": metrics.capture_requested_ms,
                "capture_signal_received": metrics.capture_signal_received_ms,
                "result_dialog_opened": metrics.capture_completed_ms,
                "measurement_end": metrics.end_ms,
            },
            "capture_interval_absolute_utc": {
                "measurement_started": metrics.measurement_started_utc,
                "shutter_button_requested": metrics.capture_requested_utc,
                "result_dialog_opened": metrics.capture_completed_utc,
                "measurement_ended": metrics.measurement_ended_utc,
            },
            "source_read_gate_manual_request_timestamp_ms": source_read_request_ms,
            "capture_results": metrics.capture_results,
            "capture_error": metrics.capture_error,
            **interval,
        }
        capture_validation["gate"] = _capture_gate(capture_validation)
    report = {
        "source": "public VideoMatte240K test foreground clips; local paced replay, no frames saved",
        "clips": [{"name": Path(path).name, "sha256": _sha256(path)} for path in args.video],
        "input_width": args.input_width,
        "live_width": config.LIVE_WIDTH,
        "modnet_model_input_widths": observed_widths,
        "modnet_model_input_heights": observed_heights,
        "provider": args.provider,
        "modnet_execution_providers": sorted({
            event["execution_provider"] for event in all_results_in_interval
            if event["execution_provider"] is not None
        }),
        "provider_backends": sorted({
            backend
            for event in all_results_in_interval
            for backend in (event["provider_backends"] or [])
        }),
        "measurement_seconds": measured_seconds,
        "input_frame_rate_hz": 30,
        "startup_ms_to_first_result": (
            metrics.model_events[0]["received_ms"] - started_ms if metrics.model_events else None
        ),
        "startup_ms_to_first_face_result": (
            first_face_result["received_ms"] - started_ms if first_face_result else None
        ),
        "startup_ms_to_first_face_paint": (
            first_face_paint["painted_ms"] - started_ms if first_face_paint else None
        ),
        "warmup_seconds_after_first_face_result_is_painted": args.warmup_seconds,
        "startup_timeout": metrics.startup_timeout,
        "source_eof_timestamp_ms": source.eof_timestamp_ms,
        "source_eof_before_measurement_end":
            gate_summary["source_eof_before_measurement_end"],
        "per_clip": per_clip,
        "counts": {
            "source_frames_read": len(read_ids),
            "worker_frames_processed": len(processed_ids),
            "unique_fresh_modnet_results": len(all_results_in_interval),
            "unique_fresh_main_view_paints": len(all_painted_ids),
            "fresh_results_with_face": len(face_result_ids),
            "fresh_face_matched_main_view_paints": len(face_painted_ids),
            "fully_processed_results": len(fully_processed_ids),
            "fully_processed_main_view_paints": len(fully_processed_painted_ids),
            "landmark_sources": {
                source: sum(event["landmark_source"] == source
                            for event in all_results_in_interval)
                for source in sorted({event["landmark_source"] for event in all_results_in_interval})
            },
            "track_break_results": sum(event["track_break"] for event in all_results_in_interval),
            "landmark_detector_calls": sum(
                event["landmark_detector_calls"] for event in all_results_in_interval
            ),
            "fresh_modnet_results_not_painted": len(all_result_ids - all_painted_ids),
            "fresh_face_results_not_painted": len(face_result_ids - face_painted_ids),
            "source_frames_not_processed": len(read_ids - processed_ids),
            "worker_frames_without_fresh_model_result": len(processed_ids - all_result_ids),
            "no_face_model_results": sum(not e["face_detected"] for e in all_results_in_interval),
        },
        "rates_hz": {
            "unique_fresh_modnet_results": (
                len(all_result_ids) / measured_seconds if measured_seconds else 0
            ),
            "fresh_results_with_face": (
                len(face_result_ids) / measured_seconds if measured_seconds else 0
            ),
            "unique_fresh_main_view_paints": (
                len(all_painted_ids) / measured_seconds if measured_seconds else 0
            ),
            "fresh_face_matched_main_view_paints": (
                len(face_painted_ids) / measured_seconds if measured_seconds else 0
            ),
            "fully_processed_results": (
                len(fully_processed_ids) / measured_seconds if measured_seconds else 0
            ),
            "fully_processed_main_view_paints": (
                len(fully_processed_painted_ids) / measured_seconds if measured_seconds else 0
            ),
        },
        "source_read_to_result_age": _quantiles(source_to_result_ages),
        "source_read_to_result_minus_inference": _quantiles(non_inference_ages),
        "stage_latency_ms": {name: _quantiles(values) for name, values in stage_ages.items()},
        "modnet_wall_estimate_ms": _quantiles([
            event["wall_estimate_ms"] for event in all_results_in_interval
            if event["wall_estimate_ms"] is not None
        ]),
        "modnet_composite_ms": _quantiles([
            event["composite_ms"] for event in all_results_in_interval
            if event["composite_ms"] is not None
        ]),
        "modnet_inference_time": _quantiles([
            event["inference_ms"] for event in all_results_in_interval
            if event["inference_ms"] is not None
        ]),
        "result_receipt_to_paint_age": _quantiles(result_to_paint_ages),
        "source_read_to_any_fresh_paint_age": _quantiles(source_to_paint_ages),
        "source_read_to_face_matched_paint_age": _quantiles(face_source_to_paint_ages),
        "source_read_to_fully_processed_paint_age":
            _quantiles(fully_processed_source_to_paint_ages),
        "process_clock": _process_clock_report(metrics.process_clock_samples),
        "longest_fresh_paint_gap_ms": _longest_gap_ms(
            paint["painted_ms"] for paint in paint_by_id.values()
        ),
        "capture_requested": args.capture_offset_seconds is not None,
        "capture_validation": capture_validation,
        "sustained_10_second_windows": results_per_window,
        "gate": gate_summary,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "source_frame_id", "clip", "source_timestamp_ms", "result_received_ms", "face_detected",
            "input_width", "input_height", "inference_ms", "prepare_finished_ms",
            "model_job_submitted_ms", "model_job_started_ms", "inference_started_ms",
            "inference_finished_ms", "model_job_finished_ms", "finalized_ms",
            "wall_estimate_ms", "composite_ms", "execution_provider", "provider_backends",
            "landmark_source",
            "track_break", "light_valid", "face_mask_valid", "geometry_valid", "rule_count",
            "fully_processed", "landmark_detector_calls", "painted_ms",
            "source_read_to_result_age_ms", "result_to_paint_age_ms",
            "source_read_to_paint_age_ms", "source_read_to_prepare_ms",
            "prepare_to_submit_ms", "model_queue_wait_ms",
            "inference_to_job_finish_ms", "job_finish_to_finalize_ms",
            "finalize_to_paint_ms",
            "pipeline_landmarks_ms", "pipeline_relight_ms", "pipeline_matting_ms",
            "pipeline_geometry_ms", "pipeline_checker_ms",
        ])
        writer.writeheader()
        for event in all_results_in_interval:
            painted = paint_by_id.get(event["frame_id"])
            job_timing = event["model_job_timing_ms"] or {}
            writer.writerow({
                "source_frame_id": event["frame_id"],
                "clip": clip_by_id.get(event["frame_id"], ""),
                "source_timestamp_ms": event["source_timestamp_ms"],
                "result_received_ms": event["received_ms"],
                "face_detected": event["face_detected"],
                "input_width": event["input_width"],
                "input_height": event["input_height"],
                "inference_ms": event["inference_ms"],
                "prepare_finished_ms": event["prepare_finished_ms"],
                "model_job_submitted_ms": job_timing.get("submitted", ""),
                "model_job_started_ms": job_timing.get("started", ""),
                "inference_started_ms": event["inference_started_ms"],
                "inference_finished_ms": event["inference_finished_ms"],
                "model_job_finished_ms": job_timing.get("finished", ""),
                "finalized_ms": event["finalized_ms"],
                "wall_estimate_ms": event["wall_estimate_ms"],
                "composite_ms": event["composite_ms"],
                "execution_provider": event["execution_provider"],
                "provider_backends": ";".join(event["provider_backends"] or []),
                "landmark_source": event["landmark_source"],
                "track_break": event["track_break"],
                "light_valid": event["light_valid"],
                "face_mask_valid": event["face_mask_valid"],
                "geometry_valid": event["geometry_valid"],
                "rule_count": event["rule_count"],
                "fully_processed": event["fully_processed"],
                "landmark_detector_calls": event["landmark_detector_calls"],
                "painted_ms": painted["painted_ms"] if painted else "",
                "source_read_to_result_age_ms": event["received_ms"] - event["source_timestamp_ms"],
                "result_to_paint_age_ms": (
                    painted["painted_ms"] - event["received_ms"] if painted else ""
                ),
                "source_read_to_paint_age_ms": (
                    painted["painted_ms"] - event["source_timestamp_ms"] if painted else ""
                ),
                "source_read_to_prepare_ms": (
                    event["prepare_finished_ms"] - event["source_timestamp_ms"]
                    if event["prepare_finished_ms"] is not None else ""
                ),
                "prepare_to_submit_ms": (
                    job_timing["submitted"] - event["prepare_finished_ms"]
                    if event["prepare_finished_ms"] is not None and job_timing else ""
                ),
                "model_queue_wait_ms": (
                    job_timing["started"] - job_timing["submitted"] if job_timing else ""
                ),
                "inference_to_job_finish_ms": (
                    job_timing["finished"] - event["inference_finished_ms"]
                    if event["inference_finished_ms"] is not None and job_timing else ""
                ),
                "job_finish_to_finalize_ms": (
                    event["finalized_ms"] - job_timing["finished"]
                    if event["finalized_ms"] is not None and job_timing else ""
                ),
                "finalize_to_paint_ms": (
                    painted["painted_ms"] - event["finalized_ms"]
                    if painted and event["finalized_ms"] is not None else ""
                ),
                **_pipeline_timing_columns(event),
            })
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out} and {csv_path}")
    return _cli_exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
