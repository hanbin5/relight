import numpy as np
import pytest

from idphoto.context import FrameContext
from idphoto.matting import composite
from idphoto import presentation_modnet
from tools.presentation_modnet import (
    ModnetPrediction,
    PresentationMODNet,
    composite_dense,
    estimate_local_wall,
    estimate_wall_color,
    estimate_walls,
    model_input_size,
)
from tools.bench_modnet_live import (
    ReplayMetrics,
    _capture_gate,
    _capture_interval_summary,
    _cli_exit_code,
    _process_clock_report,
    _pipeline_timing_columns,
    _remaining_delay_ms,
    _same_timestamp,
    _summarize_gates,
)
from idphoto.presentation_modnet import estimate_walls, estimate_walls_live


def test_capture_metrics_link_model_worker_and_ctx_to_the_button_read_boundary():
    from types import SimpleNamespace

    source = SimpleNamespace(read_events=[{
        "frame_id": 7,
        "timestamp_ms": 200.0,
        "clip_name": "fgr-0000.mp4",
    }])
    metrics = ReplayMetrics(None, source, None, 0, 3, 80, 13)
    metrics.start_ms = 0.0
    metrics.capture_requested_ms = 198.0
    metrics.model_events.append({
        "frame_id": np.int64(7),
        "source_timestamp_ms": np.float64(200),
        "received_ms": np.float64(200.5),
    })
    metrics.processed_events.append({
        "frame_id": np.int64(7),
        "source_timestamp_ms": np.float64(200),
        "received_ms": np.float64(200.6),
    })
    result = {
        "report": {
            "settings": {"capture_matte": "birefnet-portrait"},
            "rules": {},
            "geometry": {"crop_boxes": {"passport": [1, 2, 3, 4]}},
        },
        "ctx": SimpleNamespace(
            raw_frame=np.zeros((4, 4, 3), np.uint8),
            timestamp_ms=200.0,
            stats={
                "birefnet_portrait": {
                    "provider": "CPUExecutionProvider",
                    "source_frame_id": np.int64(7),
                    "source_timestamp_ms": np.float64(200),
                    "source_read_timestamp_ms": np.float64(200),
                    "manual_request_timestamp_ms": np.float64(199),
                },
                "capture_source": {
                    "frame_id": np.int64(7),
                    "timestamp_ms": np.float64(200),
                    "source_read_timestamp_ms": np.float64(200),
                    "manual_request_timestamp_ms": np.float64(199),
                },
                "capture_timing_ms": {},
            },
        ),
    }
    guard = {
        "model": "birefnet",
        "frame_id": np.int64(7),
        "timestamp_ms": np.float64(200),
        "source_read_timestamp_ms": np.float64(200),
        "manual_request_timestamp_ms": np.float64(199),
        "model_job_timing_ms": {},
        "capture_timing_ms": {"result_dialog_opened_ms": 220.0},
        "capture_worker_input": {
            "frame_id": np.int64(7),
            "timestamp_ms": np.float64(200),
            "source_read_timestamp_ms": np.float64(200),
            "manual_request_timestamp_ms": np.float64(199),
            "started_ms": np.float64(201),
        },
        "result_dialog_is_expected_type": True,
        "result_dialog_visible": True,
    }
    guard["model_job_timing_ms"] = {
        "submitted": np.float64(200.8),
        "started": np.float64(201),
        "finished": np.float64(210),
    }

    metrics.on_capture_displayed(result, guard)
    assert metrics.capture_results[0]["source_linked"] is True
    assert metrics.capture_results[0]["source_read_event"]["clip_name"] == "fgr-0000.mp4"

    guard["manual_request_timestamp_ms"] = 200.0
    guard["capture_worker_input"]["manual_request_timestamp_ms"] = np.float64(200)
    result["ctx"].stats["capture_source"]["manual_request_timestamp_ms"] = np.float64(200)
    result["ctx"].stats["birefnet_portrait"]["manual_request_timestamp_ms"] = np.float64(200)
    metrics.on_capture_displayed(result, guard)
    assert metrics.capture_results[1]["source_linked"] is False


def test_modnet_dimensions_are_onnx_stride_aligned_for_arbitrary_widths():
    for height, width, cap in ((360, 413, 448), (720, 853, 1024), (480, 640, 448)):
        model_width, model_height = model_input_size(height, width, cap)
        assert model_width % 32 == 0
        assert model_height % 32 == 0
        assert model_width <= width


def test_modnet_raw_prediction_is_frame_matched_and_composable():
    class Session:
        def run(self, _outputs, _inputs):
            return [np.full((1, 1, 18, 32), 0.75, np.float32)]

        def get_providers(self):
            return ["CPUExecutionProvider"]

    model = object.__new__(PresentationMODNet)
    session = Session()
    model.provider = "cpu"
    model.live_width = model.capture_width = 512
    model._session_for_grid = lambda _width, _height: (session, "input", "cpu")
    model.prepare_input = lambda _raw_rgb, capture=False: True

    rgb = np.zeros((360, 640, 3), np.uint8)
    prediction = model.predict_alpha(rgb)
    assert isinstance(prediction, ModnetPrediction)
    assert prediction.stats["input_width"] == 512
    assert prediction.stats["input_height"] == 288
    assert prediction.stats["provider"] == "cpu"
    assert prediction.alpha.shape == (360, 640)
    assert not prediction.alpha.flags.writeable

    ctx = FrameContext(raw_frame=rgb)
    ctx.linear_rgb = np.full((360, 640, 3), 0.5, np.float32)
    ctx.relit_frame = ctx.linear_rgb.copy()
    model.apply_prediction(ctx, prediction, np.ones(3, np.float32))
    np.testing.assert_array_equal(ctx.alpha, prediction.alpha)
    assert ctx.output_frame.shape == ctx.relit_frame.shape


def test_modnet_prepares_only_the_first_observed_grid():
    class Session:
        def __init__(self):
            self.input_shapes = []

        def run(self, _outputs, inputs):
            value = next(iter(inputs.values()))
            self.input_shapes.append(value.shape)
            return [np.zeros((1, 1, 11, 16), np.float32)]

    session = Session()
    model = object.__new__(PresentationMODNet)
    model.live_width = model.capture_width = 512
    model.prepared_grids = set()
    model._session_for_grid = lambda width, height: (session, "input", "cpu")
    rgb = np.zeros((427, 640, 3), np.uint8)
    assert model_input_size(427, 640, 512) == (512, 352)
    assert model.prepare_input(rgb) is False
    assert model.prepare_input(rgb) is True
    assert session.input_shapes == [(1, 3, 352, 512)]


def test_sustained_gate_rejects_a_bad_ten_second_window():
    windows = []
    for index in range(6):
        windows.append({
            "fresh_results_rate_pass": True,
            "fresh_face_results_rate_pass": index != 2,
            "fresh_paints_rate_pass": True,
            "fresh_face_paints_rate_pass": index != 2,
            "fully_processed_results_rate_pass": True,
            "fully_processed_paints_rate_pass": True,
        })
    rates = {name: 26.0 for name in (
        "fresh_results", "fresh_face_results", "fresh_paints", "fresh_face_paints",
        "fully_processed_results", "fully_processed_paints",
    )}
    result = _summarize_gates(60.2, windows, rates, {
        "any_paint": [80.0] * 100,
        "face_paint": [80.0] * 100,
        "fully_processed_paint": [80.0] * 100,
    })
    assert result["at_least_24_fresh_face_matched_main_view_paints_per_second"] is False
    assert result["overall_pass"] is False


def test_sustained_gate_checks_seven_windows_for_longer_runs():
    windows = []
    for _ in range(7):
        windows.append({
            "fresh_results_rate_pass": True,
            "fresh_face_results_rate_pass": True,
            "fresh_paints_rate_pass": True,
            "fresh_face_paints_rate_pass": True,
            "fully_processed_results_rate_pass": True,
            "fully_processed_paints_rate_pass": True,
        })
    rates = {name: 26.0 for name in (
        "fresh_results", "fresh_face_results", "fresh_paints", "fresh_face_paints",
        "fully_processed_results", "fully_processed_paints",
    )}
    result = _summarize_gates(70.2, windows, rates, {
        "any_paint": [80.0] * 100,
        "face_paint": [80.0] * 100,
        "fully_processed_paint": [80.0] * 100,
    })
    assert result["duration_at_least_60s_with_complete_10s_windows"] is True
    assert result["overall_pass"] is True


def test_sustained_gate_and_cli_fail_when_source_reaches_eof_early():
    windows = [{
        "fresh_results_rate_pass": True,
        "fresh_face_results_rate_pass": True,
        "fresh_paints_rate_pass": True,
        "fresh_face_paints_rate_pass": True,
        "fully_processed_results_rate_pass": True,
        "fully_processed_paints_rate_pass": True,
    } for _ in range(6)]
    rates = {name: 26.0 for name in (
        "fresh_results", "fresh_face_results", "fresh_paints", "fresh_face_paints",
        "fully_processed_results", "fully_processed_paints",
    )}
    ages = {
        "any_paint": [80.0] * 100,
        "face_paint": [80.0] * 100,
        "fully_processed_paint": [80.0] * 100,
    }
    passed = _summarize_gates(60.2, windows, rates, ages)
    failed = _summarize_gates(
        60.2, windows, rates, ages, source_eof_before_end=True
    )
    assert passed["overall_pass"] is True
    assert _cli_exit_code({"gate": passed}) == 0
    assert failed["source_eof_before_measurement_end"] is True
    assert failed["overall_pass"] is False
    assert _cli_exit_code({"gate": failed}) == 1
    assert _cli_exit_code({"gate": {"overall_pass": False}}) == 1


def test_capture_timing_boundaries_use_absolute_precision_and_never_fire_early():
    deadline_ms = 174_059_788_113.458
    assert _remaining_delay_ms(deadline_ms, deadline_ms - 0.1) == 1
    assert _remaining_delay_ms(deadline_ms, deadline_ms) == 0
    assert _remaining_delay_ms(deadline_ms, deadline_ms + 0.1) == 0
    assert _same_timestamp(deadline_ms, deadline_ms + 0.005)
    assert not _same_timestamp(deadline_ms, deadline_ms + 0.02)


def test_process_clock_report_tracks_aggregate_cpu_and_dialog_visibility():
    samples = [
        {
            "event": "measurement_start", "observed_monotonic_ms": 1000,
            "process_cpu_seconds": 10, "dialog_visible": False,
        },
        {
            "event": "result_dialog_opened", "observed_monotonic_ms": 21000,
            "process_cpu_seconds": 31, "dialog_visible": True,
        },
        {
            "event": "window_boundary", "observed_monotonic_ms": 30000,
            "process_cpu_seconds": 41, "dialog_visible": True,
        },
        {
            "event": "measurement_end", "observed_monotonic_ms": 81000,
            "process_cpu_seconds": 100, "dialog_visible": True,
        },
    ]
    report = _process_clock_report(samples)
    assert report["intervals"][0]["process_cpu_to_wall_ratio"] == pytest.approx(1.05)
    assert report["dialog_visible_at_all_post_open_samples"] is True
    assert report["post_open_sample_count"] == 2
    samples[-1]["dialog_visible"] = False
    assert _process_clock_report(samples)[
        "dialog_visible_at_all_post_open_samples"
    ] is False


def test_pipeline_stage_timing_columns_preserve_available_worker_timings():
    assert _pipeline_timing_columns({
        "pipeline_timings_ms": {
            "landmarks": 1.0,
            "relight": 2.0,
            "matting": 3.0,
            "geometry": 4.0,
            "checker": 5.0,
        }
    }) == {
        "pipeline_landmarks_ms": 1.0,
        "pipeline_relight_ms": 2.0,
        "pipeline_matting_ms": 3.0,
        "pipeline_geometry_ms": 4.0,
        "pipeline_checker_ms": 5.0,
    }


@pytest.mark.parametrize("layout", ["clear", "opaque", "strided", "transposed"])
def test_live_wall_estimator_is_bit_exact_to_capture_baseline(layout):
    rng = np.random.default_rng(20261127)
    rgb = rng.random((37, 45, 3), dtype=np.float32)
    alpha = rng.random((37, 45), dtype=np.float32)
    if layout == "clear":
        alpha.fill(0)
    elif layout == "opaque":
        alpha.fill(1)
    elif layout == "strided":
        rgb, alpha = rgb[::2, 1::2], alpha[::2, 1::2]
    elif layout == "transposed":
        rgb, alpha = rgb.transpose(1, 0, 2), alpha.T

    reference = estimate_walls(rgb, alpha)
    live = estimate_walls_live(rgb, alpha)
    for expected, actual in zip(reference, live):
        assert actual.dtype == expected.dtype == np.float32
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_capture_interval_counts_unique_fully_processed_paints_with_endpoints():
    model_events = [
            {"frame_id": 1, "source_timestamp_ms": 110, "received_ms": 120,
             "fully_processed": True},
            {"frame_id": 2, "source_timestamp_ms": 140, "received_ms": 150,
             "fully_processed": True},
            {"frame_id": 3, "source_timestamp_ms": 190, "received_ms": 200,
             "fully_processed": True},
        ]
    paint_events = [
            {"frame_id": 1, "painted_ms": 130},
            {"frame_id": 1, "painted_ms": 131},
            {"frame_id": 2, "painted_ms": 170},
            {"frame_id": 3, "painted_ms": 175},
        ]
    source_read_events = [
        {"frame_id": 1, "timestamp_ms": 110},
        {"frame_id": 2, "timestamp_ms": 140},
        {"frame_id": 3, "timestamp_ms": 190},
    ]
    summary = _capture_interval_summary(
        100, 180, model_events, paint_events,
        source_read_events=source_read_events, request_ms=100,
    )
    assert summary["duration_ms"] == 80
    assert summary["fully_processed_results"] == 2
    assert summary["unique_fully_processed_paints"] == 2
    assert summary["fully_processed_paint_rate_hz"] == 25
    assert summary["fully_processed_paint_rate_pass"] is True
    assert summary["source_read_to_paint"]["p95_ms"] == 29.5
    assert summary["source_read_to_paint_sample_count"] == 2
    assert summary["source_read_to_paint_p95_pass"] is True
    assert summary["source_read_events_linked"] is True
    assert summary["longest_fresh_paint_gap_ms_including_request_and_result"] == 40
    no_paints = _capture_interval_summary(100, 1100, [], [])
    assert no_paints["fully_processed_paint_rate_hz"] == 0
    assert no_paints["fully_processed_paint_rate_pass"] is False
    assert no_paints["source_read_to_paint_p95_pass"] is False

    stale_boundary = _capture_interval_summary(
        100,
        1100,
        [
            {"frame_id": 0, "source_timestamp_ms": 99, "received_ms": 120,
             "fully_processed": True},
            {"frame_id": 1, "source_timestamp_ms": 100, "received_ms": 121,
             "fully_processed": True},
            *[
                {
                    "frame_id": frame_id,
                    "source_timestamp_ms": 101 + frame_id - 2,
                    "received_ms": 170 + frame_id - 2,
                    "fully_processed": True,
                }
                for frame_id in range(2, 25)
            ],
        ],
        [
            {"frame_id": 0, "painted_ms": 140},
            {"frame_id": 1, "painted_ms": 141},
            *[
                {"frame_id": frame_id, "painted_ms": 191 + frame_id - 2}
                for frame_id in range(2, 25)
            ],
        ],
        source_read_events=[
            {"frame_id": 0, "timestamp_ms": 99},
            {"frame_id": 1, "timestamp_ms": np.float64(100)},
            *[
                {"frame_id": frame_id, "timestamp_ms": np.int64(101 + frame_id - 2)}
                for frame_id in range(2, 25)
            ],
        ],
        request_ms=np.int64(100),
    )
    assert stale_boundary["pre_request_source_read_excluded_count"] == 2
    assert stale_boundary["fully_processed_results"] == 23
    assert stale_boundary["unique_fully_processed_paints"] == 23
    assert stale_boundary["fully_processed_paint_rate_hz"] == 23
    assert stale_boundary["fully_processed_paint_rate_pass"] is False
    assert stale_boundary["source_read_to_paint_sample_count"] == 23
    assert stale_boundary["source_read_to_paint"]["p95_ms"] == 90
    assert stale_boundary["source_read_events_linked"] is True

    actual_worker_request = _capture_interval_summary(
        100, 1100,
        [
            {"frame_id": 1, "source_timestamp_ms": 102, "received_ms": 120,
             "fully_processed": True},
            {"frame_id": 2, "source_timestamp_ms": 104, "received_ms": 130,
             "fully_processed": True},
            {"frame_id": 3, "source_timestamp_ms": 105, "received_ms": 140,
             "fully_processed": True},
        ],
        [
            {"frame_id": 1, "painted_ms": 150},
            {"frame_id": 2, "painted_ms": 160},
            {"frame_id": 3, "painted_ms": 170},
        ],
        source_read_events=[
            {"frame_id": 1, "timestamp_ms": 102},
            {"frame_id": 2, "timestamp_ms": 104},
            {"frame_id": 3, "timestamp_ms": 105},
        ],
        request_ms=104,
    )
    assert actual_worker_request["pre_request_source_read_excluded_count"] == 2
    assert actual_worker_request["fully_processed_results"] == 1
    assert actual_worker_request["unique_fully_processed_paints"] == 1

    capture = {
        "capture_result_count": 1,
        "capture_display_count": 1,
        "capture_results": [{
            "model": "birefnet",
            "capture_matte": "birefnet-portrait",
            "model_stats": {
                "provider": "CPUExecutionProvider",
                "inference_ms": 120,
                "inference_count": 1,
                "session_create_ms": 1,
                "preprocess_ms": 2,
                "alpha_postprocess_ms": 3,
                "total_predict_ms": 126,
                "source_frame_id": np.int64(7),
                "source_timestamp_ms": 70.0,
                "source_read_timestamp_ms": np.float64(70),
                "manual_request_timestamp_ms": np.int64(60),
            },
            "source_frame_id": 7,
            "source_timestamp_ms": 70.0,
            "source_linked": True,
            "model_source_frame_id": np.int64(7),
            "model_source_timestamp_ms": np.float64(70),
            "returned_ctx_source_frame_id": 7,
            "returned_ctx_source_timestamp_ms": np.float64(70),
            "returned_ctx_timestamp_ms": np.float64(70),
            "capture_requested_timestamp_ms": 50.0,
            "manual_request_timestamp_ms": np.float64(60),
            "source_read_timestamp_ms": 70,
            "worker_guard_manual_request_timestamp_ms": 60.0,
            "worker_guard_source_read_timestamp_ms": np.float64(70),
            "source_read_event": {"frame_id": 7, "timestamp_ms": np.int64(70)},
            "modnet_model_snapshot_event": {
                "frame_id": np.int64(7),
                "source_timestamp_ms": 70.0,
                "received_ms": 75.0,
            },
            "modnet_worker_frame_ready_event": {
                "frame_id": 7,
                "source_timestamp_ms": np.float64(70),
                "received_ms": 76.0,
            },
            "capture_worker_input": {
                "frame_id": 7,
                "timestamp_ms": np.float64(70),
                "source_read_timestamp_ms": 70.0,
                "manual_request_timestamp_ms": np.int64(60),
                "started_ms": 1001.0,
            },
            "crop_boxes": {"passport": [1, 2, 3, 4]},
            "model_job_timing_ms": {
                "submitted": 1000, "started": 1001, "finished": 1310,
            },
            "capture_timing_ms": {
                "worker_result_callback_ms": 1311,
                "worker_guard_checked_ms": 1312,
                "worker_signal_emitted_ms": 1313,
                "gui_result_received_ms": 1314,
                "gui_guard_passed_ms": 1315,
                "result_dialog_opened_ms": 1316,
            },
            "result_dialog_is_expected_type": True,
            "result_dialog_visible": True,
            "pipeline_capture_timing_ms": {
                "render_final_started_monotonic_ms": 1010,
                "render_final_finished_monotonic_ms": 1300,
                "frame_decode_relight_ms": 10,
                "matte_apply_call_ms": 150,
                "checker_wall_estimate_ms": 2,
                "checker_ms": 5,
                "composite_ms": 50,
                "crop_photo_sheet_render_ms": 20,
                "render_final_total_ms": 290,
            },
            "dialog_opened": True,
        }],
        "capture_error": None,
        **summary,
    }
    capture["gate"] = _capture_gate(capture)
    passing = {"gate": {"overall_pass": True}, "capture_requested": True,
               "capture_validation": capture}
    no_result = {"gate": {"overall_pass": True}, "capture_requested": True,
                 "capture_validation": None}
    legacy_capture = {
        "gate": {"overall_pass": True},
        "capture_validation": {"capture_result_received": True},
    }
    duplicate = {**capture, "capture_result_count": 2}
    duplicate["gate"] = _capture_gate(duplicate)
    no_dialog = {
        **capture,
        "capture_display_count": 0,
        "capture_results": [{**capture["capture_results"][0], "dialog_opened": False}],
    }
    no_dialog["gate"] = _capture_gate(no_dialog)
    insufficient_rate = {**capture, "fully_processed_paint_rate_pass": False}
    insufficient_rate["gate"] = _capture_gate(insufficient_rate)
    zero_paints = {**capture, **no_paints}
    zero_paints["gate"] = _capture_gate(zero_paints)
    slow_paint = {**capture, "source_read_to_paint_p95_pass": False}
    slow_paint["gate"] = _capture_gate(slow_paint)
    missing_capture_evidence = {
        **capture,
        "capture_results": [{
            **capture["capture_results"][0],
            "model_stats": None,
            "source_frame_id": None,
            "source_timestamp_ms": None,
            "crop_boxes": None,
        }],
    }
    missing_capture_evidence["gate"] = _capture_gate(missing_capture_evidence)
    numpy_numeric_evidence = {
        **capture,
        "capture_results": [{
            **capture["capture_results"][0],
            "source_frame_id": np.int64(7),
            "source_timestamp_ms": np.float64(70),
        }],
    }
    numpy_numeric_evidence["gate"] = _capture_gate(numpy_numeric_evidence)
    boolean_source_id = {
        **capture,
        "capture_results": [{
            **capture["capture_results"][0],
            "source_frame_id": True,
        }],
    }
    boolean_source_id["gate"] = _capture_gate(boolean_source_id)
    capture_error = {**capture, "capture_error": "capture failed"}
    capture_error["gate"] = _capture_gate(capture_error)
    missing_timing = {
        **capture,
        "capture_results": [{
            **capture["capture_results"][0],
            "pipeline_capture_timing_ms": None,
        }],
    }
    missing_timing["gate"] = _capture_gate(missing_timing)

    assert _cli_exit_code(passing) == 0
    assert numpy_numeric_evidence["gate"]["overall_pass"] is True
    assert boolean_source_id["gate"]["overall_pass"] is False
    assert capture_error["gate"]["overall_pass"] is False
    assert missing_timing["gate"]["overall_pass"] is False
    assert _cli_exit_code(no_result) == 1
    assert _cli_exit_code(legacy_capture) == 1
    assert _cli_exit_code({"gate": {"overall_pass": True}}) == 0
    for rejected in (
        duplicate, no_dialog, insufficient_rate, zero_paints, slow_paint,
        missing_capture_evidence,
    ):
        assert rejected["gate"]["overall_pass"] is False
        assert _cli_exit_code({
            "gate": {"overall_pass": True},
            "capture_requested": True,
            "capture_validation": rejected,
        }) == 1


def test_coreml_static_rejects_session_without_active_coreml_provider(monkeypatch, tmp_path):
    from types import SimpleNamespace

    class SessionOptions:
        def __init__(self):
            self.overrides = {}

        def add_free_dimension_override_by_name(self, name, value):
            self.overrides[name] = value

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="rgb", shape=["batch", 3, "height", "width"])]

        def get_providers(self):
            return ["CPUExecutionProvider"]

    fake_runtime = SimpleNamespace(
        __version__="test",
        SessionOptions=SessionOptions,
        InferenceSession=lambda *_args, **_kwargs: Session(),
    )
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_runtime)
    monkeypatch.setattr(presentation_modnet.Path, "home", lambda: tmp_path)
    model = object.__new__(PresentationMODNet)
    model.provider = "coreml-static"
    model.model_path = tmp_path / "model.onnx"
    model.static_sessions = {}
    model.input_names = {}
    model.prepared_grids = set()
    with pytest.raises(RuntimeError, match="did not activate CoreMLExecutionProvider"):
        model._session_for_grid(512, 288)


def test_modnet_local_wall_uses_nearby_confident_background_and_global_fallback():
    linear_rgb = np.zeros((24, 40, 3), np.float32)
    linear_rgb[:, :20] = (0.1, 0.2, 0.3)
    linear_rgb[:, 20:] = (0.8, 0.7, 0.6)
    alpha = np.zeros((24, 40), np.float32)
    alpha[:, 19:21] = 1.0
    fallback = estimate_wall_color(linear_rgb, alpha)
    field = estimate_local_wall(linear_rgb, alpha, fallback)

    np.testing.assert_allclose(field[12, 5], (0.1, 0.2, 0.3), atol=1e-6)
    np.testing.assert_allclose(field[12, 34], (0.8, 0.7, 0.6), atol=1e-6)
    np.testing.assert_allclose(fallback, (0.45, 0.45, 0.45), atol=1e-5)

    no_background = estimate_local_wall(linear_rgb, np.ones_like(alpha), fallback)
    np.testing.assert_allclose(
        no_background, np.broadcast_to(fallback, no_background.shape), atol=1e-6
    )


def test_fused_wall_estimate_matches_separate_global_and_local_estimates():
    rng = np.random.default_rng(42)
    linear_rgb = rng.random((37, 53, 3), dtype=np.float32)
    alpha = rng.random((37, 53), dtype=np.float32)
    for matte in (alpha, np.ones_like(alpha)):
        expected_global = estimate_wall_color(linear_rgb, matte)
        expected_local = estimate_local_wall(linear_rgb, matte, expected_global)
        actual_global, actual_local = estimate_walls(linear_rgb, matte)
        np.testing.assert_allclose(actual_global, expected_global, atol=2e-6, rtol=0)
        np.testing.assert_allclose(actual_local, expected_local, atol=2e-6, rtol=0)


def test_modnet_dense_composite_matches_standard_spill_removal():
    rng = np.random.default_rng(12)
    for spatial_wall in (False, True):
        for with_ratio, with_color in ((False, False), (True, False), (True, True)):
            alpha = rng.random((24, 32), dtype=np.float32)
            alpha[0] = 0.0
            alpha[-1] = 1.0
            ctx = FrameContext(raw_frame=np.zeros((24, 32, 3), np.uint8))
            ctx.alpha = alpha
            ctx.relit_frame = rng.random((24, 32, 3), dtype=np.float32)
            ctx.ratio_map = rng.uniform(0.65, 1.55, alpha.shape).astype(np.float32) \
                if with_ratio else None
            ctx.color_gain = rng.uniform(0.8, 1.25, (24, 32, 3)).astype(np.float32) \
                if with_color else None
            wall = rng.random((24, 32, 3), dtype=np.float32) if spatial_wall \
                else rng.random(3, dtype=np.float32)
            for background in (np.zeros(3, np.float32), np.ones(3, np.float32)):
                expected = composite(ctx, wall, background)
                actual = composite_dense(ctx, wall, background)
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
