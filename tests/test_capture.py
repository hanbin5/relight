"""Exposure / white-balance lock and its check (module A, plan B.2), with fake cameras."""

from types import SimpleNamespace

import numpy as np
import pytest

from idphoto import capture, config
from idphoto.capture import AVFoundationCamera, Camera, LockMonitor, SettleDetector, lock_outcome

# --- lock_outcome ---------------------------------------------------------------------


@pytest.mark.parametrize("exposure, white_balance, expected", [
    (True, True, "locked"),
    (True, False, "partial"),
    (True, None, "partial"),
    (False, True, "failed"),
    (False, False, "failed"),
    (None, True, "failed"),
    (False, None, "failed"),
    (None, None, "unsupported"),
])
def test_lock_outcome(exposure, white_balance, expected):
    assert lock_outcome(exposure, white_balance) == expected


# --- SettleDetector -------------------------------------------------------------------


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def frame(value: float) -> np.ndarray:
    return np.full((64, 64, 3), value, np.uint8)


def test_settle_waits_for_the_minimum_warmup():
    clock = Clock()
    settle = SettleDetector(clock)
    for _ in range(20):  # steady from the start, but too early
        clock.t += 0.05
        assert not settle.update(frame(100))
    clock.t = config.WARMUP_S
    assert settle.update(frame(100))


def test_settle_waits_while_the_brightness_moves():
    clock = Clock()
    settle = SettleDetector(clock)
    clock.t = config.WARMUP_S + 0.5
    results = [settle.update(frame(80 + 10 * i)) for i in range(6)]  # auto exposure still ramping
    assert not any(results)
    results = [settle.update(frame(140)) for _ in range(config.WARMUP_SETTLE_FRAMES)]
    assert results[-1]  # steady again


def test_settle_gives_up_after_the_maximum():
    clock = Clock()
    settle = SettleDetector(clock)
    clock.t = config.WARMUP_MAX_S
    assert settle.update(frame(10))  # one frame, brightness never steady


# --- OpenCV camera --------------------------------------------------------------------

CV2 = SimpleNamespace(CAP_PROP_AUTO_EXPOSURE=1, CAP_PROP_EXPOSURE=2, CAP_PROP_AUTO_WB=3, CAP_PROP_WB_TEMPERATURE=4)


class FakeCap:
    """A capture whose properties can be made read-only, like a backend without the control."""

    def __init__(self, values, accepts=()):
        self.values = dict(values)
        self.accepts = set(accepts)  # properties that take a new value
        self.sets = []

    def get(self, prop):
        return self.values.get(prop, 0.0)

    def set(self, prop, value):
        self.sets.append((prop, value))
        if prop in self.accepts:
            self.values[prop] = value
            return True
        return False


def opencv_camera(backend, cap) -> Camera:
    camera = object.__new__(Camera)
    camera._cv2, camera.cap, camera.backend = CV2, cap, backend
    camera.lock_state = "warming"
    return camera


SETTLED = {CV2.CAP_PROP_EXPOSURE: -6.0, CV2.CAP_PROP_WB_TEMPERATURE: 4600.0,
           CV2.CAP_PROP_AUTO_EXPOSURE: 0.75, CV2.CAP_PROP_AUTO_WB: 1}


@pytest.mark.parametrize("backend, manual", [("DSHOW", 0.25), ("MSMF", 0.25), ("V4L2", 1)])
def test_opencv_locks_both_and_keeps_the_settled_values(backend, manual):
    cap = FakeCap(SETTLED, accepts=set(CV2.__dict__.values()))
    camera = opencv_camera(backend, cap)
    assert camera.lock() == "locked"
    assert cap.values[CV2.CAP_PROP_AUTO_EXPOSURE] == manual
    assert cap.values[CV2.CAP_PROP_AUTO_WB] == 0
    assert (CV2.CAP_PROP_EXPOSURE, -6.0) in cap.sets  # re-set after switching to manual
    assert (CV2.CAP_PROP_WB_TEMPERATURE, 4600.0) in cap.sets


def test_opencv_reports_partial_when_white_balance_is_not_accepted():
    cap = FakeCap(SETTLED, accepts={CV2.CAP_PROP_AUTO_EXPOSURE, CV2.CAP_PROP_EXPOSURE})
    assert opencv_camera("V4L2", cap).lock() == "partial"


def test_opencv_does_not_trust_a_property_the_backend_lacks():
    # get() answers 0 for a missing property, which must not read as "auto white balance is off"
    cap = FakeCap({CV2.CAP_PROP_AUTO_WB: 0}, accepts=set())
    assert opencv_camera("DSHOW", cap).lock() == "failed"


def test_opencv_reports_failed_when_exposure_does_not_stick():
    cap = FakeCap(SETTLED, accepts={CV2.CAP_PROP_AUTO_WB})
    assert opencv_camera("MSMF", cap).lock() == "failed"


def test_opencv_macos_backend_is_unsupported():
    assert opencv_camera("AVFOUNDATION", FakeCap({})).lock() == "unsupported"


def test_opencv_unlock_gives_the_controls_back():
    cap = FakeCap(SETTLED, accepts=set(CV2.__dict__.values()))
    camera = opencv_camera("V4L2", cap)
    camera.lock_state = camera.lock()
    camera.release = lambda: None
    camera._release_lock()
    assert cap.values[CV2.CAP_PROP_AUTO_EXPOSURE] == 3
    assert cap.values[CV2.CAP_PROP_AUTO_WB] == 1
    assert camera.lock_state == "locked"  # the state stays readable after release


def test_camera_locks_once_after_settling(monkeypatch):
    clock = Clock()
    camera = opencv_camera("V4L2", FakeCap(SETTLED, accepts=set(CV2.__dict__.values())))
    camera._settle = SettleDetector(clock)
    calls = []
    original = camera.lock
    camera.lock = lambda: calls.append(1) or original()
    for i in range(40):
        clock.t = i * 0.1
        camera._track(frame(120))
    assert camera.lock_state == "locked"
    assert len(calls) == 1


# --- AVFoundation camera --------------------------------------------------------------

LOCKED, AUTO = 0, 2


class FakeDevice:
    def __init__(self, exposure: bool, white_balance: bool, hold: bool = True):
        self.supports = {("exposure", LOCKED): exposure, ("exposure", AUTO): True,
                         ("wb", LOCKED): white_balance, ("wb", AUTO): white_balance}
        self.hold = hold
        self.exposure, self.wb = AUTO, AUTO
        self.configuring = False

    def isExposureModeSupported_(self, mode):
        return self.supports.get(("exposure", mode), False)

    def isWhiteBalanceModeSupported_(self, mode):
        return self.supports.get(("wb", mode), False)

    def lockForConfiguration_(self, error):
        self.configuring = True
        return True, None

    def unlockForConfiguration(self):
        self.configuring = False

    def setExposureMode_(self, mode):
        assert self.configuring, "device settings change only inside lockForConfiguration"
        self.exposure = mode if self.hold else AUTO

    def setWhiteBalanceMode_(self, mode):
        assert self.configuring
        self.wb = mode if self.hold else AUTO

    def exposureMode(self):
        return self.exposure

    def whiteBalanceMode(self):
        return self.wb


AV = SimpleNamespace(AVCaptureExposureModeLocked=LOCKED, AVCaptureExposureModeContinuousAutoExposure=AUTO,
                     AVCaptureWhiteBalanceModeLocked=LOCKED,
                     AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance=AUTO)


def av_camera(device) -> AVFoundationCamera:
    camera = object.__new__(AVFoundationCamera)
    camera._AV, camera.device = AV, device
    camera.lock_state = "warming"
    return camera


@pytest.mark.parametrize("exposure, white_balance, expected", [
    (True, True, "locked"),
    (True, False, "partial"),  # an external webcam like the Brio 300
    (False, False, "unsupported"),  # the built-in MacBook camera
])
def test_avfoundation_lock_follows_what_the_device_offers(exposure, white_balance, expected):
    device = FakeDevice(exposure, white_balance)
    assert av_camera(device).lock() == expected
    assert not device.configuring  # configuration is always released
    if exposure:
        assert device.exposure == LOCKED


def test_avfoundation_reports_failed_when_the_mode_does_not_stick():
    assert av_camera(FakeDevice(True, True, hold=False)).lock() == "failed"


def test_avfoundation_unlock_returns_to_auto():
    device = FakeDevice(True, True)
    camera = av_camera(device)
    camera.lock_state = camera.lock()
    camera._release_lock()
    assert (device.exposure, device.wb) == (AUTO, AUTO)
    assert not device.configuring


# --- CVPixelBuffer conversion (macOS) -------------------------------------------------


def test_pixel_buffer_to_rgb_reads_bgra_with_row_padding():
    Quartz = pytest.importorskip("Quartz")
    status, buffer = Quartz.CVPixelBufferCreate(None, 7, 4, Quartz.kCVPixelFormatType_32BGRA, None, None)
    assert status == 0
    Quartz.CVPixelBufferLockBaseAddress(buffer, 0)
    stride = Quartz.CVPixelBufferGetBytesPerRow(buffer)
    base = Quartz.CVPixelBufferGetBaseAddress(buffer)
    raw = np.frombuffer(base.as_buffer(stride * 4), np.uint8).reshape(4, stride // 4, 4)
    raw[:] = 0
    raw[1, 2] = (10, 20, 30, 255)  # B, G, R, A
    Quartz.CVPixelBufferUnlockBaseAddress(buffer, 0)
    rgb = capture.pixel_buffer_to_rgb(buffer)
    assert rgb.shape == (4, 7, 3) and rgb.dtype == np.uint8 and rgb.flags.c_contiguous
    assert tuple(rgb[1, 2]) == (30, 20, 10)
    assert rgb.sum() == 60


# --- LockMonitor ----------------------------------------------------------------------


def face(shift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    lm = np.zeros((478, 3), np.float32)
    lm[:, :2] = rng.uniform(100, 300, (478, 2)) + shift
    return lm


def feed(monitor, brightness, seconds=1.2, fps=30, landmarks=None, start=0):
    state = monitor.state
    for i in range(int(seconds * fps)):
        b = brightness(i) if callable(brightness) else brightness
        state = monitor.update(b, face() if landmarks is None else landmarks, start + int(1000 * i / fps))
    return state


def test_monitor_is_steady_when_brightness_holds():
    assert feed(LockMonitor(), 0.40) == "steady"


def test_monitor_flags_drifting_brightness():
    assert feed(LockMonitor(), lambda i: 0.40 + 0.002 * i) == "drifting"  # 0.40 -> 0.47 in 1.2 s


def test_monitor_ignores_a_moving_head():
    monitor = LockMonitor()
    state = "checking"
    for i in range(40):  # head moves every frame: the hold-still window never fills
        state = monitor.update(0.4 + 0.02 * i, face(shift=3.0 * i), 33 * i)
    assert state == "checking"


def test_monitor_resets_without_a_face():
    monitor = LockMonitor()
    feed(monitor, 0.4, seconds=0.6)
    monitor.update(float("nan"), None, 1000)
    assert monitor.window == []
