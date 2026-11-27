"""Module A: frame sources, with the webcam's exposure and white-balance lock (plan B.2).

Auto exposure and white balance would shift brightness and color between frames and make
the relit output flicker, so the camera settles (at least 1.5 s, and until the mean
brightness is steady) and then both are locked. The lock state is one of:

    warming      still settling
    locked       exposure and white balance both locked and read back
    partial      exposure locked, white balance could not be (many external webcams on macOS)
    unsupported  the camera or backend has no lock; the light fit is averaged over more frames
    failed       a lock was tried but did not hold; same fallback (C.4)

Two capture paths exist. OpenCV (Windows, Linux, and macOS without pyobjc): its exposure
flag differs by backend (V4L2: 1 manual / 3 auto; DirectShow and Media Foundation:
0.25 / 0.75), so each backend gets its own values and the result is read back. On macOS,
OpenCV's AVFoundation backend has no exposure control, so AVFoundationCamera captures through
AVFoundation itself and locks with AVCaptureDevice (what the device supports: the built-in
MacBook camera supports neither lock, and a Brio 300 supports the exposure lock only).
LockMonitor then checks the lock: face brightness must stay steady while the user holds still.

Every source returns (rgb uint8, landmarks or None); OpenCV is used here only for I/O.
"""

import platform
import threading
import time
from pathlib import Path

import numpy as np

from . import config, synthetic

# Manual / auto values of CAP_PROP_AUTO_EXPOSURE per backend.
_EXPOSURE_FLAGS = {"V4L2": (1, 3), "DSHOW": (0.25, 0.75), "MSMF": (0.25, 0.75)}


def camera_permission() -> str:
    """This process's macOS camera authorization: authorized, not_determined, denied,
    restricted, or unknown (other systems, or the check failed). Reading it shows no prompt."""
    if platform.system() != "Darwin":
        return "unknown"
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        av = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
        objc.objc_getClass.restype = objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        status = send(objc.objc_getClass(b"AVCaptureDevice"),
                      objc.sel_registerName(b"authorizationStatusForMediaType:"),
                      ctypes.c_void_p.in_dll(av, "AVMediaTypeVideo").value)
    except Exception:
        return "unknown"
    return {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}.get(status, "unknown")


def lock_outcome(exposure: bool | None, white_balance: bool | None) -> str:
    """Lock state from what each control did. None: the camera has no such control;
    True: it was set and read back as locked; False: it was set and did not hold.
    Exposure is what matters most for flicker, so white balance alone is not enough."""
    if exposure is None and white_balance is None:
        return "unsupported"
    if not exposure:
        return "failed"
    return "locked" if white_balance else "partial"


class SettleDetector:
    """Decides when the camera's auto exposure has settled and the lock may be set: at least
    WARMUP_S have passed and the last few frames' mean brightness is steady, or WARMUP_MAX_S
    have passed whatever the brightness does."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.start = clock()
        self.recent: list = []

    def update(self, rgb: np.ndarray) -> bool:
        elapsed = self.clock() - self.start
        self.recent = (self.recent + [float(rgb[::8, ::8].mean())])[-config.WARMUP_SETTLE_FRAMES :]
        if elapsed >= config.WARMUP_MAX_S:
            return True
        if elapsed < config.WARMUP_S or len(self.recent) < config.WARMUP_SETTLE_FRAMES:
            return False
        spread = (max(self.recent) - min(self.recent)) / max(float(np.mean(self.recent)), 1.0)
        return spread <= config.WARMUP_SETTLE_DRIFT


class _LockingCamera:
    """What both webcam classes share: warm up, lock once, unlock when released."""

    lock_state = "warming"  # warming -> locked | partial | unsupported | failed

    def _track(self, rgb: np.ndarray):
        if self.lock_state == "warming" and self._settle.update(rgb):
            self.lock_state = self.lock()

    def lock(self) -> str:
        raise NotImplementedError

    def unlock(self):
        """Give exposure and white balance back to the camera."""

    def _release_lock(self):
        if self.lock_state in ("locked", "partial"):
            try:
                self.unlock()
            except Exception:  # the camera is going away anyway
                pass


class Camera(_LockingCamera):
    """A webcam through OpenCV. On macOS, create it on the main thread: only there can OpenCV
    ask the user for camera access (the first attempt shows the request and fails; retry)."""

    def __init__(self, index: int = 0, size=config.CAMERA_SIZE):
        import cv2

        self._cv2 = cv2
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        apis = {"Darwin": [cv2.CAP_AVFOUNDATION], "Windows": [cv2.CAP_DSHOW, cv2.CAP_MSMF],
                "Linux": [cv2.CAP_V4L2, cv2.CAP_ANY]}.get(platform.system(), [cv2.CAP_ANY])
        for api in apis:
            cap = cv2.VideoCapture(index, api)
            if cap.isOpened():
                break
        if not cap.isOpened():
            raise RuntimeError(f"Camera {index} is not available (check the camera permission).")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap = cap
        self.backend = cap.getBackendName().upper()
        self._settle = SettleDetector()
        self.name = f"camera {index} ({self.backend})"

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None, None
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        self._track(rgb)
        return rgb, None

    def lock(self) -> str:
        """Lock exposure and white balance at their settled values; returns the lock state."""
        cv2, cap = self._cv2, self.cap
        flags = _EXPOSURE_FLAGS.get(self.backend)
        if flags is None:
            return lock_outcome(None, None)
        exposure = cap.get(cv2.CAP_PROP_EXPOSURE)  # read both before leaving auto mode
        temperature = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
        # cap.get() returns 0 for a property the backend lacks, so a lock counts only when
        # the set call was accepted and the value reads back.
        exposure_ok = bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, flags[0])) \
            and abs(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) - flags[0]) < 1e-3
        if exposure_ok and exposure:
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)  # switching to manual may reset the value
        wb_ok = bool(cap.set(cv2.CAP_PROP_AUTO_WB, 0)) and cap.get(cv2.CAP_PROP_AUTO_WB) == 0
        if wb_ok and temperature > 0:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, temperature)
        return lock_outcome(exposure_ok, wb_ok)

    def unlock(self):
        flags = _EXPOSURE_FLAGS.get(self.backend)
        if flags is not None:
            self.cap.set(self._cv2.CAP_PROP_AUTO_EXPOSURE, flags[1])
            self.cap.set(self._cv2.CAP_PROP_AUTO_WB, 1)

    def release(self):
        self._release_lock()
        self.cap.release()


def pixel_buffer_to_rgb(pixel_buffer) -> np.ndarray:
    """A 32BGRA CVPixelBuffer as an RGB uint8 (H, W, 3) copy."""
    import Quartz

    read_only = getattr(Quartz, "kCVPixelBufferLock_ReadOnly", 1)
    Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, read_only)
    try:
        width = Quartz.CVPixelBufferGetWidth(pixel_buffer)
        height = Quartz.CVPixelBufferGetHeight(pixel_buffer)
        stride = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)
        base = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
        raw = np.frombuffer(base.as_buffer(stride * height), np.uint8).reshape(height, stride // 4, 4)
        pixels = raw[:, :width]  # drop the row padding
        try:
            import cv2

            return cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGB)  # B, G, R, A -> R, G, B (copy), ~15x faster
        except ImportError:
            return np.ascontiguousarray(pixels[:, :, 2::-1])
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, read_only)


_frame_delegate_class = None


def _get_frame_delegate_class():
    """The Objective-C object AVFoundation calls with each frame (made once, on first use)."""
    global _frame_delegate_class
    if _frame_delegate_class is None:
        import objc
        from Foundation import NSObject

        class IdphotoFrameDelegate(NSObject):
            def initWithCallback_(self, callback):
                self = objc.super(IdphotoFrameDelegate, self).init()
                if self is not None:
                    self._callback = callback
                return self

            def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample_buffer, connection):
                self._callback(sample_buffer)

        _frame_delegate_class = IdphotoFrameDelegate
    return _frame_delegate_class


# Session presets by width, smallest first (AVFoundation only offers fixed sizes).
_PRESET_NAMES = [(640, "AVCaptureSessionPreset640x480"), (960, "AVCaptureSessionPreset960x540"),
                 (1280, "AVCaptureSessionPreset1280x720"), (1920, "AVCaptureSessionPreset1920x1080")]


class AVFoundationCamera(_LockingCamera):
    """A webcam through AVFoundation (macOS, needs pyobjc), so exposure and white balance can
    be locked with AVCaptureDevice, which OpenCV's backend cannot do. Create it on the main
    thread: if camera access is not decided yet, this asks for it and raises; retry after."""

    backend = "AVFOUNDATION"

    def __init__(self, index: int = 0, size=config.CAMERA_SIZE):
        import AVFoundation as AV
        import dispatch
        import Quartz

        self._AV = AV
        status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
        if status == AV.AVAuthorizationStatusNotDetermined:
            AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AV.AVMediaTypeVideo, lambda granted: None)
            raise RuntimeError("Camera access was requested: allow it, then try again.")
        if status != AV.AVAuthorizationStatusAuthorized:
            raise RuntimeError(f"Camera {index} is not available (camera access is denied).")
        devices = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
        if not 0 <= index < len(devices):
            raise RuntimeError(f"Camera {index} is not available ({len(devices)} found).")
        self.device = devices[index]

        self._cond = threading.Condition()
        self._frame, self._fresh = None, False
        self.session = AV.AVCaptureSession.alloc().init()
        for width, name in _PRESET_NAMES:
            preset = getattr(AV, name, None)
            if width >= size[0] and preset is not None and self.session.canSetSessionPreset_(preset):
                self.session.setSessionPreset_(preset)
                break
        camera_input, error = AV.AVCaptureDeviceInput.deviceInputWithDevice_error_(self.device, None)
        if camera_input is None:
            raise RuntimeError(f"Camera {index} could not be opened: {error}")
        output = AV.AVCaptureVideoDataOutput.alloc().init()
        output.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA})
        output.setAlwaysDiscardsLateVideoFrames_(True)
        self._delegate = _get_frame_delegate_class().alloc().initWithCallback_(self._on_sample)
        self._queue = dispatch.dispatch_queue_create(b"idphoto.camera", None)
        output.setSampleBufferDelegate_queue_(self._delegate, self._queue)
        if not (self.session.canAddInput_(camera_input) and self.session.canAddOutput_(output)):
            raise RuntimeError(f"Camera {index} could not be set up.")
        self.session.addInput_(camera_input)
        self.session.addOutput_(output)
        self.session.startRunning()
        self._settle = SettleDetector()
        self.name = f"{self.device.localizedName()} (AVFoundation)"

    def _on_sample(self, sample_buffer):
        import CoreMedia

        try:
            rgb = pixel_buffer_to_rgb(CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer))
        except Exception:  # a bad frame is skipped, the stream goes on
            return
        with self._cond:
            self._frame, self._fresh = rgb, True
            self._cond.notify()

    def read(self):
        with self._cond:
            if not self._cond.wait_for(lambda: self._fresh, timeout=1.0):
                return None, None
            rgb, self._fresh = self._frame, False
        self._track(rgb)
        return rgb, None

    def capabilities(self) -> dict:
        """Which locks this camera offers (for `tools/check_lock.py`)."""
        AV, dev = self._AV, self.device
        return {"exposure_lock": bool(dev.isExposureModeSupported_(AV.AVCaptureExposureModeLocked)),
                "white_balance_lock": bool(dev.isWhiteBalanceModeSupported_(AV.AVCaptureWhiteBalanceModeLocked))}

    def lock(self) -> str:
        """Lock exposure and white balance at their settled values; returns the lock state."""
        AV, dev = self._AV, self.device
        can = self.capabilities()
        if not any(can.values()):
            return lock_outcome(None, None)
        done, _ = dev.lockForConfiguration_(None)
        if not done:
            return "failed"
        try:
            if can["exposure_lock"]:
                dev.setExposureMode_(AV.AVCaptureExposureModeLocked)
            if can["white_balance_lock"]:
                dev.setWhiteBalanceMode_(AV.AVCaptureWhiteBalanceModeLocked)
        finally:
            dev.unlockForConfiguration()
        exposure = dev.exposureMode() == AV.AVCaptureExposureModeLocked if can["exposure_lock"] else None
        white_balance = dev.whiteBalanceMode() == AV.AVCaptureWhiteBalanceModeLocked if can["white_balance_lock"] else None
        return lock_outcome(exposure, white_balance)

    def unlock(self):
        AV, dev = self._AV, self.device
        done, _ = dev.lockForConfiguration_(None)
        if not done:
            return
        try:
            if dev.isExposureModeSupported_(AV.AVCaptureExposureModeContinuousAutoExposure):
                dev.setExposureMode_(AV.AVCaptureExposureModeContinuousAutoExposure)
            if dev.isWhiteBalanceModeSupported_(AV.AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance):
                dev.setWhiteBalanceMode_(AV.AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance)
        finally:
            dev.unlockForConfiguration()

    def release(self):
        self._release_lock()
        self.session.stopRunning()


def open_camera(index: int = 0, size=config.CAMERA_SIZE):
    """The best webcam class for this machine: AVFoundation on macOS when pyobjc is there
    (it can lock the exposure), OpenCV otherwise."""
    if platform.system() == "Darwin":
        try:
            import AVFoundation  # noqa: F401
            import dispatch  # noqa: F401
            import Quartz  # noqa: F401
        except ImportError:
            pass
        else:
            return AVFoundationCamera(index, size)
    return Camera(index, size)


class ImageSource:
    """A still photo served as a live stream (testing without a camera)."""

    def __init__(self, path):
        import cv2

        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        self.frame = np.ascontiguousarray(bgr[:, :, ::-1])
        self.lock_state = "locked"
        self.name = Path(path).name

    def read(self):
        time.sleep(1 / 30)
        return self.frame, None

    def release(self):
        pass


class VideoSource:
    """A video file, looped."""

    def __init__(self, path):
        import cv2

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.delay = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.lock_state = "locked"
        self.name = Path(path).name

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = self.cap.read()
            if not ok:
                return None, None
        time.sleep(self.delay)
        return np.ascontiguousarray(bgr[:, :, ::-1]), None

    def release(self):
        self.cap.release()


class DemoSource:
    """A synthetic sitter by a side window, with exact landmarks (no camera or detector needed).

    The loop shows the checker at work: the head turns, straightens, blinks once and then
    holds still, so the auto shutter fires. Frames are rendered once and cached.
    """

    FRAMES = 48

    def __init__(self, size=(960, 540)):
        self.size = size
        self.cache: dict = {}
        self.i = 0
        self.lock_state = "locked"
        self.name = "demo (synthetic face)"

    def scene(self, i: int) -> synthetic.Scene:
        t = i / self.FRAMES
        yaw = 14.0 * max(0.0, 1.0 - t / 0.35) * np.sin(2 * np.pi * t / 0.35)
        return synthetic.Scene(
            width=self.size[0], height=self.size[1], face_height=0.40, center=(0.5, 0.5),
            yaw=float(yaw), roll=float(3.0 * max(0.0, 1.0 - t / 0.3)),
            light=(-0.55, 0.2, 0.8), ambient=0.25, eyes_open=not (0.45 <= t < 0.5), seed=i,
        )

    def read(self):
        i = self.i % self.FRAMES
        self.i += 1
        if i not in self.cache:
            rgb, lm, _ = synthetic.render(self.scene(i))
            self.cache[i] = (rgb, lm)
        else:
            time.sleep(1 / 15)
        return self.cache[i]

    def release(self):
        pass


def is_camera(spec: str | None) -> bool:
    return spec is None or str(spec).isdigit()


def open_source(spec: str | None, camera_size=config.CAMERA_SIZE):
    """'demo', a file path (image or video) or a camera index (default 0)."""
    if spec == "demo":
        return DemoSource()
    if spec and not spec.isdigit():
        path = Path(spec)
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            return ImageSource(path)
        return VideoSource(path)
    return open_camera(int(spec or 0), camera_size)


class FrameGrabber(threading.Thread):
    """Reads frames continuously so processing always gets the newest one (no stale buffer)."""

    def __init__(self, source):
        super().__init__(daemon=True)
        self.source = source
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self.running = True
        self.error = None

    def run(self):
        while self.running:
            try:
                read_result = self.source.read()
            except Exception as exc:  # camera unplugged, file error
                self.error = exc
                break
            if len(read_result) == 3:
                frame, landmarks, read_timestamp_ms = read_result
            else:
                frame, landmarks = read_result
                read_timestamp_ms = int(time.monotonic() * 1000)
            if frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._seq += 1
                self._latest = (self._seq, frame, landmarks, read_timestamp_ms)

    def latest(self, after_seq: int = 0):
        """(seq, frame, landmarks, timestamp_ms) newer than after_seq, or None."""
        with self._lock:
            if self._latest is None or self._latest[0] <= after_seq:
                return None
            return self._latest

    def stop(self):
        self.running = False
        self.join(timeout=2.0)
        self.source.release()


class LockMonitor:
    """Checks the exposure lock: face brightness must stay steady while the head holds still."""

    def __init__(self):
        self.window: list = []
        self.state = "checking"  # checking -> steady | drifting

    def update(self, face_brightness: float, landmarks: np.ndarray | None, timestamp_ms: int) -> str:
        if landmarks is None or not np.isfinite(face_brightness):
            self.window.clear()
            return self.state
        pts = landmarks[:468, :2]
        if self.window:
            moved = float(np.abs(pts - self.window[-1][1]).mean())
            face_width = float(np.ptp(pts[:, 0]))
            if moved > 0.01 * face_width:  # head moved: restart the hold-still window
                self.window.clear()
        self.window.append((timestamp_ms, pts.copy(), face_brightness))
        span = (self.window[-1][0] - self.window[0][0]) / 1000.0
        if span >= config.LOCK_CHECK_S:
            values = np.array([w[2] for w in self.window])
            drift = (values.max() - values.min()) / max(values.mean(), 1e-6)
            self.state = "steady" if drift <= config.LOCK_MAX_DRIFT else "drifting"
            self.window = self.window[len(self.window) // 2 :]
        return self.state
