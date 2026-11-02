"""Module A: frame sources, with the webcam's exposure and white-balance lock (plan B.2).

Auto exposure and white balance would shift brightness and color between frames and make
the relit output flicker, so the camera settles for about 1.5 s and then both are locked.
OpenCV's exposure flag differs by backend (V4L2: 1 manual / 3 auto; DirectShow and Media
Foundation: 0.25 / 0.75), so each backend gets its own values and the result is read back.
OpenCV's AVFoundation backend (macOS) offers no exposure control; there the lock is reported
as unavailable and the light fit is averaged over more frames instead (C.4 fallback).
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


class Camera:
    """A webcam. On macOS, create it on the main thread: only there can OpenCV ask the
    user for camera access (the first attempt shows the request and fails; retry after it)."""

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
        self.opened_at = time.monotonic()
        self.lock_state = "warming"  # warming -> locked | unsupported | failed
        self.name = f"camera {index} ({self.backend})"

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None, None
        if self.lock_state == "warming" and time.monotonic() - self.opened_at >= config.WARMUP_S:
            self.lock_state = self.lock()
        return np.ascontiguousarray(bgr[:, :, ::-1]), None

    def lock(self) -> str:
        """Lock exposure and white balance at their settled values; returns the lock state."""
        cv2, cap = self._cv2, self.cap
        flags = _EXPOSURE_FLAGS.get(self.backend)
        if flags is None:
            return "unsupported"
        exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        temperature = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
        ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, flags[0])
        if exposure:
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if temperature > 0:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, temperature)
        ok = ok and abs(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) - flags[0]) < 1e-3
        return "locked" if ok else "failed"

    def release(self):
        self.cap.release()


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
    return Camera(int(spec or 0), camera_size)


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
                frame, landmarks = self.source.read()
            except Exception as exc:  # camera unplugged, file error
                self.error = exc
                break
            if frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._seq += 1
                self._latest = (self._seq, frame, landmarks, int(time.monotonic() * 1000))

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
