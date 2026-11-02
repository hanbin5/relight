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
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import color, config, geometry
from .capture import DemoSource, FrameGrabber, LockMonitor, camera_permission, is_camera, open_source
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


# ---------------------------------------------------------------------------------------
# Processing thread


class Worker(QThread):
    frame_ready = Signal(object)
    captured = Signal(object)
    failed = Signal(str)

    def __init__(self, source, settings: Settings):
        super().__init__()
        self.source = source  # opened on the main thread (see MainWindow.open_source)
        self.settings = settings
        self.running = True
        self.view_before = False
        self.mirror = True
        self.shutter_request = False
        self.rearm_request = False

    def run(self):
        source = self.source
        try:
            detector = None
            if not isinstance(source, DemoSource):
                from .landmarks import FaceLandmarkDetector

                detector = FaceLandmarkDetector()
        except Exception as exc:
            source.release()
            self.failed.emit(str(exc))
            return
        grabber = FrameGrabber(source)
        grabber.start()
        pipeline = Pipeline(detector)
        monitor = LockMonitor()
        seq, fps, last = 0, 0.0, time.perf_counter()
        while self.running:
            item = grabber.latest(seq)
            if item is None:
                if grabber.error is not None:
                    self.failed.emit(str(grabber.error))
                    break
                self.msleep(3)
                continue
            seq, frame, lm_full, ts = item
            live, lm_live = to_live(frame, lm_full)
            if self.rearm_request:
                pipeline.shutter.rearm()
                self.rearm_request = False
            ctx, fired = pipeline.process(live, self.settings, landmarks=lm_live, timestamp_ms=ts, full_frame=frame)

            lock = getattr(source, "lock_state", "locked")
            if lock == "locked" and ctx.face_mask is not None and ctx.face_mask.any():
                if monitor.update(float(ctx.Y[ctx.face_mask].mean()), ctx.landmarks, ts) == "drifting":
                    lock = "drifting"
            frames = config.SMOOTH_FRAMES_FALLBACK if lock in FALLBACK_LOCKS else config.SMOOTH_FRAMES
            if pipeline.relighter.history.maxlen != frames:
                pipeline.set_light_smoothing(frames)

            if (fired or self.shutter_request) and ctx.landmarks is not None and ctx.light_coeffs is not None:
                self.shutter_request = False
                pipeline.shutter.armed = False
                try:
                    self.captured.emit(pipeline.render_final(frame, ctx, self.settings))
                except Exception as exc:  # keep the live view running
                    self.failed.emit(f"capture failed: {exc}")
            self.shutter_request = False

            now = time.perf_counter()
            dt = max(now - last, 1e-3)
            fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
            last = now
            shown = ctx.raw_frame if self.view_before else color.encode_u8(ctx.output_frame)
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
                "source": getattr(source, "name", ""),
            })
        grabber.stop()
        if detector is not None:
            detector.close()


# ---------------------------------------------------------------------------------------
# Live view


class LiveView(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.info = None
        self.lang = "ko"
        self.before = False
        self.mirror = True
        self.message = None
        self.flash = 0.0
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

        geo = self.info.get("geometry")
        if geo and not self.before:
            self._draw_guide(p, geo, to_view)
        if not self.info.get("face"):
            self._banner(p, t("no_face", self.lang) + " · " + t("msg_no_face", self.lang))
        self._pill(p, QPointF(18, 16), t("pill_before" if self.before else "pill_relit", self.lang), dark=not self.before)
        self._pill(p, QPointF(self.width() - 18, 16), t("hold_b", self.lang), dark=False, right=True)
        if self.message:
            self._banner(p, self.message, bottom=True)
        if self.flash > 0:
            p.fillRect(self.rect(), QColor(255, 255, 255, int(220 * self.flash)))

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
    def __init__(self, source, lang: str, camera_size):
        super().__init__()
        self.lang = lang
        self.settings = Settings()
        self.dialog = None
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
        self.worker = None
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
        self.worker = Worker(source, self.settings)
        self.worker.mirror = self.view.mirror
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.captured.connect(self.on_captured)
        self.worker.failed.connect(self.on_failed)
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

        self.strength_label = self._section(v)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(int(round(self.settings.strength * 100)))
        self.strength_slider.valueChanged.connect(lambda val: setattr(self.settings, "strength", val / 100))
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
        self.settings.format = key
        self.head_slider.blockSignals(True)
        self.head_slider.setValue(int(round(self.settings.head_share[key] * 100)))
        self.head_slider.blockSignals(False)
        self._update_head_label()

    def set_head_share(self, value: int):
        self.settings.head_share[self.settings.format] = value / 100
        self._update_head_label()

    def _update_head_label(self):
        pct = int(round(self.settings.head_share[self.settings.format] * 100))
        self.head_value.setText(f"{pct}%")

    def set_background(self, key: str):
        self.settings.background = key
        self.bg_name.setText(t(f"bg_{key}", self.lang))

    def set_auto(self, on: bool):
        self.settings.auto_shutter = on
        if on and self.worker:
            self.worker.rearm_request = True

    def set_mirror(self, on: bool):
        if self.worker:
            self.worker.mirror = on
        self.view.mirror = on

    def set_before(self, on: bool):
        if self.worker:
            self.worker.view_before = on
        self.view.before = on
        self.view.update()

    def take_photo(self):
        if self.dialog is None and self.worker:
            self.worker.shutter_request = True

    def toggle_language(self):
        self.lang = "en" if self.lang == "ko" else "ko"
        self.view.lang = self.lang
        self.retranslate()

    def retranslate(self):
        L = self.lang
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
    def on_frame(self, info: dict):
        L = self.lang
        info["format"] = self.settings.format
        info["head_pct"] = int(round(self.settings.head_share[self.settings.format] * 100))
        self.view.message = None
        geo = info.get("geometry")
        if geo is not None and not geo.get("crop_fits", True):
            self.view.message = t("msg_move_back", L)
        self.view.set_frame(info)
        self.strength_value.setText(f"{self.strength_slider.value()}%")

        verdicts = info["verdicts"] or {}
        first_fail = None
        for n in range(1, 8):
            v = verdicts.get(n)
            name = t(f"rule_{v.key}", L) if v else ""
            if v and n == 1 and v.value is not None and math.isfinite(v.value):
                name += f" (R_LR {v.value:.2f})"
            self.rule_names[n].setText(f"{n} · {name}")
            ok = bool(v and v.passed)
            self.rule_marks[n].setText("✓" if ok else "✕")
            self.rule_marks[n].setStyleSheet(f"color: {'#2a7a3b' if ok else '#c0392b'}; font-weight: 600;")
            if v and not v.passed and first_fail is None:
                first_fail = v
        if first_fail is not None:
            name = t(f"rule_{first_fail.key}", L)
            self.coach.setText(f"<b>{name}:</b> {t('msg_' + (first_fail.message or 'no_face'), L)}")
            self.coach.setProperty("ok", False)
        else:
            self.coach.setText(f"<b>{t('all_pass', L)}</b>")
            self.coach.setProperty("ok", True)
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
        self.status.setText(f"{info['source']} · {info['fps']:.0f} fps{rlr}\n{t(lock_key, L)}")

    def on_captured(self, result: dict):
        self.view.flash = 1.0
        self.flash_timer.start(30)
        self.dialog = ResultDialog(self, result, self.lang)
        self.dialog.finished.connect(self._dialog_closed)
        self.dialog.open()

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
            self.worker.wait(3000)
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
    parser.add_argument("--snapshot", default=None, help=argparse.SUPPRESS)  # save a window screenshot and quit
    parser.add_argument("--snapshot-after", type=float, default=6.0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    lang = args.lang or ("ko" if QApplication.instance() and _system_is_korean() else "en")
    size = tuple(int(x) for x in args.camera_size.lower().split("x"))
    window = MainWindow(args.source, lang, size)
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
