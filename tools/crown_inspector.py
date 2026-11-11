"""Crown and crop inspector: tune the crown estimate on still photos (plan B.5, C.1).

    uv run python tools/crown_inspector.py photo1.jpg photo2.jpg
    uv run python tools/crown_inspector.py --demo        # synthetic faces, no camera or model

Left: the processed frame (not mirrored) with both crown cues, the final crown, the chin,
the eye line and the crop box. Right: sliders for k, the band in which the mask cue is
accepted and the head share, and the 300 DPI photo these settings would save.

Click the photo to mark the true crown (top of the head, hair included); right-click
clears the mark. Marks are kept in out/crown_marks.json. With marks, the panel shows the
head-length error in mm (target: within +-1 mm, plan C.1) and the k that fits the marked
photos best. Demo faces are marked automatically from the renderer's person mask.
Left and right arrow keys switch photos.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idphoto import color, config, geometry, mesh, synthetic  # noqa: E402
from idphoto.gui import to_qimage  # noqa: E402  (also clears OpenCV's Qt plugin path)
from idphoto.pipeline import Pipeline, Settings, to_live  # noqa: E402

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPen, QPixmap, QShortcut  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parents[1]
MARKS = ROOT / "out" / "crown_marks.json"
DEFAULT_K, DEFAULT_AGREE = config.CROWN_K, config.CROWN_MASK_AGREE

C_PROP, C_MASK, C_CROWN, C_CHIN, C_EYE = "#f39c12", "#8e44ad", "#e74c3c", "#27ae60", "#2f6fdf"
C_TRUTH, C_BOX, C_BOX_OUT = "#e91e8c", "#2f6fdf", "#e0712c"

DEMO_SCENES = [
    dict(face_height=0.40, center=(0.50, 0.50)),
    dict(face_height=0.32, center=(0.46, 0.48), roll=4.0),
    dict(face_height=0.46, center=(0.54, 0.56), light=(0.5, 0.3, 0.8)),
]


# ---------------------------------------------------------------------------------------
# Samples: one processed frame each


class Sample:
    def __init__(self, name: str, path: str | None = None, scene: dict | None = None):
        self.name, self.path, self.scene = name, path, scene
        self.ctx = None
        self.truth = None  # (x, y) of the true crown in live-frame pixels
        self.auto_truth = False

    @property
    def key(self) -> str:
        return str(Path(self.path).resolve()) if self.path else f"demo:{self.name}"


class Processor:
    """Runs the pipeline once per sample (5 identical frames, like tools/process_image.py)."""

    def __init__(self):
        self.detector = None
        self.timestamp = 0

    def run(self, sample: Sample):
        if sample.scene is not None:
            scene = synthetic.Scene(width=960, height=540, ambient=0.25, light=(-0.55, 0.2, 0.8),
                                    **{k: v for k, v in sample.scene.items()})
            rgb, lm, person = synthetic.render(scene)
            live, lm_live = to_live(rgb, lm)
            col = int(round(0.5 * (lm[mesh.IRIS_R, 0] + lm[mesh.IRIS_L, 0])))
            rows = np.nonzero(person[:, col])[0]
            if rows.size and sample.truth is None:
                s = live.shape[1] / rgb.shape[1]
                sample.truth, sample.auto_truth = np.array([(col + 0.5) * s, rows.min() * s], np.float32), True
            pipe, frame = Pipeline(None), rgb
        else:
            from idphoto.capture import ImageSource

            frame = ImageSource(sample.path).frame
            live, lm_live = to_live(frame)
            if self.detector is None:
                from idphoto.landmarks import FaceLandmarkDetector

                self.detector = FaceLandmarkDetector()
            pipe = Pipeline(self.detector)
        for _ in range(5):
            self.timestamp += 33
            ctx, _ = pipe.process(live, Settings(), landmarks=lm_live, timestamp_ms=self.timestamp, full_frame=frame)
        sample.ctx = ctx

    def close(self):
        if self.detector is not None:
            self.detector.close()


# ---------------------------------------------------------------------------------------
# Measurements for the current settings


def evaluate(sample: Sample, fmt_key: str, share: float, use_mask: bool) -> dict | None:
    ctx = sample.ctx
    if ctx is None or ctx.landmarks is None:
        return None
    alpha = ctx.alpha if use_mask else None
    cues = geometry.crown_cues(ctx.landmarks, alpha)
    geo = geometry.measure(ctx.landmarks, alpha)
    fmt = geometry.FORMATS[fmt_key]
    box = geometry.crop_box(geo, fmt, share)
    out = {"cues": cues, "geo": geo, "box": box, "fits": geometry.crop_fits(box, ctx.size)}
    if sample.truth is not None:
        chin, up = geo["chin"], cues["up"]
        est_len = float((geo["crown"] - chin) @ up)
        true_len = float((sample.truth - chin) @ up)
        h_crop = box[3] - box[1]
        doc_h, unit = (fmt.size_mm[1], "mm") if fmt.size_mm else (fmt.size_px[1], "px")
        out.update(crown_err_px=est_len - true_len, head_err=(est_len - true_len) / h_crop * doc_h,
                   unit=unit, k_true=true_len / cues["ec"])
    return out


def fitted_k(samples) -> tuple:
    """Least-squares k for the proportion rule over the marked samples: sum(ec d) / sum(ec^2)."""
    num = den = 0.0
    n = 0
    for s in samples:
        if s.truth is None or s.ctx is None or s.ctx.landmarks is None:
            continue
        cues = geometry.crown_cues(s.ctx.landmarks)
        d = float((s.truth - s.ctx.landmarks[mesh.CHIN, :2]) @ cues["up"])
        num, den, n = num + cues["ec"] * d, den + cues["ec"] ** 2, n + 1
    return (num / den if den else None), n


# ---------------------------------------------------------------------------------------
# Frame view


class FrameView(QWidget):
    marked = Signal(object)  # (x, y) in frame pixels, or None to clear

    def __init__(self):
        super().__init__()
        self.image = None
        self.result = None
        self.truth = None
        self.agree = DEFAULT_AGREE
        self.use_mask = True
        self.setMinimumSize(560, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _transform(self):
        iw, ih = self.image.width(), self.image.height()
        s = min(self.width() / iw, self.height() / ih)
        return s, (self.width() - iw * s) / 2, (self.height() - ih * s) / 2

    def mousePressEvent(self, event):
        if self.image is None:
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.marked.emit(None)
            return
        s, ox, oy = self._transform()
        x, y = (event.position().x() - ox) / s, (event.position().y() - oy) / s
        if 0 <= x < self.image.width() and 0 <= y < self.image.height():
            self.marked.emit(np.array([x, y], np.float32))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#2b2b2b"))
        if self.image is None:
            p.setPen(QColor("white"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No frame")
            return
        s, ox, oy = self._transform()
        p.drawImage(QRectF(ox, oy, self.image.width() * s, self.image.height() * s), self.image)
        r = self.result
        if r is None:
            self._label(p, QPointF(12, 12), "No face found", "#c0392b")
            return

        def v(pt):
            return QPointF(ox + float(pt[0]) * s, oy + float(pt[1]) * s)

        cues, geo = r["cues"], r["geo"]
        chin, up, ec, k = geo["chin"], cues["up"], cues["ec"], config.CROWN_K
        # Midline, and the band in which the mask cue is accepted.
        p.setPen(QPen(QColor(255, 255, 255, 110), 1))
        p.drawLine(v(chin), v(chin + 1.4 * k * ec * up))
        if self.use_mask:
            lo, hi = self.agree
            band = QPen(QColor(46, 204, 113, 90), 10)
            band.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(band)
            p.drawLine(v(chin + lo * k * ec * up), v(chin + hi * k * ec * up))
        # Crop box.
        x0, y0, x1, y1 = r["box"]
        p.setPen(QPen(QColor(C_BOX if r["fits"] else C_BOX_OUT), 2, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(v((x0, y0)), v((x1, y1))))
        # Eye line, chin, crown.
        left = np.array([-up[1], up[0]], np.float32)  # across the face
        half = 0.9 * ec
        eye = geo["eye_center"]
        self._cross_line(p, v(eye - half * left), v(eye + half * left), C_EYE, 1)
        self._cross_line(p, v(chin - half * left), v(chin + half * left), C_CHIN, 2, "chin")
        crown = geo["crown"]
        self._cross_line(p, v(crown - half * left), v(crown + half * left), C_CROWN, 2, "crown")
        self._marker(p, v(cues["proportion"]), C_PROP, "x")
        if cues["mask"] is not None:
            self._marker(p, v(cues["mask"]), C_MASK, "o")
        if self.truth is not None:
            t = self.truth
            p.setPen(QPen(QColor(C_TRUTH), 2, Qt.PenStyle.DashLine))
            p.drawLine(v(t - half * left), v(t + half * left))
            self._marker(p, v(t), C_TRUTH, "+")
        # Legend.
        items = [("x proportion cue", C_PROP), ("o mask cue", C_MASK), ("— crown (used)", C_CROWN),
                 ("— chin", C_CHIN), ("- - true crown (click)", C_TRUTH)]
        if self.use_mask:
            items.append(("▮ mask cue accepted here", "#2ecc71"))
        y = 10
        for text, col in items:
            y += self._label(p, QPointF(10, y), text, col) + 4

    def _cross_line(self, p, a, b, col, width, text=None):
        p.setPen(QPen(QColor(col), width))
        p.drawLine(a, b)
        if text:
            self._label(p, QPointF(b.x() + 6, b.y() - 10), text, col)

    def _marker(self, p, at, col, kind):
        p.setPen(QPen(QColor(col), 2.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        d = 7
        if kind == "x":
            p.drawLine(at + QPointF(-d, -d), at + QPointF(d, d))
            p.drawLine(at + QPointF(-d, d), at + QPointF(d, -d))
        elif kind == "+":
            p.drawLine(at + QPointF(-d, 0), at + QPointF(d, 0))
            p.drawLine(at + QPointF(0, -d), at + QPointF(0, d))
        else:
            p.drawEllipse(at, d, d)

    def _label(self, p, at, text, col) -> float:
        m = p.fontMetrics()
        rect = QRectF(at.x(), at.y(), m.horizontalAdvance(text) + 10, m.height() + 4)
        p.fillRect(rect, QColor(0, 0, 0, 160))
        p.setPen(QColor(col))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        return rect.height()


# ---------------------------------------------------------------------------------------
# Main window


class Inspector(QMainWindow):
    def __init__(self, samples):
        super().__init__()
        self.samples, self.index = samples, 0
        self.processor = Processor()
        self._load_marks()
        self.setWindowTitle("Crown and crop inspector")
        self.view = FrameView()
        self.view.marked.connect(self.set_mark)

        panel = QWidget()
        panel.setFixedWidth(380)
        v = QVBoxLayout(panel)
        v.setContentsMargins(16, 12, 16, 12)

        nav = QHBoxLayout()
        prev_b, next_b = QPushButton("◀"), QPushButton("▶")
        prev_b.clicked.connect(lambda: self.show_sample(self.index - 1))
        next_b.clicked.connect(lambda: self.show_sample(self.index + 1))
        self.name = QLabel()
        self.name.setWordWrap(True)
        nav.addWidget(prev_b)
        nav.addWidget(self.name, 1)
        nav.addWidget(next_b)
        v.addLayout(nav)

        grid = QGridLayout()
        self.layer = QComboBox()
        self.layer.addItems(["processed frame", "original frame", "alpha mask"])
        self.layer.currentIndexChanged.connect(self.refresh)
        self.fmt = QComboBox()
        for key in geometry.FORMATS:
            self.fmt.addItem(key)
        self.fmt.currentTextChanged.connect(self.format_changed)
        grid.addWidget(QLabel("Show"), 0, 0)
        grid.addWidget(self.layer, 0, 1, 1, 2)
        grid.addWidget(QLabel("Format"), 1, 0)
        grid.addWidget(self.fmt, 1, 1, 1, 2)
        self.sliders = {}
        rows = [("k", "k", 150, 260, DEFAULT_K * 100), ("lo", "mask band low", 50, 100, DEFAULT_AGREE[0] * 100),
                ("hi", "mask band high", 100, 170, DEFAULT_AGREE[1] * 100), ("share", "head share", 40, 90, 75)]
        for r, (key, text, lo, hi, val) in enumerate(rows, start=2):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi)
            s.setValue(int(round(val)))
            value = QLabel()
            value.setMinimumWidth(44)
            s.valueChanged.connect(self.refresh)
            grid.addWidget(QLabel(text), r, 0)
            grid.addWidget(s, r, 1)
            grid.addWidget(value, r, 2)
            self.sliders[key] = (s, value)
        v.addLayout(grid)
        self.use_mask = QCheckBox("Use the mask cue")
        self.use_mask.setChecked(True)
        self.use_mask.toggled.connect(self.refresh)
        v.addWidget(self.use_mask)

        self.readout = QLabel()
        self.readout.setTextFormat(Qt.TextFormat.RichText)
        self.readout.setWordWrap(True)
        self.readout.setStyleSheet("background: #f6f6f4; border: 1px solid #ddd; border-radius: 6px; padding: 8px;")
        v.addWidget(self.readout)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(260)
        v.addWidget(self.preview, 1)

        buttons = QHBoxLayout()
        reset = QPushButton("Reset")
        reset.setToolTip("Back to the values in idphoto/config.py")
        reset.clicked.connect(self.reset)
        use_fit = QPushButton("Use fitted k")
        use_fit.clicked.connect(self.use_fitted_k)
        copy = QPushButton("Copy config")
        copy.setToolTip("Copy CROWN_K and CROWN_MASK_AGREE lines for idphoto/config.py")
        copy.clicked.connect(self.copy_config)
        for b in (reset, use_fit, copy):
            buttons.addWidget(b)
        v.addLayout(buttons)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view, 1)
        lay.addWidget(panel)
        self.setCentralWidget(central)
        self.resize(1320, 820)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self.show_sample(self.index - 1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self.show_sample(self.index + 1))
        QTimer.singleShot(0, lambda: self.show_sample(0))

    # -- marks ---------------------------------------------------------------------------
    def _load_marks(self):
        try:
            marks = json.loads(MARKS.read_text())
        except (OSError, ValueError):
            marks = {}
        for s in self.samples:
            if s.key in marks:
                s.truth = np.array(marks[s.key], np.float32)

    def _save_marks(self):
        try:
            marks = json.loads(MARKS.read_text())
        except (OSError, ValueError):
            marks = {}
        for s in self.samples:
            if s.auto_truth:
                continue
            if s.truth is None:
                marks.pop(s.key, None)
            else:
                marks[s.key] = [float(x) for x in s.truth]
        MARKS.parent.mkdir(parents=True, exist_ok=True)
        MARKS.write_text(json.dumps(marks, indent=2, ensure_ascii=False))

    def set_mark(self, pt):
        sample = self.samples[self.index]
        sample.truth, sample.auto_truth = pt, False
        self._save_marks()
        self.refresh()

    # -- settings ------------------------------------------------------------------------
    def value(self, key) -> float:
        return self.sliders[key][0].value() / 100

    def format_changed(self, key):
        s = self.sliders["share"][0]
        s.blockSignals(True)
        s.setValue(int(round(geometry.FORMATS[key].head_share * 100)))
        s.blockSignals(False)
        self.refresh()

    def reset(self):
        for key, val in (("k", DEFAULT_K), ("lo", DEFAULT_AGREE[0]), ("hi", DEFAULT_AGREE[1])):
            self.sliders[key][0].setValue(int(round(val * 100)))
        self.format_changed(self.fmt.currentText())

    def use_fitted_k(self):
        k, n = fitted_k(self.samples)
        if k is not None:
            self.sliders["k"][0].setValue(int(round(k * 100)))

    def copy_config(self):
        text = (f"CROWN_K = {self.value('k'):.2f}  # crown ~ chin - k (chin - eye line)\n"
                f"CROWN_MASK_AGREE = ({self.value('lo'):.2f}, {self.value('hi'):.2f})"
                "  # mask-top cue is used only if it agrees with the proportion\n")
        QApplication.clipboard().setText(text)
        print(text, end="")

    # -- display -------------------------------------------------------------------------
    def show_sample(self, i: int):
        self.index = i % len(self.samples)
        sample = self.samples[self.index]
        if sample.ctx is None:
            self.name.setText(f"Processing {sample.name}…")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            try:
                self.processor.run(sample)
            except Exception as exc:  # show the error, keep the window
                self.name.setText(f"{sample.name}: {exc}")
                return
            finally:
                QApplication.restoreOverrideCursor()
        self.name.setText(f"<b>{self.index + 1}/{len(self.samples)}</b> {sample.name}")
        self.refresh()

    def refresh(self, *_):
        config.CROWN_K = self.value("k")
        config.CROWN_MASK_AGREE = (self.value("lo"), self.value("hi"))
        self.sliders["k"][1].setText(f"{self.value('k'):.2f}")
        self.sliders["lo"][1].setText(f"{self.value('lo'):.2f}")
        self.sliders["hi"][1].setText(f"{self.value('hi'):.2f}")
        self.sliders["share"][1].setText(f"{self.sliders['share'][0].value()}%")
        sample = self.samples[self.index]
        ctx = sample.ctx
        if ctx is None:
            return
        layer = self.layer.currentIndex()
        if layer == 0:
            img = color.encode_u8(ctx.output_frame)
        elif layer == 1:
            img = ctx.raw_frame
        else:
            a = np.clip(ctx.alpha * 255 + 0.5, 0, 255).astype(np.uint8)
            img = np.repeat(a[..., None], 3, axis=2)
        fmt_key, share, use_mask = self.fmt.currentText(), self.value("share"), self.use_mask.isChecked()
        r = evaluate(sample, fmt_key, share, use_mask)
        self.view.image, self.view.result, self.view.truth = to_qimage(img), r, sample.truth
        self.view.agree, self.view.use_mask = config.CROWN_MASK_AGREE, use_mask
        self.view.update()
        self.readout.setText(self._readout(sample, r, fmt_key, share, use_mask))
        if r is not None:
            fmt = geometry.FORMATS[fmt_key]
            photo = geometry.render_photo(ctx.output_frame, r["box"], fmt, color.hex_to_linear("#ffffff"))
            pix = QPixmap.fromImage(to_qimage(photo))
            self.preview.setPixmap(pix.scaled(320, 300, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))
        else:
            self.preview.clear()

    def _readout(self, sample, r, fmt_key, share, use_mask) -> str:
        if r is None:
            return "No face found in this photo."
        geo, cues = r["geo"], r["cues"]
        if not use_mask:
            mask = "off"
        elif cues["mask"] is None:
            mask = "not found (mask runs off the frame or never ends)"
        else:
            lo, hi = config.CROWN_MASK_AGREE
            ok = lo <= cues["agree"] <= hi
            mask = f"agree {cues['agree']:.2f} → {'<b>used</b>' if ok else 'rejected'}"
        lines = [
            f"Crown source: <b>{geo['crown_source']}</b>",
            f"Mask cue: {mask}",
            f"Head length: {geo['head_length']:.1f} px (frame) · chin–eye {cues['ec']:.1f} px",
            f"Crop: {'inside the frame' if r['fits'] else '<span style=\"color:#c0392b\">runs off the frame</span>'}",
        ]
        if "head_err" in r:
            err = r["head_err"]
            good = abs(err) <= 1.0 if r["unit"] == "mm" else True
            tag = " (auto from demo mask)" if sample.auto_truth else ""
            lines.append(f"<br><b>This photo</b>{tag}: head-length error "
                         f"<span style='color:{'#2a7a3b' if good else '#c0392b'}'><b>{err:+.2f} {r['unit']}</b></span>"
                         f" ({r['crown_err_px']:+.1f} px); k from the mark {r['k_true']:.2f}")
        else:
            lines.append("<br>Click the top of the head to mark the true crown.")
        errs = [x["head_err"] for s in self.samples if s.truth is not None
                for x in [evaluate(s, fmt_key, share, use_mask)] if x is not None and "head_err" in x]
        k, n = fitted_k(self.samples)
        if n:
            unit = "mm" if geometry.FORMATS[fmt_key].size_mm else "px"
            lines.append(f"<b>All {n} marked</b>: mean |error| {np.mean(np.abs(errs)):.2f} {unit}, "
                         f"max {np.max(np.abs(errs)):.2f} {unit}; fitted k (proportion only) <b>{k:.2f}</b>")
        return "<br>".join(lines)

    def closeEvent(self, event):
        self.processor.close()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos", nargs="*", help="still photos (JPG/PNG)")
    parser.add_argument("--demo", action="store_true", help="add synthetic faces (no camera or model needed)")
    parser.add_argument("--snapshot", default=None, help=argparse.SUPPRESS)  # save a screenshot and quit
    args = parser.parse_args()
    samples = [Sample(Path(p).name, path=p) for p in args.photos]
    if args.demo or not samples:
        samples += [Sample(f"demo {i + 1}", scene=sc) for i, sc in enumerate(DEMO_SCENES)]
    app = QApplication(sys.argv[:1])
    font = QFont(app.font())
    font.setPointSizeF(max(font.pointSizeF(), 12))
    app.setFont(font)
    win = Inspector(samples)
    win.show()
    if args.snapshot:
        def snap():
            win.grab().save(args.snapshot)
            win.close()
        QTimer.singleShot(4000, snap)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
