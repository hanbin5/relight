"""Reproduce Figure 3: relight one frame of a synthetic face lit by a side window (plan B.2).

    uv run python tools/relight_demo.py
    uv run python tools/relight_demo.py --light=-0.7,0.2,0.65 --out out/harsh.png

Prints R_LR and the shadowed share of the face before and after, and saves the five
panels of Figure 3: webcam frame, fitted shading, target shading, shading ratio, relit output.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from idphoto import color, mesh, npimg, synthetic
from idphoto.context import FrameContext
from idphoto.relight import Relighter

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--light", default="-0.55,0.2,0.8", help="key light direction x,y,z (x right, y up, z to camera)")
    parser.add_argument("--ambient", type=float, default=0.25)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--out", default=str(ROOT / "out" / "relight_demo.png"))
    args = parser.parse_args()

    light = tuple(float(v) for v in args.light.split(","))
    rgb, lm, _ = synthetic.render(synthetic.Scene(light=light, ambient=args.ambient))
    ctx = FrameContext(raw_frame=rgb, landmarks=lm)
    ctx.linear_rgb = color.decode_u8(rgb)
    ctx.Y = color.luminance(ctx.linear_rgb)
    relighter = Relighter()
    for _ in range(5):  # fill the 5-frame light average
        relighter.process(ctx, args.strength)
    s = ctx.stats
    print(f"R_LR          {s['r_lr_before']:.2f} -> {s['r_lr_after']:.2f}   (balanced: 0.90-1.10)")
    print(f"shadow share  {s['shadow_before']:.0%} -> {s['shadow_after']:.0%}   (pass: < 5%)")
    print("fitted light  " + " ".join(f"{c:+.3f}" for c in ctx.light_coeffs))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    g = s["fit_grid"]
    cov = g["coverage"]
    x0, y0, x1, y1 = mesh.face_box(lm, pad=1.6)
    crop = lambda img: color.encode_u8(np.clip(npimg.resize(img, 256, 256, "linear", box=(x0, y0, x1, y1)), 0, 1))  # noqa: E731
    masked = lambda a: np.ma.masked_where(~cov, a)  # noqa: E731
    vmax = float(max(g["fitted"][cov].max(), g["target"][cov].max()))
    blue_orange = LinearSegmentedColormap.from_list("ratio", ["#2a5db0", "#ffffff", "#e0812a"])

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
    panels = [
        ("Webcam frame", crop(ctx.linear_rgb), {}, f"R_LR = {s['r_lr_before']:.2f}, shadow {s['shadow_before']:.0%}"),
        ("Fitted shading", masked(g["fitted"]), dict(cmap="gray", vmin=0, vmax=vmax), "9 SH terms, least squares over skin"),
        ("Target shading", masked(g["target"]), dict(cmap="gray", vmin=0, vmax=vmax), "even, frontal; no left-right term"),
        ("Shading ratio", masked(g["ratio"]), dict(cmap=blue_orange, norm=LogNorm(0.5, 2.0)), "blue darkens, orange brightens"),
        ("Relit output", crop(ctx.relit_frame), {}, f"R_LR = {s['r_lr_after']:.2f}, shadow {s['shadow_after']:.0%}"),
    ]
    for ax, (title, img, kw, caption) in zip(axes, panels):
        ax.imshow(img, **kw)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(caption, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Relighting one frame (B.2 formulas, synthetic face lit by a side window)", fontsize=13)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
