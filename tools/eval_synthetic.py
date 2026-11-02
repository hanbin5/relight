"""Synthetic evaluation of modules A, B and D (the plan's C.1 checks, without volunteers).

    uv run python tools/eval_synthetic.py

- Relighting (A): R_LR and shadow share before and after, over a sweep of side lights.
- Matting (B): IoU against the renderer's true person mask, for several walls and tops.
- Checker (D): staged faults like C.1's (side shadow, glare, blur, tilt, closed eyes,
  open mouth, top matching the wall); prints which rules fire, and recall per rule.

These numbers only show that the methods behave as designed on clean synthetic frames.
The plan's claims are tested with real people and the baselines in C.1.
"""

import sys

import numpy as np

from idphoto import synthetic
from idphoto.checker import RULES
from idphoto.pipeline import Pipeline, Settings

FRONT = dict(light=(0.0, 0.25, 1.0), ambient=0.3)


def run(scene: synthetic.Scene, frames: int = 6):
    rgb, lm, person = synthetic.render(scene)
    pipeline = Pipeline()
    for i in range(frames):
        ctx, _ = pipeline.process(rgb, Settings(), landmarks=lm, timestamp_ms=33 * i)
    return ctx, person


def relighting():
    print("Relighting (A): side light swept from frontal to the side")
    print("  light x   R_LR before -> after   shadow before -> after")
    for x in (0.0, -0.25, -0.45, -0.55, -0.65, -0.75):
        z = float(np.sqrt(max(1.0 - x * x - 0.04, 0.05)))
        ctx, _ = run(synthetic.Scene(light=(x, 0.2, z), ambient=0.25), frames=5)
        s = ctx.stats
        ok = 0.90 <= s["r_lr_after"] <= 1.10
        print(f"  {x:+.2f}      {s['r_lr_before']:.2f} -> {s['r_lr_after']:.2f} {'ok ' if ok else 'out'}"
              f"        {s['shadow_before']:5.1%} -> {s['shadow_after']:5.1%}")


def matting():
    print("\nMatting (B): IoU against the true person mask")
    cases = {
        "light wall, navy top": synthetic.Scene(**FRONT),
        "side light": synthetic.Scene(light=(-0.55, 0.2, 0.8), ambient=0.25),
        "blue wall, black top": synthetic.Scene(wall="#9fb6cf", top="#222222", **FRONT),
        "beige wall, red top": synthetic.Scene(wall="#e8dcc8", top="#9c2f2f", **FRONT),
        "top matching the wall": synthetic.Scene(top="#d8d6d2", light=(-0.55, 0.2, 0.8), ambient=0.25),
    }
    for name, scene in cases.items():
        ctx, person = run(scene, frames=3)
        a = ctx.alpha > 0.5
        iou = (a & person).sum() / (a | person).sum()
        v7 = ctx.rule_verdicts[7]
        print(f"  {name:24s} IoU {iou:.3f}   rule 7 {'pass' if v7.passed else 'FAIL'}"
              f" (uncertain {v7.detail['uncertain_share']:.1%}, contrast {v7.detail['contrast']:.2f})")


def checker():
    print("\nChecker (D): staged faults -> rules that fire")
    faults = {
        "good frame": (synthetic.Scene(**FRONT), set()),
        "side shadow": (synthetic.Scene(light=(-0.8, 0.2, 0.55), ambient=0.12), {1}),
        "glare on glasses": (synthetic.Scene(glare=True, **FRONT), {2}),
        "blur": (synthetic.Scene(blur=3, **FRONT), {3}),
        "head tilt": (synthetic.Scene(roll=6.0, **FRONT), {4}),
        "head turned": (synthetic.Scene(yaw=12.0, **FRONT), {4}),
        "closed eyes": (synthetic.Scene(eyes_open=False, **FRONT), {5}),
        "open mouth": (synthetic.Scene(mouth_open=0.12, **FRONT), {6}),
        "top matching the wall": (synthetic.Scene(top="#d8d6d2", light=(-0.55, 0.2, 0.8), ambient=0.25), {7}),
    }
    hits = {n: [0, 0] for n in RULES}
    false_alarms = 0
    for name, (scene, expected) in faults.items():
        ctx, _ = run(scene)
        fired = {n for n, v in ctx.rule_verdicts.items() if not v.passed}
        for n in expected:
            hits[n][1] += 1
            hits[n][0] += n in fired
        if not expected:
            false_alarms += len(fired)
        extra = sorted(fired - expected)
        print(f"  {name:22s} fired {sorted(fired) or '-'}" + (f"   (also {extra})" if expected and extra else ""))
    print("  recall per rule: " + ", ".join(f"{n}: {h}/{t}" for n, (h, t) in hits.items() if t))
    print(f"  false alarms on the good frame: {false_alarms}")


def main() -> int:
    relighting()
    matting()
    checker()
    return 0


if __name__ == "__main__":
    sys.exit(main())
