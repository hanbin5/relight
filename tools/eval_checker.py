"""Precision and recall of each checker rule on real photos (plan C.1, module D).

    uv run python tools/eval_checker.py testset/

The test set is one folder per staged fault, plus good/ for photos with no fault:

    testset/good/          testset/blur/          testset/mouth_open/
    testset/side_shadow/   testset/tilt/          testset/top_like_wall/
    testset/glare/         testset/eyes_closed/

File names don't matter (e.g. alex.jpg in every folder). For each rule, a fault photo
of that rule counts as caught if the rule fails; any other photo where the rule fails
is a false alarm. Every measured value is also written to a CSV, for picking thresholds.

Each photo is run on its own, so rule 3 uses the starting bound (120) instead of the
per-camera one; use the sharpness values in the CSV to pick a bound for each camera.
"""

import argparse
import csv
import sys
from pathlib import Path

from idphoto.capture import ImageSource
from idphoto.checker import RULES
from idphoto.landmarks import FaceLandmarkDetector
from idphoto.pipeline import Pipeline, Settings, to_live

# folder name -> the rule that should fail (None: every rule should pass)
FAULTS = {
    "good": None,
    "side_shadow": 1,
    "glare": 2,
    "blur": 3,
    "tilt": 4,
    "eyes_closed": 5,
    "mouth_open": 6,
    "top_like_wall": 7,
}
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}


def find_photos(folder: Path):
    """(path, fault folder name) for every photo in the known fault folders."""
    photos = []
    for name in FAULTS:
        sub = folder / name
        if sub.is_dir():
            photos += [(p, name) for p in sorted(sub.iterdir()) if p.suffix.lower() in IMAGE_TYPES]
    return photos


def score(results):
    """results: list of (fault folder name, set of failed rules).
    Returns {rule: (caught, faults, false_alarms, precision, recall)}."""
    table = {}
    for rule in RULES:
        caught = sum(1 for name, failed in results if FAULTS[name] == rule and rule in failed)
        faults = sum(1 for name, _ in results if FAULTS[name] == rule)
        false_alarms = sum(1 for name, failed in results if FAULTS[name] != rule and rule in failed)
        flagged = caught + false_alarms
        precision = caught / flagged if flagged else None
        recall = caught / faults if faults else None
        table[rule] = (caught, faults, false_alarms, precision, recall)
    return table


def check_photo(path, detector, timestamp):
    frame = ImageSource(path).frame
    live, _ = to_live(frame)
    pipeline = Pipeline(detector)
    for _ in range(5):  # same as process_image.py: a still photo stands in for 5 frames
        timestamp += 33
        ctx, _ = pipeline.process(live, Settings(), timestamp_ms=timestamp, full_frame=frame)
    return ctx, timestamp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder")
    parser.add_argument("--csv", default="out/checker_eval.csv")
    args = parser.parse_args()

    photos = find_photos(Path(args.folder))
    if not photos:
        print(f"No photos found. Expected subfolders: {', '.join(FAULTS)}")
        return 1

    detector = FaceLandmarkDetector()
    timestamp = 0  # landmark detection needs increasing timestamps, across photos too
    results, rows, no_face = [], [], []
    for path, name in photos:
        ctx, timestamp = check_photo(path, detector, timestamp)
        if ctx.landmarks is None:
            no_face.append(path)
            continue
        verdicts = ctx.rule_verdicts
        failed = {n for n, v in verdicts.items() if not v.passed}
        results.append((name, failed))
        row = {"photo": str(path), "fault": name}
        for n, v in verdicts.items():
            row[f"{n}_{v.key}_pass"] = int(v.passed)
            row[f"{n}_{v.key}_value"] = "" if v.value is None else round(v.value, 4)
        rows.append(row)
        print(f"{path}: failed {sorted(failed) or '-'}")
    detector.close()

    print(f"\n{len(results)} photos checked, {len(no_face)} with no face found")
    for path in no_face:
        print(f"  no face: {path}")

    print("\nrule              caught  false alarms  precision  recall")
    for rule, (caught, faults, fa, precision, recall) in score(results).items():
        p = "-" if precision is None else f"{precision:.0%}"
        r = "-" if recall is None else f"{recall:.0%}"
        print(f"{rule} {RULES[rule]:14s}  {caught:>3}/{faults:<3}  {fa:>12}  {p:>9}  {r:>6}")
    good = [failed for name, failed in results if name == "good"]
    if good:
        print(f"\ngood photos passing every rule: {sum(not f for f in good)}/{len(good)}")

    if rows:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"values saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
