"""Score the usability test: our app vs the current workflow (plan C.1, Appendix D).

    uv run python tools/score_usability.py usability.csv

usability.csv has one row per user, filled in by hand during the test:

    user,app_retakes,phone_retakes,app_seconds,phone_seconds
    u1,1,4,95,240

"phone" is the current workflow: a phone camera plus an existing ID photo app. Retakes
count the photos taken before the user had one they would use; seconds is the time to
that photo. Pass mark: fewer retakes with our app for at least 6 of 10 users.
"""

import argparse
import csv
import sys

import numpy as np

from eval_stats import bootstrap_ci, wilcoxon


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{k: (v if k == "user" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def summary(rows):
    retakes = [r["phone_retakes"] - r["app_retakes"] for r in rows]  # > 0: fewer with our app
    seconds = [r["phone_seconds"] - r["app_seconds"] for r in rows]  # > 0: faster with our app
    return {
        "users": len(rows),
        "fewer_retakes": sum(d > 0 for d in retakes),
        "same_retakes": sum(d == 0 for d in retakes),
        "retakes_wilcoxon_p": wilcoxon(retakes)[1],
        "seconds_saved_median": float(np.median(seconds)),
        "seconds_saved_ci": bootstrap_ci(seconds),
        "seconds_wilcoxon_p": wilcoxon(seconds)[1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        print("No users in the file.")
        return 1
    s = summary(rows)
    lo, hi = s["seconds_saved_ci"]
    print(f"{s['users']} users")
    print(f"fewer retakes with our app: {s['fewer_retakes']} of {s['users']} (same: {s['same_retakes']})"
          f"   Wilcoxon one-sided p = {s['retakes_wilcoxon_p']:.3f}")
    print(f"time saved: median {s['seconds_saved_median']:.0f} s, 95% bootstrap CI {lo:.0f} to {hi:.0f} s"
          f"   Wilcoxon one-sided p = {s['seconds_wilcoxon_p']:.3f}")
    print("plan's pass mark: fewer retakes for at least 6 of 10 users")
    return 0


if __name__ == "__main__":
    sys.exit(main())
