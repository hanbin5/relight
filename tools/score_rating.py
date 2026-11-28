"""Score the blind rating: how many raters prefer the relit photo (plan C.1, Appendix D).

    uv run python tools/score_rating.py responses.csv
    uv run python tools/score_rating.py responses.csv --key out/rating/key.csv

responses.csv is the form's CSV export: one row per rater, one column per pair. A column
counts as pair N if its header contains the number N (e.g. "Pair 3" or "3. Which photo
looks better?"), and each answer must end in A or B ("A", "Photo B", ...). Other columns,
like the timestamp, are ignored.

A rater "prefers relit" if they picked the relit photo in more than half of their pairs;
raters with an exact tie are left out. Pass mark: at least 32 of 50 raters, one-sided
sign test p < 0.05.
"""

import argparse
import csv
import re
import sys

from eval_stats import sign_test


def read_key(path):
    with open(path, newline="") as f:
        return {int(r["pair"]): r["relit_side"] for r in csv.DictReader(f)}


def pair_columns(header, key):
    """Header index -> pair number, for headers that contain a pair number from the key."""
    cols = {}
    for i, name in enumerate(header):
        m = re.search(r"\d+", name)
        if m and int(m.group()) in key and "timestamp" not in name.lower():
            cols[i] = int(m.group())
    return cols


def score(rows, cols, key):
    """Returns (per-rater relit share list, per-pair relit counts {pair: (relit, answered)})."""
    raters, pairs = [], {p: [0, 0] for p in key}
    for row in rows:
        picked = []
        for i, p in cols.items():
            answer = row[i].strip().upper() if i < len(row) else ""
            if not answer or answer[-1] not in "AB":
                continue  # skipped question
            relit = answer[-1] == key[p]
            picked.append(relit)
            pairs[p][0] += relit
            pairs[p][1] += 1
        if picked:
            raters.append(sum(picked) / len(picked))
    return raters, pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("responses")
    parser.add_argument("--key", default="out/rating/key.csv")
    args = parser.parse_args()

    key = read_key(args.key)
    with open(args.responses, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    cols = pair_columns(header, key)
    if not cols:
        print("No pair columns found: headers need the pair number, e.g. 'Pair 1'.")
        return 1
    raters, pairs = score(rows, cols, key)

    print(f"{len(raters)} raters, {len(cols)} pairs\n")
    print("pair  photo-relit picked")
    for p in sorted(pairs):
        relit, n = pairs[p]
        print(f"{p:4d}  {relit:3d} / {n:<3d} ({relit / n:.0%})" if n else f"{p:4d}  no answers")

    prefer = sum(s > 0.5 for s in raters)
    ties = sum(s == 0.5 for s in raters)
    n = len(raters) - ties
    print(f"\nraters preferring relit: {prefer} of {n} (ties left out: {ties})")
    if n:
        p = sign_test(prefer, n)
        print(f"one-sided sign test p = {p:.3f} -> {'significant' if p < 0.05 else 'not significant'} at 0.05")
    print("plan's pass mark: at least 32 of 50 raters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
