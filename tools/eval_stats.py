"""Small statistics helpers for the studies in plan C.1 (no SciPy needed).

- sign_test: one-sided sign test, used for the blind rating (32 of 50 -> p about 0.03).
- wilcoxon: exact one-sided Wilcoxon signed-rank test for small paired samples.
- bootstrap_ci: 95% bootstrap interval of a paired difference (median by default).
"""

from itertools import product
from math import comb

import numpy as np


def sign_test(wins: int, n: int) -> float:
    """P(at least `wins` successes out of n) if each side were equally likely."""
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2**n


def wilcoxon(diffs) -> tuple[float, float]:
    """Exact one-sided Wilcoxon signed-rank test that the differences are > 0.

    Returns (W+, p). Zero differences are dropped; ties get average ranks. The exact
    null distribution is found by trying every sign pattern, fine for n up to ~16.
    """
    d = np.asarray([x for x in diffs if x != 0], float)
    if len(d) == 0:
        return 0.0, 1.0
    order = np.argsort(np.abs(d))
    ranks = np.empty(len(d))
    ranks[order] = np.arange(1, len(d) + 1)
    for v in np.unique(np.abs(d)):  # average ranks for ties
        same = np.abs(d) == v
        ranks[same] = ranks[same].mean()
    w_plus = float(ranks[d > 0].sum())
    count = total = 0
    for signs in product((0, 1), repeat=len(d)):
        total += 1
        count += float(np.dot(signs, ranks)) >= w_plus - 1e-9
    return w_plus, count / total


def bootstrap_ci(diffs, stat=np.median, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of stat(diffs)."""
    d = np.asarray(diffs, float)
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(d, len(d))) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
