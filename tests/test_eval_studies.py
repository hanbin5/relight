import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_stats  # noqa: E402
import score_rating  # noqa: E402
import score_usability  # noqa: E402


def test_sign_test_matches_the_plan():
    # plan C.1: 32 of 50 gives p about 0.03, 30 of 50 about 0.10
    assert eval_stats.sign_test(32, 50) == pytest.approx(0.032, abs=0.001)
    assert eval_stats.sign_test(30, 50) == pytest.approx(0.101, abs=0.001)


def test_wilcoxon_all_positive_is_the_smallest_p():
    w, p = eval_stats.wilcoxon([1, 2, 3, 4, 5])
    assert w == 15 and p == pytest.approx(1 / 32)
    assert eval_stats.wilcoxon([0, 0])[1] == 1.0


def test_score_rating_reads_form_columns():
    key = {1: "A", 2: "B", 3: "A"}
    header = ["Timestamp", "Pair 1", "Pair 2", "Pair 3"]
    rows = [
        ["2026/11/26", "A", "B", "Photo A"],  # relit every time
        ["2026/11/26", "B", "B", "B"],  # relit once
        ["2026/11/26", "A", "", "B"],  # skipped pair 2: relit 1 of 2, a tie
    ]
    cols = score_rating.pair_columns(header, key)
    assert cols == {1: 1, 2: 2, 3: 3}
    raters, pairs = score_rating.score(rows, cols, key)
    assert raters == [1.0, pytest.approx(1 / 3), 0.5]
    assert pairs[1] == [2, 3] and pairs[2] == [2, 2]


def test_usability_summary():
    rows = [
        {"user": "u1", "app_retakes": 1, "phone_retakes": 4, "app_seconds": 90, "phone_seconds": 200},
        {"user": "u2", "app_retakes": 2, "phone_retakes": 2, "app_seconds": 120, "phone_seconds": 150},
        {"user": "u3", "app_retakes": 3, "phone_retakes": 1, "app_seconds": 160, "phone_seconds": 100},
    ]
    s = score_usability.summary(rows)
    assert s["fewer_retakes"] == 1 and s["same_retakes"] == 1
    assert s["seconds_saved_median"] == 30
