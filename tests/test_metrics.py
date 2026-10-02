import math

from jianjin.checks import CheckResult
from jianjin.eval.inject import Defect
from jianjin.eval.metrics import overlaps, score_kind, wilson
from jianjin.types import Cause, Finding


def test_wilson():
    lo, hi = wilson(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_overlaps():
    assert overlaps((1, 3), (2, 4))
    assert not overlaps((1, 2), (2.5, 4))


def pause(t0, t1):
    return Finding("long_pause", Cause.LONG_PAUSE, t0, t1, t1 - t0, "")


def test_score_kind():
    labels = [Defect("insert_pause", Cause.LONG_PAUSE, 2, 6)] * 3
    injected = [
        CheckResult("a", [pause(2.1, 6.0)]),  # 检出且位置对
        CheckResult("b", [pause(8.0, 12.0)]),  # 检出但位置错
        CheckResult("c", []),  # 漏了
    ]
    clean = [CheckResult("a"), CheckResult("b", [pause(1, 5)]), CheckResult("c"), CheckResult("d")]
    s = score_kind("insert_pause", labels, injected, clean)
    assert (s["n_injected"], s["detected"], s["localized"]) == (3, 2, 1)
    assert (s["n_clean"], s["clean_flagged"]) == (4, 1)
    assert math.isclose(s["detect_rate"], 2 / 3)
    assert math.isclose(s["clean_flag_rate"], 0.25)
