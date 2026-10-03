from jianjin.diagnose import Label, diagnose, label_of, overlaps
from jianjin.judges.base import Curve
from jianjin.types import Cause, Finding

DUR = 10.0


def f(cause, t0, t1, source="joints", hard=False, check="x"):
    return Finding(check, cause, t0, t1, 1.0, f"{cause.value} {t0}-{t1}", hard=hard, source=source)


def test_overlaps():
    assert overlaps((1, 3), (2, 4)) and not overlaps((1, 2), (2.5, 4))


def test_nothing_found_is_pass():
    assert label_of([], DUR) == (Label.PASS, [])


def test_hard_rule_rejects_alone():
    label, reasons = label_of([f(Cause.DATA_ERROR, 0, 0.5, hard=True)], DUR)
    assert label == Label.REJECT and reasons[0].startswith("客观硬规则")


def test_one_soft_signal_is_doubt():
    label, reasons = label_of([f(Cause.LONG_PAUSE, 2, 6)], DUR)
    assert label == Label.DOUBT and "只有关节" in reasons[0]


def test_same_source_twice_is_still_doubt():
    assert label_of([f(Cause.LONG_PAUSE, 2, 6), f(Cause.LONG_PAUSE, 5, 8)], DUR)[0] == Label.DOUBT


def test_two_sources_localized_and_overlapping_reject():
    label, reasons = label_of([f(Cause.LONG_PAUSE, 2, 6), f(Cause.DROP, 5, 7, source="robometer")], DUR)
    assert label == Label.REJECT and "Robometer" in reasons[0] and "第 5.0 到 6.0 秒" in reasons[0]


def test_two_sources_disjoint_is_doubt():
    label, reasons = label_of([f(Cause.LONG_PAUSE, 2, 4), f(Cause.DROP, 6, 7, source="robometer")], DUR)
    assert label == Label.DOUBT and "对不上" in reasons[0]


def test_whole_episode_finding_needs_same_cause():
    whole = f(Cause.WRONG_TASK, 0, DUR, source="robometer")
    assert label_of([whole, f(Cause.LONG_PAUSE, 2, 6)], DUR)[0] == Label.DOUBT
    assert label_of([whole, f(Cause.WRONG_TASK, 0, DUR, source="topreward")], DUR)[0] == Label.REJECT


def test_diagnose_sorts_causes_and_keeps_extras():
    late, early = f(Cause.LONG_PAUSE, 6, 8), f(Cause.LONG_PAUSE, 1, 3)
    c = Curve("robometer", "v", [0.0, 10.0], [0.0, 1.0])
    v = diagnose("e1", DUR, [late, early], {"end_pose": "同任务只有 3 条"}, c)
    assert v.episode_id == "e1" and v.label == Label.DOUBT
    assert v.causes == [early, late] and v.not_checked == {"end_pose": "同任务只有 3 条"} and v.progress is c
