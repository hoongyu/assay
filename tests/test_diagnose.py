from helpers import assert_bilingual
from jianjin.diagnose import Label, diagnose, label_of, overlaps
from jianjin.judges.base import Curve
from jianjin.types import Bi, Cause, Finding, en

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
    short = Finding("too_short", Cause.DATA_ERROR, 0, 0.5, 0.5, Bi("整条只有 0.50 秒", "Only 0.50 s long"), hard=True)
    (r,) = label_of([short], DUR)[1]
    assert_bilingual(r)
    assert en(r) == "Objective hard rule: Only 0.50 s long"


def test_one_soft_signal_is_doubt():
    label, reasons = label_of([f(Cause.LONG_PAUSE, 2, 6)], DUR)
    assert label == Label.DOUBT and "只有关节" in reasons[0]
    assert_bilingual(reasons[0])


def test_same_source_twice_is_still_doubt():
    assert label_of([f(Cause.LONG_PAUSE, 2, 6), f(Cause.LONG_PAUSE, 5, 8)], DUR)[0] == Label.DOUBT


def test_two_sources_localized_and_overlapping_reject():
    label, reasons = label_of([f(Cause.LONG_PAUSE, 2, 6), f(Cause.DROP, 5, 7, source="robometer")], DUR)
    assert label == Label.REJECT and "Robometer" in reasons[0] and "第 5.0 到 6.0 秒" in reasons[0]
    assert_bilingual(reasons[0])


def test_two_sources_disjoint_is_doubt():
    label, reasons = label_of([f(Cause.LONG_PAUSE, 2, 4), f(Cause.DROP, 6, 7, source="robometer")], DUR)
    assert label == Label.DOUBT and "对不上" in reasons[0]
    assert_bilingual(reasons[0])


def test_whole_episode_finding_needs_same_cause():
    whole = f(Cause.WRONG_TASK, 0, DUR, source="robometer")
    assert label_of([whole, f(Cause.LONG_PAUSE, 2, 6)], DUR)[0] == Label.DOUBT
    label, reasons = label_of([whole, f(Cause.WRONG_TASK, 0, DUR, source="topreward")], DUR)
    assert label == Label.REJECT
    assert_bilingual(reasons[0])


def test_diagnose_sorts_causes_and_keeps_extras():
    late, early = f(Cause.LONG_PAUSE, 6, 8), f(Cause.LONG_PAUSE, 1, 3)
    c = Curve("robometer", "v", [0.0, 10.0], [0.0, 1.0])
    v = diagnose("e1", DUR, [late, early], {"end_pose": "同任务只有 3 条"}, c)
    assert v.episode_id == "e1" and v.label == Label.DOUBT
    assert v.causes == [early, late] and v.not_checked == {"end_pose": "同任务只有 3 条"} and v.progress is c


import numpy as np  # noqa: E402

from helpers import make_episode  # noqa: E402
from jianjin.diagnose import diagnose_batch  # noqa: E402


def smooth_batch(n=6):
    x = np.linspace(0, 1, 300)
    return [make_episode(np.tile(x[:, None], (1, 6)) * (1 + 0.01 * i), ep_id=f"e{i}") for i in range(n)]


def rising(top):
    return Curve("robometer", "v", [0.0, 5.0, 10.0], [0.1, top / 2, top])


def test_batch_without_video_says_so():
    vs = diagnose_batch(smooth_batch())
    assert [v.label for v in vs] == [Label.PASS] * 6
    assert all("没跑视频评判" in v.not_checked["video_judges"] for v in vs)
    assert_bilingual(vs[0].not_checked["video_judges"])


def test_batch_with_video_rules_and_progress():
    eps = smooth_batch()
    curves = {ep.id: rising(0.9) for ep in eps}
    curves["e0"] = rising(0.3)
    vs = diagnose_batch(eps, {"robometer": curves, "topreward": {}})
    v0 = vs[0]
    assert v0.label == Label.DOUBT and [f.cause for f in v0.causes] == [Cause.WRONG_TASK]
    assert v0.progress is curves["e0"]
    assert vs[1].label == Label.PASS and "video_judges" not in vs[1].not_checked
    assert "缓存里没有 TOPReward" in vs[1].not_checked["topreward.wrong_task"]


def test_two_video_judges_agreeing_reject():
    """两个视频模型都报同一条「指令不符」：判作废（2026-10-04 TOPReward 定了阈值以后才有这条路）。"""
    eps = smooth_batch()
    rob = {ep.id: rising(0.9) for ep in eps}
    rob["e0"] = rising(0.3)

    def top(success):
        return Curve("topreward", "v", [0.0, 4.0, 8.0], [0.0, 0.5, 1.0], success)

    tr = {ep.id: top([0.5, 0.8, 0.9]) for ep in eps}
    tr["e0"] = top([0.1, 0.05, 0.05])
    vs = diagnose_batch(eps, {"robometer": rob, "topreward": tr})
    assert vs[0].label == Label.REJECT
    assert sorted(f.source for f in vs[0].causes) == ["robometer", "topreward"]
    assert all(v.label == Label.PASS for v in vs[1:])


def test_reasons_space_latin_names():
    from jianjin.types import spaced

    assert spaced("关节和Robometer都报了「多余动作」") == "关节和 Robometer 都报了「多余动作」"
    _, reasons = label_of([f(Cause.LONG_PAUSE, 2, 6), f(Cause.DROP, 5, 7, source="robometer")], DUR)
    assert "、Robometer 报「物体掉落」" in reasons[0]


def test_empty_recording_and_wrong_task_agree():
    empty = f(Cause.EMPTY, 0, DUR)
    wrong = f(Cause.WRONG_TASK, 0, DUR, source="robometer")
    label, reasons = label_of([empty, wrong], DUR)
    assert label == Label.REJECT
    assert reasons == ["关节报「空录」、Robometer 报「指令不符」，说的都是整条没做指令里的任务"]
    assert en(reasons[0]) == ("joints (empty recording) and Robometer (wrong task): "
                              "both say the episode never does the task in the instruction")


def test_empty_recording_needs_the_right_partner():
    empty = f(Cause.EMPTY, 0, DUR)
    assert label_of([empty], DUR)[0] == Label.DOUBT
    assert label_of([empty, f(Cause.DATA_ERROR, 0, DUR, source="meta")], DUR)[0] == Label.DOUBT
    part = f(Cause.EMPTY, 0, 5)  # 不是整条的不算
    assert label_of([part, f(Cause.WRONG_TASK, 0, DUR, source="robometer")], DUR)[0] == Label.DOUBT
