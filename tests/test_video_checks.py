import numpy as np
import pytest

from helpers import assert_bilingual, make_episode
from assay.checks import CheckConfig
from assay.checks.video import WRONG_TASK, relative, wrong_task
from assay.judges.base import Curve
from assay.types import Cause


def batch(n=6, task="pick"):
    return [make_episode(np.zeros(300), ep_id=f"e{i}", task=task) for i in range(n)]


def curve(peak):
    return Curve("robometer", "v", [0.0, 5.0, 10.0], [0.1, peak / 2, peak])


def test_relative_uses_median_and_iqr_with_floor():
    assert relative([0.5], [0.1, 0.2, 0.3, 0.4, 0.5]).tolist() == pytest.approx([1.0])
    assert relative([0.2], [0.1, 0.1, 0.1]).tolist() == pytest.approx([10.0])  # 四分位距 0，按 0.01 算


def test_wrong_task_compares_with_same_task_batch():
    eps = batch()
    curves = {ep.id: curve(0.9) for ep in eps}
    curves["e0"] = curve(0.3)
    found, skipped = wrong_task(eps, curves, "robometer", CheckConfig())
    assert list(found) == ["e0"] and not skipped
    (f,) = found["e0"]
    assert f.cause == Cause.WRONG_TASK and f.source == "robometer"
    assert (f.t0, f.t1) == (0.0, eps[0].duration) and f.score > WRONG_TASK["robometer"][1]
    assert "30%" in f.evidence and "90%" in f.evidence and "pick" in f.evidence
    assert_bilingual(f.evidence)


def test_wrong_task_skips_small_batches_and_missing_curves():
    eps = batch(4) + [make_episode(np.zeros(300), ep_id="p0", task="place")]
    curves = {ep.id: curve(0.9) for ep in eps[:3]}
    found, skipped = wrong_task(eps, curves, "robometer", CheckConfig())
    assert not found
    assert "只有 3 条" in skipped["e0"] and "缓存里没有" in skipped["e3"]
    assert_bilingual(skipped["e0"])
    assert_bilingual(skipped["e3"])


def test_wrong_task_uses_reference_when_given():
    clean = batch()
    odd = make_episode(np.zeros(300), ep_id="odd", task="pick")
    curves = {ep.id: curve(0.9) for ep in clean} | {"odd": curve(0.3)}
    found, _ = wrong_task([odd], curves, "robometer", CheckConfig(), reference=clean)
    assert list(found) == ["odd"]


def test_wrong_task_without_threshold_is_not_checked():
    eps = batch()
    found, skipped = wrong_task(eps, {ep.id: curve(0.9) for ep in eps}, "newjudge", CheckConfig())
    assert not found and "还没在开发数据上给 newjudge 定" in skipped["e0"]
    assert_bilingual(skipped["e0"])


from assay.checks.video import drawdown, progress_drop  # noqa: E402


def test_drawdown_finds_biggest_fall_from_running_peak():
    c = Curve("robometer", "v", [0, 1, 2, 3, 4], [0.1, 0.6, 0.3, 0.7, 0.65])
    assert drawdown(c) == pytest.approx((0.3, 1.0, 2.0, 0.6, 0.3))


def test_drawdown_of_rising_curve_is_zero():
    assert drawdown(Curve("robometer", "v", [0, 1, 2], [0.1, 0.2, 0.3]))[0] == 0.0


def test_progress_drop_reports_where_progress_fell():
    eps = [make_episode(np.zeros(90), ep_id=f"e{i}") for i in range(2)]
    curves = {"e0": Curve("robometer", "v", [0, 1, 2], [0.2, 0.8, 0.3]),
              "e1": Curve("robometer", "v", [0, 1, 2], [0.2, 0.5, 0.9])}
    found, skipped = progress_drop(eps, curves, "robometer", CheckConfig(), threshold=0.3)
    (f,) = found["e0"]
    assert f.cause == Cause.DROP and f.source == "robometer" and (f.t0, f.t1) == (1.0, 2.0)
    assert "80%" in f.evidence and "30%" in f.evidence
    assert_bilingual(f.evidence)
    assert "e1" not in found and not skipped


def test_progress_drop_without_threshold_or_curve():
    eps = [make_episode(np.zeros(90), ep_id="e0")]
    _, skipped = progress_drop(eps, {}, "newjudge", CheckConfig())
    assert "还没在开发数据上给 newjudge 定" in skipped["e0"]
    assert_bilingual(skipped["e0"])
    _, skipped = progress_drop(eps, {}, "robometer", CheckConfig(), threshold=0.3)
    assert "缓存里没有" in skipped["e0"]


from assay.checks.video import clip, return_start  # noqa: E402


def out_and_back():
    """0 到 1 秒走到远处，1 到 2 秒停住，2 到 3 秒走回开头的姿态；夹爪不动。"""
    x = np.concatenate([np.linspace(0, 1, 30), np.ones(30), np.linspace(1, 0, 30)])
    return make_episode(np.stack([x, x, np.zeros(90)], axis=1), names=["j0", "j1", "gripper"])


def test_return_start_is_last_time_far_from_start_pose():
    # 离开头姿态的距离正比于 x；最后一次不小于最远的 0.7 倍是第 68 帧（x = 1 - 8/29）
    assert return_start(out_and_back()) == pytest.approx(68 / 30)


def test_return_start_without_coming_back_is_episode_end():
    ep = make_episode(np.tile(np.linspace(0, 1, 90)[:, None], (1, 2)))  # 一路走远，不回来
    assert return_start(ep) == ep.duration


def test_return_start_ignores_gripper():
    ep = make_episode(np.stack([np.zeros(90), np.linspace(0, 1, 90)], axis=1), names=["j0", "gripper"])
    assert return_start(ep) == ep.duration


def test_clip_keeps_points_up_to_end():
    c = Curve("robometer", "v", [0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3])
    k = clip(c, 2.0)
    assert (k.times, k.progress, k.success) == ([0, 1, 2], [0.1, 0.2, 0.3], [0.0, 0.1, 0.2])
    assert clip(c, -1.0).times == [0]


def test_progress_drop_ignores_fall_while_returning_home():
    ep = out_and_back()
    home = Curve("robometer", "v", [0, 1, 2, 2.5, 3], [0.1, 0.5, 0.9, 0.4, 0.2])  # 2.27 秒开始回原位以后才掉
    early = Curve("robometer", "v", [0, 1, 1.5, 2, 3], [0.1, 0.9, 0.3, 0.5, 0.2])  # 还在远处就掉了
    found, _ = progress_drop([ep], {ep.id: home}, "robometer", CheckConfig(), threshold=0.3)
    assert not found
    found, _ = progress_drop([ep], {ep.id: early}, "robometer", CheckConfig(), threshold=0.3)
    assert (found[ep.id][0].t0, found[ep.id][0].t1) == (1.0, 1.5)


def test_video_rules_ignore_curve_past_joint_data():
    eps = batch()  # 每条 10 秒，关节不动
    curves = {ep.id: curve(0.9) for ep in eps}
    curves["e0"] = Curve("robometer", "v", [0.0, 5.0, 10.0, 14.0], [0.1, 0.2, 0.3, 0.95])  # 高点在关节数据以外
    found, _ = wrong_task(eps, curves, "robometer", CheckConfig())
    assert list(found) == ["e0"]
    late_drop = {"e1": Curve("robometer", "v", [0.0, 5.0, 10.0, 12.0], [0.1, 0.5, 0.9, 0.1])}
    found, _ = progress_drop([eps[1]], late_drop, "robometer", CheckConfig(), threshold=0.3)
    assert not found


def test_progress_drop_off_for_topreward_says_why():
    """TOPReward 的「进度回落」定过阈值但没启用（开头几帧打分不稳，成功示范上报出大量假回落）：记进没查，写原因。"""
    eps = [make_episode(np.zeros(90), ep_id="e0")]
    curves = {"e0": Curve("topreward", "v", [0, 1, 2], [0.8, 0.0, 1.0], [0.4, 0.01, 0.95])}
    found, skipped = progress_drop(eps, curves, "topreward", CheckConfig())
    assert not found and "没启用" in skipped["e0"]
    assert_bilingual(skipped["e0"])
    found, _ = progress_drop(eps, curves, "topreward", CheckConfig(), threshold=0.5)  # 明给阈值时照查，考卷要用
    assert list(found) == ["e0"]
