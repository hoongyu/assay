from dataclasses import replace

import numpy as np
import pytest

from helpers import assert_bilingual, make_episode
from jianjin.checks.batch import end_pose, extra_motion, regrasp, regrasp_scores
from jianjin.checks.config import CheckConfig
from jianjin.types import Cause, en

CFG = CheckConfig()
# 收尾规则的测试把参数写死，只验机制；默认值在开发集上重选
END = CheckConfig(end_pose_max=3.0, end_pose_floor=0.1, end_spread_max=0.15)


def there():
    return np.linspace(0, 1, 60)


def test_back_and_forth_flagged():
    eps = [make_episode(there(), ep_id=f"ok{i}") for i in range(5)]
    wobble = np.concatenate([there(), there()[::-1], there(), there()[::-1], there()])
    eps.append(make_episode(wobble, ep_id="bad"))
    found, skipped = extra_motion(eps, CFG)
    assert skipped == {}
    assert set(found) == {"bad"}
    f = found["bad"][0]
    assert f.cause is Cause.EXTRA_MOTION and not f.hard
    assert f.score > 4
    assert_bilingual(f.evidence)


def test_small_task_not_checked():
    eps = [make_episode(there(), ep_id=f"e{i}") for i in range(3)]
    for check in (extra_motion, end_pose):
        found, skipped = check(eps, CFG)
        assert found == {}
        assert set(skipped) == {"e0", "e1", "e2"}
        assert "3" in skipped["e0"]
        assert_bilingual(skipped["e0"])


def test_tasks_compared_separately():
    a = [make_episode(there(), ep_id=f"a{i}", task="A") for i in range(5)]
    b = [make_episode(np.linspace(0, 1, 60) * 5, ep_id=f"b{i}", task="B") for i in range(5)]
    assert extra_motion(a + b, CFG)[0] == {}
    assert end_pose(a + b, END)[0] == {}


def test_end_pose_flags_stopping_halfway():
    eps = [make_episode(there(), ep_id=f"ok{i}") for i in range(5)]
    eps.append(make_episode(there()[:30], ep_id="half"))
    found, skipped = end_pose(eps, END)
    assert skipped == {}
    assert set(found) == {"half"}
    f = found["half"][0]
    assert f.cause is Cause.INCOMPLETE and not f.hard
    assert np.isclose(f.t1, 1.0) and np.isclose(f.t0, 0.0)
    assert f.score > 0.4
    assert_bilingual(f.evidence)


def test_reference_used_instead_of_batch():
    """整批都没做完时，跟整批自己比看不出来；要跟参照的好示范比。"""
    good = [make_episode(there(), ep_id=f"ok{i}") for i in range(5)]
    halves = [make_episode(there()[:30], ep_id=f"half{i}") for i in range(5)]
    assert end_pose(halves, END)[0] == {}
    found, _ = end_pose(halves, END, reference=good)
    assert set(found) == {f"half{i}" for i in range(5)}


from jianjin.checks.batch import empty_recording  # noqa: E402


def test_arm_that_never_moved_is_empty_recording():
    eps = [make_episode(there(), ep_id=f"ok{i}") for i in range(5)]
    still = np.full(60, 0.3) + np.random.default_rng(0).normal(0, 1e-4, 60)  # 只有读数噪声
    eps.append(make_episode(still, ep_id="still"))
    found, skipped = empty_recording(eps, CFG)
    assert skipped == {} and set(found) == {"still"}
    (f,) = found["still"]
    assert f.cause is Cause.EMPTY and not f.hard and (f.t0, f.t1) == (0.0, eps[-1].duration)
    assert "手臂几乎没动" in f.evidence and f.score < 0.01
    assert_bilingual(f.evidence)


def test_partial_motion_is_not_empty():
    eps = [make_episode(there(), ep_id=f"ok{i}") for i in range(5)]
    eps.append(make_episode(there()[:30], ep_id="half"))  # 走了一半，归「没做完」管
    assert empty_recording(eps, CFG)[0] == {}


def test_too_short_left_to_hard_rule():
    eps = [make_episode(there(), ep_id=f"ok{i}") for i in range(5)]
    eps.append(make_episode(np.zeros(10), ep_id="tiny"))  # 0.33 秒，「示范过短」已经判了
    assert empty_recording(eps, CFG)[0] == {}


def ends_at(final_wrist, n=60):
    """六个关节：前五个从 0 走到 1；最后一个（像手腕转角）收尾停在 final_wrist。"""
    arm = np.tile(np.linspace(0, 1, n)[:, None], (1, 5))
    return np.concatenate([arm, np.linspace(0, final_wrist, n)[:, None]], axis=1)


def test_joint_that_never_ends_the_same_is_not_flagged():
    # 最后一个关节的收尾位置在好示范之间就差很多，同样的差法不该被当成没做完
    eps = [make_episode(ends_at(w), ep_id=f"ok{i}") for i, w in enumerate([0.2, 0.5, 0.8, 0.35, 0.65])]
    eps.append(make_episode(ends_at(0.95), ep_id="odd_wrist"))
    eps.append(make_episode(ends_at(0.5)[:30], ep_id="half"))
    found, skipped = end_pose(eps, END)
    assert skipped == {} and set(found) == {"half"}
    assert "倍" in found["half"][0].evidence


def test_batch_without_a_common_end_pose_is_not_checked():
    rng = np.random.default_rng(0)
    eps = [make_episode(np.stack([np.linspace(0, e, 60) for e in rng.uniform(0, 1, 3)], axis=1), ep_id=f"r{i}")
           for i in range(8)]
    found, skipped = end_pose(eps, END)
    assert found == {} and len(skipped) == 8
    assert "停下来的姿态本来就不固定" in skipped["r0"]
    assert_bilingual(skipped["r0"])


REG = CheckConfig(grip_band=0.15, grip_dwell_s=0.2, regrasp_quantile=0.75, regrasp_margin=0, regrasp_source="action")


def grip(cycles: int, n: int = 90) -> np.ndarray:
    """夹爪从合着开始，在 n 帧里开合 cycles 回（张开、合上各换一次档，共 2×cycles 次）。"""
    g = np.zeros(n)
    seg = (n - 10) // (2 * cycles) if cycles else 0
    for k in range(cycles):
        start = 5 + 2 * k * seg
        g[start:start + seg] = 1.0
    return g


def grip_ep(cycles, ep_id, action_cycles=None, task="pick the cube"):
    state = np.column_stack([np.linspace(0, 1, 90), grip(cycles)])
    action = None if action_cycles is None else np.column_stack([np.linspace(0, 1, 90), grip(action_cycles)])
    return make_episode(state, ep_id=ep_id, task=task, names=["j0", "gripper"], action=action)


def test_regrasp_flags_extra_cycles():
    eps = [grip_ep(2, f"ok{i}") for i in range(6)] + [grip_ep(4, "again")]
    found, skipped = regrasp(eps, REG)
    assert skipped == {} and set(found) == {"again"}
    f = found["again"][0]
    assert f.cause is Cause.REGRASP and not f.hard and (f.t0, f.t1) == (0.0, eps[-1].duration)
    assert f.score == 4 and "开合了 8 次" in f.evidence and "常见不超过 4 次" in f.evidence
    assert len(f.moments) == 8 and list(f.moments) == sorted(f.moments) and f.moments[0] > 0
    assert_bilingual(f.evidence)


def test_regrasp_upper_follows_the_batch():
    """叠方块那种本来 2 到 6 次的批：6 次不报，8 次才报。"""
    eps = [grip_ep(c, f"e{i}") for i, c in enumerate([1, 2, 2, 3, 3, 3])] + [grip_ep(4, "again")]
    found, _ = regrasp(eps, REG)
    assert set(found) == {"again"}


def test_regrasp_margin():
    eps = [grip_ep(2, f"ok{i}") for i in range(6)] + [grip_ep(4, "again")]
    assert regrasp(eps, replace(REG, regrasp_margin=4))[0] == {}
    assert set(regrasp(eps, replace(REG, regrasp_margin=3))[0]) == {"again"}


def test_regrasp_reads_master_arm_then_falls_back():
    """从臂夹着东西合不到底，少数了换档；主臂（采集员的手）数得到。没有主臂数据就读从臂。"""
    eps = [grip_ep(2, f"ok{i}", action_cycles=2) for i in range(6)] + [grip_ep(2, "again", action_cycles=4)]
    assert set(regrasp(eps, REG)[0]) == {"again"}
    assert regrasp(eps, replace(REG, regrasp_source="state"))[0] == {}
    no_action = [grip_ep(2, f"ok{i}") for i in range(6)] + [grip_ep(4, "again")]
    assert set(regrasp(no_action, REG)[0]) == {"again"}


def test_regrasp_not_checked_without_gripper_or_when_unused():
    plain = [make_episode(there(), ep_id=f"p{i}") for i in range(5)]
    found, skipped = regrasp(plain, REG)
    assert found == {} and len(skipped) == 5 and all("认不出夹爪" in w for w in skipped.values())
    assert_bilingual(skipped["p0"])
    still = [grip_ep(0, f"s{i}") for i in range(5)]
    found, skipped = regrasp(still, REG)
    assert found == {} and len(skipped) == 5 and all("基本不开合" in w for w in skipped.values())
    assert_bilingual(skipped["s0"])


def test_regrasp_two_hands_counted_separately():
    def two(left, right, ep_id):
        s = np.column_stack([np.linspace(0, 1, 90), grip(left), np.linspace(0, 1, 90), grip(right)])
        return make_episode(s, ep_id=ep_id, names=["left_j0", "left_gripper.pos", "right_j0", "right_gripper.pos"])

    eps = [two(2, 1, f"ok{i}") for i in range(6)] + [two(2, 3, "again")]
    found, _ = regrasp(eps, REG)
    assert set(found) == {"again"} and "右手夹爪开合了 6 次" in found["again"][0].evidence
    assert "right gripper" in en(found["again"][0].evidence)


def test_regrasp_reference_from_clean_batch():
    """考卷里整批都重抓了：拿这批自己比看不出来，传原样示范当参照就看得出来。"""
    bad = [grip_ep(4, f"b{i}") for i in range(6)]
    clean = [grip_ep(2, f"c{i}") for i in range(6)]
    assert regrasp(bad, REG)[0] == {}
    assert set(regrasp(bad, REG, reference=clean)[0]) == {f"b{i}" for i in range(6)}


def test_regrasp_scores_cover_unflagged_episodes():
    eps = [grip_ep(2, f"ok{i}") for i in range(6)] + [grip_ep(4, "again")]
    scores, skipped = regrasp_scores(eps, REG)
    assert skipped == {} and len(scores) == 7
    assert scores["ok0"].excess == 0 and scores["again"].excess == 4 and scores["again"].count == 8
    assert scores["again"].times[0] == pytest.approx(5 / 30)


def test_regrasp_ignores_a_brief_overshoot():
    """放手时夹爪一闪张大又合上（停不到 grip_dwell_s），不算多开合一回；停留下限设成 0 就会算。"""
    blip = grip(2)
    blip[80:82] = 1.0
    odd = make_episode(np.column_stack([np.linspace(0, 1, 90), blip]), ep_id="blip", names=["j0", "gripper"])
    eps = [grip_ep(2, f"ok{i}") for i in range(6)] + [odd]
    assert regrasp(eps, REG)[0] == {}
    assert set(regrasp(eps, replace(REG, grip_dwell_s=0.0))[0]) == {"blip"}


# ---- 动作不流畅、速度过快：候选分数 ----

MOT = CheckConfig(still_rate=0.02, stop_s=0.3, burst_low=0.2, burst_high=0.5, burst_dwell_s=0.1, motion_quantile=0.75)


def minjerk(n, dist=1.0):
    """最小 jerk 的一下：n 帧从 0 走到 dist。"""
    t = np.linspace(0, 1, n)
    return dist * (10 * t**3 - 15 * t**4 + 6 * t**5)


def arm_ep(path, ep_id, pad=15):
    """两个手臂关节加一个不动的夹爪；前后各停 pad 帧（摆姿势、收手）。"""
    x = np.concatenate([np.zeros(pad), path, np.full(pad, path[-1])])
    return make_episode(np.stack([x, 0.5 * x, np.zeros_like(x)], axis=1), names=["j0", "j1", "gripper"], ep_id=ep_id)


def normals():
    return [arm_ep(minjerk(90, a), f"ok{i}") for i, a in enumerate([1.0, 1.05, 0.95, 1.02, 0.98, 1.01])]


def stop_and_go():
    """同样走到 1，分四下走，中间停三回、每回 0.5 秒。"""
    parts = []
    for k in range(4):
        parts.append(k * 0.25 + minjerk(30, 0.25))
        if k < 3:
            parts.append(np.full(15, (k + 1) * 0.25))
    return arm_ep(np.concatenate(parts), "jerky")


def test_motion_scores_stop_and_go_and_fast():
    from jianjin.checks.batch import motion_scores

    ref = normals()
    eps = ref + [stop_and_go(), arm_ep(minjerk(45), "fast")]
    scores, skipped = motion_scores(eps, MOT, reference=ref)
    assert skipped == {}
    jerky, fast = scores["jerky"], scores["fast"]
    assert jerky.n == 6
    assert jerky.scores["stops"] == 3 and jerky.detail["n_stops"] == 3
    assert jerky.scores["bursts"] == 3 and jerky.detail["n_bursts"] == 4
    assert jerky.scores["duration"] == pytest.approx(195 / 120, rel=0.02)
    assert jerky.scores["jerk"] > 2
    # 每一下快到头时速度已经低于门槛，停顿起点比停住的那一帧（1.5、3.0、4.5 秒）早一点
    assert [a for a, _ in jerky.detail["stop_times"]] == pytest.approx([1.5, 3.0, 4.5], abs=0.15)
    assert all(scores[e.id].scores["stops"] <= 0 and scores[e.id].scores["bursts"] <= 0 for e in ref)
    assert fast.scores["peak_speed"] == pytest.approx(2.0, rel=0.1)
    assert fast.scores["duration_short"] == pytest.approx(120 / 75, rel=0.02)
    assert fast.scores["mean_speed"] == pytest.approx(120 / 75, rel=0.05)
    assert all(0.9 < scores[e.id].scores["peak_speed"] < 1.1 for e in ref)


def test_motion_scores_skips_when_not_comparable():
    from jianjin.checks.batch import motion_scores

    scores, skipped = motion_scores(normals()[:4], MOT)
    assert scores == {} and all("没法比" in why for why in skipped.values())
    assert_bilingual(next(iter(skipped.values())))
    grippers = [make_episode(np.zeros((60, 1)), names=["gripper"], ep_id=f"g{i}") for i in range(5)]
    scores, skipped = motion_scores(grippers, MOT)
    assert scores == {} and set(skipped.values()) == {"认不出手臂关节"}
    assert_bilingual(next(iter(skipped.values())))
    still = [arm_ep(np.zeros(60), f"s{i}") for i in range(5)]
    scores, skipped = motion_scores(still, MOT)
    assert scores == {} and all("基本没动" in why for why in skipped.values())
    assert_bilingual(next(iter(skipped.values())))


def test_jerky_rule_uses_chosen_candidate():
    from jianjin.checks.batch import jerky

    ref = normals()
    eps = ref + [stop_and_go(), arm_ep(minjerk(45), "fast")]
    cfg = replace(MOT, jerky_candidate="stops", jerky_threshold=0)
    found, skipped = jerky(eps, cfg, reference=ref)
    assert skipped == {} and set(found) == {"jerky"}
    (f,) = found["jerky"]
    assert f.check == "jerky" and f.cause is Cause.JERKY and not f.hard
    assert f.t0 == 0 and f.t1 == pytest.approx(eps[-2].duration) and f.score == 3
    assert "中途停下又走了 3 回" in f.evidence and "同任务 6 条示范常见不超过 0 回" in f.evidence
    assert_bilingual(f.evidence)
    found, _ = jerky(eps, replace(MOT, jerky_candidate="duration", jerky_threshold=1.3), reference=ref)
    assert set(found) == {"jerky"} and "是同任务 6 条示范中位数的 1.6 倍" in found["jerky"][0].evidence
    assert_bilingual(found["jerky"][0].evidence)
    for cand in ("bursts", "jerk"):
        found, _ = jerky(eps, replace(MOT, jerky_candidate=cand, jerky_threshold=0), reference=ref)
        assert_bilingual(found["jerky"][0].evidence)


def test_jerky_rule_rejects_candidate_of_other_class():
    from jianjin.checks.batch import jerky

    with pytest.raises(ValueError):
        jerky(normals(), replace(MOT, jerky_candidate="peak_speed", jerky_threshold=1.5))
