import numpy as np

from helpers import make_episode
from jianjin.checks.batch import end_pose, extra_motion
from jianjin.checks.config import CheckConfig
from jianjin.types import Cause

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


def test_small_task_not_checked():
    eps = [make_episode(there(), ep_id=f"e{i}") for i in range(3)]
    for check in (extra_motion, end_pose):
        found, skipped = check(eps, CFG)
        assert found == {}
        assert set(skipped) == {"e0", "e1", "e2"}
        assert "3" in skipped["e0"]


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
