import numpy as np

from helpers import make_episode
from jianjin.checks.batch import end_pose, extra_motion
from jianjin.checks.config import CheckConfig
from jianjin.types import Cause

CFG = CheckConfig()


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
    assert end_pose(a + b, CFG)[0] == {}


def test_end_pose_flags_stopping_halfway():
    eps = [make_episode(there(), ep_id=f"ok{i}") for i in range(5)]
    eps.append(make_episode(there()[:30], ep_id="half"))
    found, skipped = end_pose(eps, CFG)
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
    assert end_pose(halves, CFG)[0] == {}
    found, _ = end_pose(halves, CFG, reference=good)
    assert set(found) == {f"half{i}" for i in range(5)}
