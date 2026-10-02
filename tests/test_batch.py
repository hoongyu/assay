import numpy as np

from helpers import make_episode
from jianjin.checks.batch import extra_motion
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
    found, skipped = extra_motion(eps, CFG)
    assert found == {}
    assert set(skipped) == {"e0", "e1", "e2"}
    assert "3" in skipped["e0"]


def test_tasks_compared_separately():
    a = [make_episode(there(), ep_id=f"a{i}", task="A") for i in range(5)]
    b = [make_episode(np.linspace(0, 1, 60) * 5, ep_id=f"b{i}", task="B") for i in range(5)]
    found, _ = extra_motion(a + b, CFG)
    assert found == {}
