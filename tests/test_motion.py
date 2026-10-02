import numpy as np
from helpers import make_episode
from jianjin.checks.config import CheckConfig
from jianjin.checks.motion import long_pause
from jianjin.types import Cause

CFG = CheckConfig()


def two_joints(x):
    return np.stack([x, np.full_like(x, 5.0)], axis=1)  # 第二个关节不动，不该影响结果


def test_pause_in_middle_found():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(150), np.linspace(1, 2, 60)])
    fs = long_pause(make_episode(two_joints(x)), CFG)
    assert len(fs) == 1
    f = fs[0]
    assert f.cause is Cause.LONG_PAUSE and not f.hard
    assert abs(f.t0 - 2.0) < 0.1 and abs(f.t1 - 7.0) < 0.1


def test_still_at_start_and_end_is_normal():
    x = np.concatenate([np.zeros(150), np.linspace(0, 1, 60), np.ones(150)])
    assert long_pause(make_episode(two_joints(x)), CFG) == []


def test_short_pause_ignored():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(30), np.linspace(1, 2, 60)])
    assert long_pause(make_episode(two_joints(x)), CFG) == []
