import numpy as np
import pytest

from helpers import make_episode
from jianjin.checks import NotApplicable
from jianjin.scores import SCORERS
from jianjin.scores.frame import accel, calm, gripper_tracking, stillness, tracking

NAMES = ["a", "b", "gripper"]


def follower(n=120):
    """从臂完全跟住主臂，只有第 50 到 60 帧被挡住（像撞到东西）。"""
    action = np.stack([np.linspace(0, 1, n), np.linspace(0, 2, n), np.linspace(0, 1, n)], axis=1)
    state = action.copy()
    state[50:60, 0] -= 0.3
    return state, action


def test_tracking_peaks_where_blocked():
    state, action = follower()
    s = tracking(make_episode(state, names=NAMES, action=action))
    assert s.shape == (120,)
    assert 45 <= int(np.argmax(s)) <= 64
    assert s[:40].max() < s[50:60].min()


def test_gripper_tracking_ignores_other_joints():
    state, action = follower()
    s = gripper_tracking(make_episode(state, names=NAMES, action=action))
    assert np.allclose(s, 0)
    state[80:90, 2] += 0.2  # 夹爪被物体挡住
    s = gripper_tracking(make_episode(state, names=NAMES, action=action))
    assert 75 <= int(np.argmax(s)) <= 94


def test_accel_peaks_at_jerk():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(60)])  # 第 60 帧急停
    s = accel(make_episode(np.stack([x, x], axis=1)))
    assert s.shape == (120,)
    assert 55 <= int(np.argmax(s)) <= 64


def test_stillness_high_when_still():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(60)])
    s = stillness(make_episode(np.stack([x, x], axis=1)))
    assert s[90] > s[30]


def test_needs_action():
    state, _ = follower()
    with pytest.raises(NotApplicable):
        tracking(make_episode(state, names=NAMES))


def test_registry():
    assert set(SCORERS) == {"tracking", "gripper_tracking", "accel", "calm", "stillness"}


def test_calm_is_negative_accel():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(60)])
    ep = make_episode(np.stack([x, x], axis=1))
    np.testing.assert_allclose(calm(ep), -accel(ep))
