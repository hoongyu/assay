import numpy as np

from helpers import make_episode
from jianjin.types import Cause, Finding


def test_duration_and_gripper():
    state = np.zeros((60, 3))
    state[:, 2] = np.linspace(0, 1, 60)
    ep = make_episode(state, fps=30, names=["a", "b", "gripper.pos"])
    assert ep.duration == 2.0
    assert ep.gripper_index == 2
    np.testing.assert_allclose(ep.gripper, state[:, 2])


def test_no_gripper():
    ep = make_episode(np.zeros((30, 2)), names=["a", "b"])
    assert ep.gripper is None


def test_finding_defaults_soft():
    f = Finding("x", Cause.LONG_PAUSE, 1.0, 2.0, 1.0, "停了")
    assert f.hard is False
    assert Cause.DATA_ERROR.value == "数据异常"
