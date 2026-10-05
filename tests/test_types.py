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


def test_bilingual_text_is_the_chinese_string_with_english_attached():
    import copy
    import json

    from jianjin.types import Bi, en

    t = Bi("停了 3.0 秒", "Paused for 3.0 s")
    assert t == "停了 3.0 秒" and isinstance(t, str) and en(t) == "Paused for 3.0 s"
    assert en("只有中文") == "只有中文"  # 没有英文的退回原文
    assert en(copy.deepcopy(t)) == "Paused for 3.0 s"
    assert json.loads(json.dumps({"x": t}, ensure_ascii=False)) == {"x": "停了 3.0 秒"}


def test_reason_keeps_the_english_of_a_not_applicable():
    from jianjin.checks.config import NotApplicable, reason
    from jianjin.types import Bi, en

    exc = NotApplicable(Bi("认不出夹爪通道", "No gripper channel found"))
    assert reason(exc) == "认不出夹爪通道" and en(reason(exc)) == "No gripper channel found"
    assert reason(NotApplicable("老写法")) == "老写法"
