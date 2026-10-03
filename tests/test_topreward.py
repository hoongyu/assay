import math

import numpy as np
import pytest

from jianjin.judges import JUDGES
from jianjin.judges.topreward import TOPRewardJudge, anchors, min_max


class FakeScore:
    """假的打分函数：对数概率随前缀变长线性上升，记下每次看到的帧数和帧率。"""

    def __init__(self):
        self.calls = []

    def __call__(self, frames, task, fps):
        self.calls.append((len(frames), task, fps))
        return -3.0 + 0.1 * len(frames)


def test_anchors_are_prefix_lengths_from_two_frames():
    assert anchors(1, 15).tolist() == [1]
    assert anchors(5, 15).tolist() == [2, 3, 4, 5]
    a = anchors(32, 15)
    assert len(a) == 15 and a[0] == 2 and a[-1] == 32


def test_min_max_like_upstream():
    assert min_max([1.0, 3.0, 2.0]).tolist() == [0.0, 1.0, 0.5]
    assert min_max([2.0, 2.0]).tolist() == [1.0, 1.0]


def test_curve_scores_growing_prefixes():
    fake = FakeScore()
    j = TOPRewardJudge(max_frames=8, n_anchors=4, score_fn=fake)
    frames = np.zeros((20, 8, 8, 3), np.uint8)
    times = np.arange(20) * 0.5
    c = j.curve(frames, times, "pick")
    lengths = [n for n, _, _ in fake.calls]
    assert lengths == [2, 4, 6, 8] and all(t == "pick" for _, t, _ in fake.calls)
    # 20 帧里均匀取 8 帧：第 0、3、5、8、11、14、16、19 帧
    assert fake.calls[0][2] == pytest.approx(7 / 9.5)  # 抽帧后的实际帧率
    assert c.times == [1.5, 4.0, 7.0, 9.5]
    assert c.progress == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert c.success == pytest.approx([math.exp(-3.0 + 0.1 * n) for n in lengths])
    assert c.model == "topreward" and "Qwen3-VL-4B-Instruct" in c.version and "f8a4" in c.version


def test_non_finite_logprob_is_an_error():
    j = TOPRewardJudge(score_fn=lambda frames, task, fps: float("nan"))
    with pytest.raises(RuntimeError):
        j.curve(np.zeros((4, 8, 8, 3), np.uint8), np.arange(4) * 0.5, "pick")


def test_registered():
    assert JUDGES["topreward"] is TOPRewardJudge
