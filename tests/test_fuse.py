import numpy as np
import pytest

from helpers import make_episode
from jianjin.checks import NotApplicable
from jianjin.scores.fuse import fuse, robust_stats


def const(v):
    return lambda ep: np.full(len(ep.state), float(v))


def test_robust_stats_pools_all_frames():
    eps = [make_episode(np.zeros(4)), make_episode(np.zeros(6))]
    med, iqr = robust_stats(lambda ep: np.arange(len(ep.state), dtype=float), eps)
    pooled = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4, 5], float)
    assert med == np.median(pooled)
    assert iqr == pytest.approx(np.subtract(*np.percentile(pooled, [75, 25])))


def test_robust_stats_skips_not_applicable_and_guards_zero_spread():
    def picky(ep):
        if ep.id == "bad":
            raise NotApplicable("x")
        return np.ones(len(ep.state))

    med, iqr = robust_stats(picky, [make_episode(np.zeros(3), ep_id="bad"), make_episode(np.zeros(3))])
    assert med == 1.0 and iqr == 1.0  # 没有离散度时不缩放，免得除零


def test_fuse_puts_parts_on_common_scale_with_equal_weight():
    ep = make_episode(np.zeros(5))
    small = lambda ep: np.array([0, 0, 0, 0, 0.01])  # 量级小，统一尺度后照样起作用
    big = lambda ep: np.array([0, 100, 0, 0, 0], float)
    f = fuse([small, big], [(0.0, 0.01), (0.0, 100.0)])
    assert f(ep).tolist() == pytest.approx([0, 0.5, 0, 0, 0.5])


def test_fuse_not_applicable_if_any_part_is():
    def nope(ep):
        raise NotApplicable("x")

    with pytest.raises(NotApplicable):
        fuse([const(1), nope], [(0, 1), (0, 1)])(make_episode(np.zeros(3)))
