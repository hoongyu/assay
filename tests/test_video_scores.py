import numpy as np
import pytest

from helpers import make_episode
from assay.checks import NotApplicable
from assay.judges.base import Curve
from assay.scores.video import CURVE_SCORES, per_frame


def ep300():
    return make_episode(np.zeros((300, 2)))  # 10 秒


def test_per_frame_interpolates():
    c = Curve("m", "1", [0.0, 5.0, 10.0], [0.0, 0.5, 1.0])
    p = per_frame(c.times, c.progress, ep300())
    assert p.shape == (300,)
    assert abs(p[150] - 0.5) < 0.01


def test_drop_peaks_where_progress_falls():
    c = Curve("m", "1", [0, 2, 4, 6, 8, 10], [0.0, 0.4, 0.8, 0.3, 0.6, 1.0])
    s = CURVE_SCORES["drop"](c, ep300())
    assert 120 <= int(np.argmax(s)) <= 180


def test_low_is_one_minus_progress():
    c = Curve("m", "1", [0.0, 10.0], [0.2, 0.2])
    np.testing.assert_allclose(CURVE_SCORES["low"](c, ep300()), 0.8)


def test_fail_needs_success_head():
    c = Curve("m", "1", [0.0, 10.0], [0.0, 1.0])
    with pytest.raises(NotApplicable):
        CURVE_SCORES["fail"](c, ep300())
    c2 = Curve("m", "1", [0.0, 10.0], [0.0, 1.0], success=[0.9, 0.1])
    s = CURVE_SCORES["fail"](c2, ep300())
    assert s[-1] > s[0]
