import numpy as np

from helpers import make_episode
from assay.checks import CheckConfig
from assay.checks.motion import long_pause
from assay.eval.inject import insert_pause, truncate_end
from assay.types import Cause, VideoRef


def base():
    x = np.linspace(0, 1, 300)  # 10 秒一直在动
    ep = make_episode(np.stack([x, x], axis=1), action=np.stack([x, x], axis=1))
    ep.cameras = {"front": VideoRef("v.mp4", 4.0, 14.0)}
    return ep


def test_truncate_end():
    rng = np.random.default_rng(0)
    ep, d = truncate_end(base(), rng)
    assert d.cause is Cause.INCOMPLETE and d.kind == "truncate_end"
    assert 150 <= len(ep.state) <= 240
    assert len(ep.action) == len(ep.state) == len(ep.timestamps)
    assert np.isclose(d.t1, ep.duration) and np.isclose(d.t0, ep.duration - 1.0)
    assert np.isclose(ep.cameras["front"].to_s, 4.0 + ep.duration)
    assert ep.id.endswith("+truncate_end")


def test_insert_pause_is_detected_where_injected():
    rng = np.random.default_rng(1)
    ep, d = insert_pause(base(), rng)
    assert d.cause is Cause.LONG_PAUSE
    assert 4.0 <= d.t1 - d.t0 <= 8.0
    assert len(ep.action) == len(ep.state)
    np.testing.assert_allclose(np.diff(ep.timestamps), 1 / 30)
    assert ep.cameras == {}  # 视频没法跟着插帧，本周先不带视频
    fs = long_pause(ep, CheckConfig())
    assert len(fs) == 1
    assert abs(fs[0].t0 - d.t0) < 0.1 and abs(fs[0].t1 - d.t1) < 0.1


def test_injection_is_deterministic():
    a = insert_pause(base(), np.random.default_rng(7))[1]
    b = insert_pause(base(), np.random.default_rng(7))[1]
    assert a == b
