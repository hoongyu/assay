import numpy as np

from helpers import make_episode
from jianjin.checks.config import CheckConfig
from jianjin.checks.integrity import missing_values, timestamp_gaps, too_short
from jianjin.types import Cause

CFG = CheckConfig()


def ramp(n=90, d=2):
    return np.tile(np.linspace(0, 1, n)[:, None], (1, d))


def test_missing_values_found_and_hard():
    s = ramp()
    s[30:33, 1] = np.nan
    fs = missing_values(make_episode(s), CFG)
    assert len(fs) == 1
    f = fs[0]
    assert f.cause is Cause.DATA_ERROR and f.hard
    assert np.isclose(f.t0, 1.0) and np.isclose(f.t1, 1.1)


def test_missing_values_checks_action_too():
    a = ramp()
    a[0, 0] = np.inf
    assert len(missing_values(make_episode(ramp(), action=a), CFG)) == 1


def test_clean_episode_has_no_integrity_findings():
    ep = make_episode(ramp())
    assert missing_values(ep, CFG) == []
    assert too_short(ep, CFG) == []
    assert timestamp_gaps(ep, CFG) == []


def test_too_short():
    fs = too_short(make_episode(ramp(n=20)), CFG)
    assert len(fs) == 1 and fs[0].hard


def test_timestamp_gap():
    ep = make_episode(ramp())
    ts = ep.timestamps.copy()
    ts[45:] += 0.5  # 第 1.5 秒附近丢了约 15 帧
    ep.timestamps = ts
    fs = timestamp_gaps(ep, CFG)
    assert len(fs) == 1
    assert fs[0].hard and fs[0].cause is Cause.DATA_ERROR
    assert np.isclose(fs[0].t0, 44 / 30)
    assert "15" in fs[0].evidence


def test_timestamp_going_backwards_is_a_gap():
    ep = make_episode(ramp())
    ts = ep.timestamps.copy()
    ts[50] = ts[48]
    ep.timestamps = ts
    assert len(timestamp_gaps(ep, CFG)) >= 1
