import numpy as np

from helpers import assert_bilingual, make_episode
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
    assert_bilingual(f.evidence)


def test_missing_values_checks_action_too():
    a = ramp()
    a[0, 0] = np.inf
    (f,) = missing_values(make_episode(ramp(), action=a), CFG)
    assert_bilingual(f.evidence)


def test_clean_episode_has_no_integrity_findings():
    ep = make_episode(ramp())
    assert missing_values(ep, CFG) == []
    assert too_short(ep, CFG) == []
    assert timestamp_gaps(ep, CFG) == []


def test_too_short():
    fs = too_short(make_episode(ramp(n=20)), CFG)
    assert len(fs) == 1 and fs[0].hard
    assert_bilingual(fs[0].evidence)


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
    assert_bilingual(fs[0].evidence)


def test_timestamp_going_backwards_is_a_gap():
    ep = make_episode(ramp())
    ts = ep.timestamps.copy()
    ts[50] = ts[48]
    ep.timestamps = ts
    fs = timestamp_gaps(ep, CFG)
    assert len(fs) >= 1
    assert_bilingual(fs[0].evidence)


from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from jianjin.checks.config import NotApplicable  # noqa: E402
from jianjin.checks.integrity import video_length  # noqa: E402
from jianjin.types import VideoRef  # noqa: E402


def with_videos(ep, **lengths):
    """lengths：相机名 -> 元数据里的视频时长（秒），None 是没写。"""
    ep.cameras = {cam: VideoRef(Path(f"{cam}.mp4"), 10.0, None if n is None else 10.0 + n)
                  for cam, n in lengths.items()}
    return ep


def test_video_as_long_as_joints_is_fine():
    ep = with_videos(make_episode(ramp()), top=3.0, wrist=3.0 + 1.0 / 30)  # 差一帧以内不算
    assert video_length(ep, CFG) == []


def test_video_longer_than_joints_needs_a_look():
    ep = with_videos(make_episode(ramp()), top=11.1, wrist=11.1)
    (f,) = video_length(ep, CFG)
    assert f.cause is Cause.DATA_ERROR and not f.hard and f.source == "meta"
    assert (f.t0, f.t1) == (0.0, ep.duration)
    assert "长 8.1 秒" in f.evidence and "2 路相机都这样" in f.evidence
    assert_bilingual(f.evidence)


def test_video_shorter_than_joints_is_hard():
    ep = with_videos(make_episode(ramp()), top=2.0, wrist=3.0)
    (f,) = video_length(ep, CFG)
    assert f.hard and "短 1.0 秒" in f.evidence and "top" in f.evidence
    assert_bilingual(f.evidence)
    assert f.t0 == pytest.approx(2.0) and f.t1 == pytest.approx(3.0)


def test_unknown_video_length_is_not_checked():
    with pytest.raises(NotApplicable) as exc:
        video_length(with_videos(make_episode(ramp()), left=None), CFG)
    from jianjin.checks.config import reason

    assert_bilingual(reason(exc.value))


def test_no_video_nothing_to_compare():
    assert video_length(make_episode(ramp()), CFG) == []


def test_runner_runs_video_length():
    from jianjin.checks import run_checks

    (r,) = run_checks([with_videos(make_episode(ramp()), top=11.1)])
    assert [f.check for f in r.findings] == ["video_length"]
