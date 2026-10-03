import numpy as np
import pytest

from helpers import make_mp4
from jianjin.types import VideoRef
from jianjin.video import read_clip


@pytest.fixture
def mp4(tmp_path):
    """10 秒、30 fps、64×48，第 i 帧的亮度是 i。"""
    return make_mp4(tmp_path / "v.mp4")


def test_samples_at_requested_rate(mp4):
    frames, times = read_clip(VideoRef(mp4, 0.0, 10.0), sample_fps=2.0)
    assert frames.dtype == np.uint8 and frames.shape[1:] == (48, 64, 3)
    assert len(frames) == 20
    np.testing.assert_allclose(times[:3], [0.0, 0.5, 1.0], atol=0.04)


def test_reads_only_this_episode(mp4):
    frames, times = read_clip(VideoRef(mp4, 4.0, 6.0), sample_fps=2.0)
    assert len(frames) == 4
    assert times[0] < 0.05  # 相对这条示范的开头
    assert abs(float(frames[0].mean()) - 120) < 12  # 第 4 秒是第 120 帧


def test_downscales(mp4):
    frames, _ = read_clip(VideoRef(mp4, 0.0, 1.0), sample_fps=2.0, max_side=32)
    assert max(frames.shape[1:3]) == 32


def test_open_ended_ref(mp4):
    frames, _ = read_clip(VideoRef(mp4, 0.0, None), sample_fps=1.0)
    assert len(frames) == 10


from jianjin.video import frames_at  # noqa: E402


def test_frames_at_takes_one_frame_per_time(mp4):
    imgs = frames_at(VideoRef(mp4, 4.0, 6.0), [0.0, 0.5, 1.0], max_side=32)
    assert imgs.shape == (3, 24, 32, 3)
    assert [round(float(i.mean())) for i in imgs] == pytest.approx([120, 135, 150], abs=12)  # 第 120、135、150 帧
