import json

import numpy as np
import pytest

from helpers import make_episode, make_mp4
from jianjin.checks import NotApplicable
from jianjin.judges import JUDGES
from jianjin.judges.base import Curve, FakeJudge
from jianjin.judges.cache import fingerprint, judge_episode, load_curve
from jianjin.types import VideoRef


@pytest.fixture
def ep(tmp_path):
    e = make_episode(np.zeros((300, 2)))
    e.cameras = {"front": VideoRef(make_mp4(tmp_path / "v.mp4"), 0.0, 10.0)}
    return e


def test_fake_judge_curve_rises_with_brightness(ep, tmp_path):
    c = judge_episode(FakeJudge(), ep, "front", tmp_path / "cache", sample_fps=1.0)
    assert isinstance(c, Curve)
    assert len(c.times) == len(c.progress) == 10
    assert c.progress[-1] > c.progress[0]
    assert c.model == "fake" and c.version == "1"


def test_second_call_hits_cache(ep, tmp_path):
    j = FakeJudge()
    judge_episode(j, ep, "front", tmp_path / "cache", sample_fps=1.0)
    judge_episode(j, ep, "front", tmp_path / "cache", sample_fps=1.0)
    assert j.calls == 1


def test_new_version_does_not_reuse_cache(ep, tmp_path):
    judge_episode(FakeJudge(), ep, "front", tmp_path / "cache", sample_fps=1.0)
    j2 = FakeJudge(version="2")
    judge_episode(j2, ep, "front", tmp_path / "cache", sample_fps=1.0)
    assert j2.calls == 1


def test_load_curve_checks_fingerprint(ep, tmp_path):
    judge_episode(FakeJudge(), ep, "front", tmp_path / "cache", sample_fps=1.0)
    assert load_curve(tmp_path / "cache", "fake", ep, "front") is not None
    ep.task = "something else"  # 任务描述变了，旧结果不能用
    assert load_curve(tmp_path / "cache", "fake", ep, "front") is None


def test_new_sampling_reruns(ep, tmp_path):
    j = FakeJudge()
    judge_episode(j, ep, "front", tmp_path / "cache", sample_fps=1.0)
    judge_episode(j, ep, "front", tmp_path / "cache", sample_fps=2.0)
    assert j.calls == 2


def test_cache_file_is_plain_json(ep, tmp_path):
    judge_episode(FakeJudge(), ep, "front", tmp_path / "cache", sample_fps=1.0)
    (f,) = list((tmp_path / "cache" / "fake").glob("*.json"))
    data = json.loads(f.read_text(encoding="utf-8"))
    assert {"key", "model", "version", "times", "progress", "sample_fps", "max_side"} <= set(data)


def test_no_camera(tmp_path):
    with pytest.raises(NotApplicable):
        judge_episode(FakeJudge(), make_episode(np.zeros((30, 2))), None, tmp_path)


def test_fingerprint_changes_with_slice(ep):
    ref = ep.cameras["front"]
    assert fingerprint(ref) != fingerprint(VideoRef(ref.path, 1.0, 10.0))


def test_registry_has_fake():
    assert "fake" in JUDGES


def test_load_curve_without_local_video(ep, tmp_path):
    """本机没下视频时（比如 RoboVAD 的视频只在 Colab 上），按示范名、相机、任务描述认缓存。"""
    judge_episode(FakeJudge(), ep, "front", tmp_path / "cache", sample_fps=1.0)
    ep.cameras = {}
    assert load_curve(tmp_path / "cache", "fake", ep, "front") is not None
    ep.task = "something else"
    assert load_curve(tmp_path / "cache", "fake", ep, "front") is None
