import numpy as np
import pytest

from helpers import make_robovad, write_cached_curve
from assay.eval.exam_b_episode import format_episode_b, run_episode_b
from assay.ingest.robovad import read_robovad

NORMAL = {f"h_stackcubes_ep{i}": d for i, d in enumerate([0.0, 0.05, 0.1, 0.2])}


def setup(tmp_path):
    names = [*NORMAL, "h_stackcubes_ep10", "h_stackcubes_ep11"]
    eps = {n: (np.zeros((150, 6)), np.zeros((150, 6))) for n in names}
    drop = [{"start_frame": 60, "end_frame": 90, "type": "object_drop", "categories": {}}]  # 第 2 到 3 秒
    root = make_robovad(tmp_path / "rv", eps, {"h_stackcubes_ep10": drop, "h_stackcubes_ep11": drop},
                        validation=names)
    cache = tmp_path / "cache"
    for ep in read_robovad(root).episodes:
        if ep.id in NORMAL:
            d = NORMAL[ep.id]
            write_cached_curve(cache, ep, "left", [0.1, 0.5, 0.5 - d, 0.9], times=[0, 1, 2, 4])
        elif ep.id == "h_stackcubes_ep10":  # 第 2 到 2.5 秒掉了 0.5，落在标注里
            write_cached_curve(cache, ep, "left", [0.1, 0.3, 0.8, 0.3, 0.6], times=[0, 1, 2, 2.5, 4])
        else:  # 只掉了 0.15，抓不到
            write_cached_curve(cache, ep, "left", [0.1, 0.6, 0.45, 0.9], times=[0, 1, 2, 4])
    return root, cache


def test_threshold_from_normal_episodes(tmp_path):
    root, cache = setup(tmp_path)
    out = run_episode_b(root, "validation", cache)
    assert out["threshold"] == pytest.approx(0.2) and out["chosen_on_this_split"]
    assert (out["n_episodes"], out["n_anomalous"], out["missing"]) == (6, 2, 0)
    assert out["detected"] == 1 and out["localized"] == 1 and out["normal_flagged"] == 0
    assert "分母" in format_episode_b(out)


def test_given_threshold_is_used_as_is(tmp_path):
    root, cache = setup(tmp_path)
    out = run_episode_b(root, "validation", cache, threshold=0.6)
    assert out["detected"] == 0 and not out["chosen_on_this_split"]


def test_fall_while_returning_home_does_not_count(tmp_path):
    x = np.concatenate([np.linspace(0, 1, 50), np.ones(50), np.linspace(1, 0, 50)])  # 5 秒：出去、停住、回原位
    arm = np.stack([x] * 5 + [np.zeros(150)], axis=1)
    names = [f"h_stackcubes_ep{i}" for i in range(5)]
    drop = [{"start_frame": 30, "end_frame": 60, "type": "object_drop", "categories": {}}]  # 第 1 到 2 秒
    root = make_robovad(tmp_path / "rv", {n: (arm, arm) for n in names}, {"h_stackcubes_ep4": drop},
                        validation=names)
    cache = tmp_path / "cache"
    for ep in read_robovad(root).episodes:
        if ep.id == "h_stackcubes_ep4":  # 还在远处就掉了
            write_cached_curve(cache, ep, "left", [0.1, 0.9, 0.3, 0.5, 0.2], times=[0, 1, 1.5, 2, 5])
        else:  # 正常示范：约 3.8 秒开始回原位，之后进度才掉
            write_cached_curve(cache, ep, "left", [0.1, 0.5, 0.9, 0.9, 0.2], times=[0, 1, 2, 3, 5])
    out = run_episode_b(root, "validation", cache)
    assert out["threshold"] == pytest.approx(0.0)
    assert out["detected"] == 1 and out["normal_flagged"] == 0
