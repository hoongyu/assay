import json

import numpy as np
import pytest

from helpers import make_episode, make_robomind_failure_h5
from assay.checks.config import CheckConfig


def minjerk(n, dist=1.0):
    t = np.linspace(0, 1, n)
    return dist * (10 * t**3 - 15 * t**4 + 6 * t**5)


def arm6(path, grip):
    """六个手臂关节都按 path 走（各乘个系数），夹爪另给；前后各停 15 帧。"""
    x = np.concatenate([np.zeros(15), path, np.full(15, path[-1])])
    return x[:, None] * np.linspace(0.5, 1.0, 6), grip(len(x))[:, None]


def closes(times):
    """夹爪开合 times 回（合上、再张开），两头张开。"""
    def f(n):
        g = np.ones(n)
        for k in range(times):
            a = int(n * (2 * k + 1) / (2 * times + 1))
            g[a: a + max(3, n // (4 * times + 2))] = 0.0
        return g
    return f


def stop_and_go():
    parts = []
    for k in range(4):
        parts.append(k * 0.25 + minjerk(30, 0.25))
        if k < 3:
            parts.append(np.full(15, (k + 1) * 0.25))
    return np.concatenate(parts)


def make_tree(root):
    """两个任务各：六条二次抓取以外的普通失败（动作正常），三条走走停停的动作不流畅，两条快一倍的速度过快，两条二次抓取。"""
    for task in ("t_dev", "t_test"):
        for i, a in enumerate([1.0, 1.05, 0.95, 1.02, 0.98, 1.01]):
            arm, hand = arm6(minjerk(90, a), closes(1))
            make_robomind_failure_h5(root / f"{task}/夹取前碰撞/n{i}/data/trajectory.hdf5", arm=arm, hand=hand)
        for i in range(3):
            arm, hand = arm6(stop_and_go(), closes(1))
            make_robomind_failure_h5(root / f"{task}/动作不流畅/j{i}/data/trajectory.hdf5", arm=arm, hand=hand)
        for i in range(2):
            arm, hand = arm6(minjerk(45), closes(1))
            make_robomind_failure_h5(root / f"{task}/速度过快/f{i}/data/trajectory.hdf5", arm=arm, hand=hand)
        for i in range(2):
            arm, hand = arm6(minjerk(90), closes(3))
            make_robomind_failure_h5(root / f"{task}/二次抓取/r{i}/data/trajectory.hdf5", arm=arm, hand=hand)
    split = tmp = root.parent / "split.json"
    tmp.write_text(json.dumps({"tasks": {"t_dev": "dev", "t_test": "test"}}), encoding="utf-8")
    return split


def success_sets():
    """两个成功数据集：动作都正常，幅度略有出入。"""
    def ep(i, a):
        x = np.concatenate([np.zeros(15), minjerk(90, a), np.full(15, a)])
        return make_episode(np.stack([x, 0.5 * x, np.zeros_like(x)], axis=1), names=["j0", "j1", "gripper"],
                            ep_id=f"s{i}")
    amps = [1.0, 1.05, 0.95, 1.02, 0.98, 1.01, 0.97, 1.03]
    return {"A": [ep(i, a) for i, a in enumerate(amps)], "B": [ep(i, a) for i, a in enumerate(amps[::-1])]}


def test_pick_threshold_is_smallest_grid_value_with_every_set_at_most_5pct():
    from assay.eval.exam_c import pick_threshold

    t, flags = pick_threshold({"A": [1.0] * 19 + [1.35], "B": [1.0] * 9 + [1.25]}, "ratio")
    assert t == pytest.approx(1.3)  # B 那条 1.25 要压住（1/10 超过 5%），A 那条 1.35 可以留（1/20）
    assert flags == {"A": {"k": 1, "n": 20}, "B": {"k": 0, "n": 10}}
    t, _ = pick_threshold({"A": [-1, 0, 0, 1, 3]}, "count")
    assert t == 3
    t, _ = pick_threshold({"A": [9.0] * 5}, "ratio")
    assert t is None  # 格子里没有满足的，这个候选出局


def test_choose_prefers_baseline_unless_strictly_better():
    from assay.eval.exam_c import choose

    def m(gap, auc, t=1.5):
        return {"threshold": t, "gap": gap, "auc": auc}

    cands = {"duration": m(0.30, 0.70), "stops": m(0.40, 0.60), "bursts": m(0.40, 0.65), "jerk": m(0.1, 0.9),
             "duration_short": m(0.20, 0.80), "peak_speed": m(0.20, 0.95), "mean_speed": m(0.1, 0.5)}
    chosen = choose(cands)
    assert chosen["jerky"] == {"candidate": "bursts", "threshold": 1.5}  # 差打平看 AUC
    assert chosen["fast"] == {"candidate": "duration_short", "threshold": 1.5}  # 没有严格大于基线
    cands["duration_short"] = m(0.0, 0.5)
    cands["peak_speed"] = m(-0.1, 0.4, t=None)
    cands["mean_speed"] = m(-0.05, 0.4)
    assert choose(cands)["fast"] is None  # 时长基线的差也不大于 0：这一类不加规则


def test_run_exam_c_dev_then_test(tmp_path):
    from assay.eval.exam_c import run_exam_c

    split = make_tree(tmp_path / "failure_data")
    cfg = CheckConfig()
    dev = run_exam_c(tmp_path / "failure_data", "dev", success_sets(), split_file=split, cfg=cfg)
    assert dev["chosen_on_this_split"] and dev["counts"]["episodes"] == 13
    stops = dev["candidates"]["stops"]
    assert stops["target"]["k"] == 3 and stops["target"]["n"] == 3 and stops["others"]["k"] == 0
    assert stops["others"]["n"] == 8  # 夹取前碰撞 6 条、二次抓取 2 条
    assert dev["chosen"]["jerky"] is not None and dev["chosen"]["fast"] is not None
    with pytest.raises(ValueError):
        run_exam_c(tmp_path / "failure_data", "test", success_sets(), split_file=split, cfg=cfg)
    test = run_exam_c(tmp_path / "failure_data", "test", success_sets(), split_file=split, cfg=cfg, dev_result=dev)
    assert not test["chosen_on_this_split"] and test["chosen"] == dev["chosen"]
    jerky = test["result"]["jerky"]
    assert jerky["target"]["k"] == 3 and jerky["others"]["k"] == 0
    assert set(test["result"]["jerky"]["success"]) == {"A", "B"}
    assert "self_reference" in test["result"]["jerky"]


def test_regrasp_on_robomind_uses_other_causes_as_reference(tmp_path):
    from assay.eval.exam_c import run_regrasp_c

    make_tree(tmp_path / "failure_data")
    out = run_regrasp_c(tmp_path / "failure_data", CheckConfig())
    assert out["target"]["k"] == 4 and out["target"]["n"] == 4  # 两个任务各两条
    assert out["others"]["k"] == 0 and out["others"]["n"] == 22
