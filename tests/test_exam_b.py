import math

import numpy as np

from helpers import make_robovad
from jianjin.eval.exam_b import SETTINGS, format_table, labels_from_annotations, run_exam_b


def blocked(n, lo, hi):
    action = np.tile(np.linspace(0, 1, n)[:, None], (1, 6))
    state = action.copy()
    state[lo:hi, 0] -= 0.3
    return state, action


def setup(tmp_path):
    eps = {
        "h_stackcubes_ep1": blocked(150, 60, 80),
        "h_stackcubes_ep2": blocked(150, 0, 0),
        "h_pour_rice_ep1": blocked(150, 100, 120),
    }
    ann = {
        "h_stackcubes_ep1": [{"start_frame": 60, "end_frame": 80, "type": "collision", "categories": {}}],
        "h_pour_rice_ep1": [{"start_frame": 100, "end_frame": 120, "type": "collision", "categories": {}}],
    }
    rows = [
        ("h_stackcubes_ep1", 0, 60, 0), ("h_stackcubes_ep1", 60, 80, 1), ("h_stackcubes_ep1", 90, 150, 0),
        ("h_stackcubes_ep2", 0, 150, 0),
        ("missing_ep", 0, 10, 0),
    ]
    evaluation = {s: rows for s in SETTINGS}
    return make_robovad(tmp_path / "rv", eps, ann, evaluation, validation=["h_pour_rice_ep1", "h_stackcubes_ep2"])


def test_labels_from_annotations():
    y = labels_from_annotations(10, [{"start_frame": 2, "end_frame": 4}, {"start_frame": 8, "end_frame": 20}])
    assert y.tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 1, 1]


def test_test_split_uses_listed_frames_only(tmp_path):
    out = run_exam_b(setup(tmp_path), "test", "tracking")
    assert [r["setting"] for r in out["results"]] == SETTINGS
    r = out["results"][0]
    assert r["n_episodes"] == 2
    assert r["n_frames"] == 60 + 20 + 60 + 150  # 80 到 90 帧没列出，不评
    assert r["n_anomalous"] == 20
    assert r["missing_episodes"] == ["missing_ep"]
    assert r["micro_auc"] > 0.95
    assert r["n_macro_episodes"] == 1  # 只有 ep1 同时有正常和异常帧


def test_validation_split_from_annotations(tmp_path):
    out = run_exam_b(setup(tmp_path), "validation", "tracking")
    (r,) = out["results"]
    assert r["setting"] == "validation"
    assert r["n_episodes"] == 2 and r["n_frames"] == 300 and r["n_anomalous"] == 20
    assert r["micro_auc"] > 0.95


def test_table_has_denominators_and_paper(tmp_path):
    out = run_exam_b(setup(tmp_path), "test", "tracking")
    t = format_table(out)
    assert "零样本" in t and "FIDeL" in t and "帧" in t
    assert not math.isnan(out["results"][0]["micro_auc"])
