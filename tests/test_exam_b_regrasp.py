import json

import numpy as np
import pytest

from helpers import make_robovad
from jianjin.eval.exam_b_regrasp import format_regrasp_b, run_regrasp_b


def arm(cycles: int, seg: int = 10) -> np.ndarray:
    """6 维：前 5 个关节匀速走，夹爪（第 6 维）从合着开始开合 cycles 回。"""
    parts = [np.zeros(5)] + [np.r_[np.full(seg, 30.0), np.zeros(seg)] for _ in range(cycles)] + [np.zeros(5)]
    g = np.concatenate(parts)
    joints = np.linspace(0, 1, len(g))[:, None] * np.arange(1, 6)
    return np.column_stack([joints, g])


def fixture(tmp_path):
    train = [f"h_pickandplace_ep{i}" for i in range(13)]
    test = [f"h_pickandplace_ep{i}" for i in range(20, 28)] + [f"h_pour_rice_ep{i}" for i in range(7)]
    eps = {name: (arm(2), arm(2)) for name in train + test}
    ann = {}
    for name, kind in [("h_pickandplace_ep10", "failed_grasp"), ("h_pickandplace_ep11", "object_drop"),
                       ("h_pickandplace_ep12", "failed_grasp"), ("h_pickandplace_ep25", "failed_grasp"),
                       ("h_pickandplace_ep26", "object_drop"), ("h_pickandplace_ep27", "collision"),
                       ("h_pour_rice_ep6", "failed_grasp")]:
        eps[name] = (arm(4), arm(4))
        ann[name] = [{"start_frame": 5, "end_frame": 15, "type": kind, "categories": {}}]
    return make_robovad(tmp_path / "rv", eps, ann, splits={"train": train, "test": test})


def test_choose_on_train_then_reuse_on_test(tmp_path):
    root = fixture(tmp_path)
    dev = run_regrasp_b(root, "train")
    assert dev["chosen_on_this_split"] and len(dev["candidates"]) == 14
    assert dev["source"] == "action" and dev["margin"] == 3  # 余量 0 到 3 都全检出、不误报，打平取大的
    assert dev["seen"]["target_detected"]["k"] == 3 and dev["seen"]["normal_flagged"]["k"] == 0
    assert dev["unseen"] is None
    test = run_regrasp_b(root, "test", source=dev["source"], margin=dev["margin"])
    assert not test["chosen_on_this_split"] and test["candidates"] == []
    seen, unseen = test["seen"], test["unseen"]
    assert seen["target_detected"]["k"] == 2 and seen["target_detected"]["n"] == 2
    assert seen["normal_flagged"]["k"] == 0 and seen["normal_flagged"]["n"] == 5
    assert seen["by_type"]["collision"]["k"] == 1 and seen["anomalous_detected"]["k"] == 3
    assert unseen["target_detected"]["k"] == 1 and unseen["normal_flagged"]["k"] == 0
    assert test["unseen_tasks"] == ["Pour the rice into the bowl."]
    assert "抓空或掉落" in format_regrasp_b(test)


def test_cli_needs_train_choice_for_test(tmp_path, monkeypatch):
    from jianjin.cli import main

    root = fixture(tmp_path)
    out = tmp_path / "train.json"
    assert main(["eval", "b-regrasp", "--robovad-root", str(root), "--split", "train", "--out", str(out)]) == 0
    dev = json.loads(out.read_text(encoding="utf-8"))
    assert dev["source"] == "action" and dev["margin"] == 3
    with pytest.raises(SystemExit):
        main(["eval", "b-regrasp", "--robovad-root", str(root), "--split", "test"])
    monkeypatch.chdir(tmp_path)
    assert main(["eval", "b-regrasp", "--robovad-root", str(root), "--split", "test", "--dev-result", str(out)]) == 0
    assert (tmp_path / "results" / "exam_b_regrasp_test.json").exists()
