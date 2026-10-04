import json

import numpy as np
import pytest

from helpers import make_episode, make_lerobot_v3
from jianjin.eval.exam_a import DatasetSpec, load_config, local_dir
from jianjin.eval.inject import swap_task
from jianjin.eval.task_swap import SCORE_NAMES, format_swap, run_swap, swap_pairs
from jianjin.judges.base import Curve
from jianjin.judges.cache import _path
from jianjin.types import Cause

CAM = "observation.images.front"
N_EP = 6  # 每个数据集够长的示范条数；同任务比较至少要 5 条


def moving6(n, seed):
    rng = np.random.default_rng(seed)
    return np.tile(np.cumsum(rng.uniform(0.5, 1.5, n))[:, None], (1, 6))


def test_swap_task_takes_a_different_task_from_pool():
    ep = make_episode(np.zeros(90), task="pick")
    new, d = swap_task(ep, np.random.default_rng(0), ["pick", "fold the towel"])
    assert new.task == "fold the towel" and new.id == "ep0+swap_task"
    assert new.meta["original_task"] == "pick"
    assert d.cause == Cause.WRONG_TASK and (d.t0, d.t1) == (0.0, ep.duration)
    assert np.array_equal(new.state, ep.state)


def test_swap_task_needs_another_task():
    with pytest.raises(ValueError):
        swap_task(make_episode(np.zeros(90), task="pick"), np.random.default_rng(0), ["pick"])


def setup(tmp_path):
    root = tmp_path / "data"
    for repo, task in [("org/a", "pick"), ("org/b", "stack"), ("org/c", "fold")]:
        eps = [(task, moving6(300, i)) for i in range(N_EP)] + [(task, moving6(30, 9))]  # 最后一条 1 秒，太短
        make_lerobot_v3(local_dir(root, repo), eps)
    specs = [
        DatasetSpec("org/a", "dev", "mit", CAM),
        DatasetSpec("org/b", "dev", "mit", CAM),
        DatasetSpec("org/c", "test", "mit", CAM),
    ]
    return root, specs


def test_load_config_reads_camera(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[[dataset]]\nrepo="x/y"\nsplit="dev"\nlicense="mit"\ncamera="observation.images.top"\n'
                 '[[dataset]]\nrepo="x/z"\nsplit="test"\nlicense="mit"\n', encoding="utf-8")
    a, b = load_config(p)
    assert a.camera == "observation.images.top" and b.camera is None


def test_swap_pairs_swap_within_split_and_skip_short(tmp_path):
    root, specs = setup(tmp_path)
    pairs = swap_pairs(specs, root, "dev", seed=0)
    assert len(pairs) == 2 * N_EP
    for orig, swapped, cam in pairs:
        assert cam == CAM
        assert {orig.task, swapped.task} == {"pick", "stack"}


def test_swap_pairs_skip_datasets_without_camera_and_excluded_tasks(tmp_path):
    """没写相机的（只用关节数据的双臂开发集）不进换描述考卷，也不改变别的示范换到哪个描述；排除的任务也不用。"""
    root, specs = setup(tmp_path)
    before = [(o.id, s.task) for o, s, _ in swap_pairs(specs, root, "dev", seed=0)]
    make_lerobot_v3(local_dir(root, "org/d"), [("pour", moving6(300, i)) for i in range(N_EP)])
    make_lerobot_v3(local_dir(root, "org/e"), [("fold", moving6(300, i)) for i in range(N_EP)])
    more = [*specs, DatasetSpec("org/d", "dev", "mit"),
            DatasetSpec("org/e", "dev", "mit", CAM, exclude_tasks=("fold",))]
    assert [(o.id, s.task) for o, s, _ in swap_pairs(more, root, "dev", seed=0)] == before


def write_curve(cache, ep, cam, progress):
    path = _path(cache, "robometer", ep, cam)
    path.parent.mkdir(parents=True, exist_ok=True)
    times = list(np.linspace(0, ep.duration, len(progress)))
    path.write_text(json.dumps({"episode": ep.id, "camera": cam, "task": ep.task, "model": "robometer",
                                "version": "v", "times": times, "progress": progress, "success": progress}),
                    encoding="utf-8")


def test_run_swap_picks_threshold_on_dev_then_applies_it(tmp_path):
    root, specs = setup(tmp_path)
    cache = tmp_path / "cache"
    for split in ["dev", "test"]:
        for i, (orig, swapped, cam) in enumerate(swap_pairs(specs, root, split, seed=0)):
            write_curve(cache, orig, cam, [0.1, 0.5, 0.9 - 0.01 * i])
            write_curve(cache, swapped, cam, [0.1, 0.2, 0.2 + 0.01 * i])
    dev = run_swap(specs, root, "dev", cache)
    assert dev["score"] in SCORE_NAMES
    best = next(r for r in dev["scores"] if r["name"] == dev["score"])
    assert best["auc"] == 1.0
    assert dev["n_pairs"] == 2 * N_EP and dev["clean_flagged"] == 0 and dev["detected"] == 2 * N_EP
    test = run_swap(specs, root, "test", cache, score=dev["score"], threshold=dev["threshold"])
    assert test["n_pairs"] == 0  # test 只有一个数据集，没得换
    assert "分母" in format_swap(dev)


def test_run_swap_counts_missing_cache(tmp_path):
    root, specs = setup(tmp_path)
    out = run_swap(specs, root, "dev", tmp_path / "empty", score="end", threshold=0.5)
    assert out["n_pairs"] == 0 and out["missing"] == 2 * N_EP


def test_relative_score_compares_within_same_task(tmp_path):
    """两个任务的进度高低本来就不同：绝对阈值只抓得到一半，和同任务那批比就都抓得到。"""
    root, specs = setup(tmp_path)
    cache = tmp_path / "cache"
    for i, (orig, swapped, cam) in enumerate(swap_pairs(specs, root, "dev", seed=0)):
        if orig.task == "pick":  # 这批很齐，都在 0.95 附近
            write_curve(cache, orig, cam, [0.1, 0.5, 0.95 - 0.002 * (i % N_EP)])
            write_curve(cache, swapped, cam, [0.1, 0.3, 0.5])  # 配上 stack：比 stack 那批都低
        else:  # stack 这批散，0.60 到 0.85
            write_curve(cache, orig, cam, [0.1, 0.3, 0.60 + 0.05 * (i % N_EP)])
            write_curve(cache, swapped, cam, [0.1, 0.3, 0.9])  # 配上 pick：绝对值不低，但比 pick 那批低
    absolute = run_swap(specs, root, "dev", cache, score="peak")
    assert absolute["detected"] == N_EP  # 只抓到配上 stack 的那一半
    out = run_swap(specs, root, "dev", cache)
    assert out["score"].startswith("rel_")
    assert out["detected"] == 2 * N_EP and out["clean_flagged"] == 0


def test_relative_score_needs_enough_episodes_per_task(tmp_path):
    root, specs = setup(tmp_path)
    cache = tmp_path / "cache"
    pairs = swap_pairs(specs, root, "dev", seed=0)
    for orig, swapped, cam in pairs[: N_EP - 2] + pairs[N_EP:]:  # pick 那批只剩 4 条有缓存
        write_curve(cache, orig, cam, [0.1, 0.9])
        write_curve(cache, swapped, cam, [0.1, 0.5])
    names = {r["name"] for r in run_swap(specs, root, "dev", cache)["scores"]}
    assert "peak" in names and not any(n.startswith("rel_") for n in names)


def test_judge_then_eval_commands(tmp_path, capsys):
    from helpers import make_mp4
    from jianjin.cli import main

    root, specs = setup(tmp_path)
    for repo in ["org/a", "org/b"]:
        mp4 = local_dir(root, repo) / "videos" / CAM / "chunk-000" / "file-000.mp4"
        mp4.parent.mkdir(parents=True)
        make_mp4(mp4, n=N_EP * 300 + 30)
    cfg = tmp_path / "c.toml"
    cfg.write_text("".join(f'[[dataset]]\nrepo="{s.repo}"\nsplit="{s.split}"\nlicense="mit"\ncamera="{CAM}"\n'
                           for s in specs), encoding="utf-8")
    common = ["--config", str(cfg), "--data-root", str(root), "--cache", str(tmp_path / "cache")]
    assert main(["judge", "exam_a", "--judge", "fake", "--split", "dev", *common]) == 0
    assert f"{4 * N_EP}/{4 * N_EP}，完成 {4 * N_EP}" in capsys.readouterr().out
    dev_json = tmp_path / "dev.json"
    assert main(["eval", "a-swap", "--judge", "fake", "--split", "dev", *common, "--out", str(dev_json)]) == 0
    dev = json.loads(dev_json.read_text(encoding="utf-8"))
    assert dev["n_pairs"] == 2 * N_EP and dev["missing"] == 0
    with pytest.raises(SystemExit):
        main(["eval", "a-swap", "--split", "test", *common])


def test_choose_only_scores_the_rule_can_use():
    """「指令不符」规则只和同任务那批比：绝对分数检出再多也不选。"""
    from jianjin.eval.task_swap import _choose

    rows = [{"name": "success", "detected_at_max": 57, "auc": 0.884},
            {"name": "rel_success", "detected_at_max": 49, "auc": 0.940},
            {"name": "rel_end", "detected_at_max": 43, "auc": 0.703}]
    assert _choose(rows) == "rel_success"
    assert _choose([{"name": "end", "detected_at_max": 9, "auc": 0.9}]) == "rel_end"  # 没有能用的：规则查不了
