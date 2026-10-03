import json

import numpy as np

from helpers import make_lerobot_v3
from jianjin.checks import CheckConfig
from jianjin.cli import main
from jianjin.eval.exam_a import DatasetSpec, format_table, local_dir, run_exam_a


def moving6(n, seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.uniform(0.5, 1.5, n))  # 一直在动、速度有起伏
    return np.tile(x[:, None], (1, 6)) * np.arange(1, 7)


def setup(tmp_path):
    root = tmp_path / "data"
    for repo, task in [("org/dev_ds", "pick"), ("org/test_ds", "place")]:
        make_lerobot_v3(local_dir(root, repo), [(task, moving6(300, i)) for i in range(6)])
    specs = [
        DatasetSpec("org/dev_ds", "dev", "apache-2.0"),
        DatasetSpec("org/test_ds", "test", "mit"),
    ]
    return root, specs


def test_run_exam_a_dev_only_uses_dev(tmp_path):
    root, specs = setup(tmp_path)
    # 阈值写死在测试里，Task 13 重定默认阈值时这条测试不受影响
    cfg = CheckConfig(still_rate=0.02, pause_s=3.0, end_pose_max=3.0, end_pose_floor=0.02)
    out = run_exam_a(specs, root, "dev", cfg=cfg, seed=0)
    assert [s["repo"] for s in out["datasets"]] == ["org/dev_ds"]
    assert out["n_episodes"] == 6
    kinds = {k["kind"]: k for k in out["kinds"]}
    assert kinds["insert_pause"]["n_injected"] == 6
    assert kinds["insert_pause"]["localized"] == 6
    assert kinds["truncate_end"]["detected"] == 6
    table = format_table(out)
    assert "insert_pause" in table and "/6" in table and "org/dev_ds" in table


def test_run_exam_a_is_reproducible(tmp_path):
    root, specs = setup(tmp_path)
    a = run_exam_a(specs, root, "dev", seed=3)
    b = run_exam_a(specs, root, "dev", seed=3)
    assert a["kinds"] == b["kinds"]


def test_eval_command(tmp_path, capsys):
    root, _ = setup(tmp_path)
    cfg = tmp_path / "exam_a.toml"
    cfg.write_text(
        '[[dataset]]\nrepo = "org/dev_ds"\nsplit = "dev"\nlicense = "apache-2.0"\n',
        encoding="utf-8",
    )
    out = tmp_path / "res.json"
    assert main(["eval", "a", "--config", str(cfg), "--data-root", str(root), "--split", "dev", "--out", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["split"] == "dev" and saved["seed"] == 0
    assert "注入" in capsys.readouterr().out


def test_excluded_tasks_left_out(tmp_path):
    root = tmp_path / "data"
    make_lerobot_v3(local_dir(root, "org/mix"), [("pick", moving6(300, i)) for i in range(6)]
                    + [("fold", moving6(300, 10 + i)) for i in range(6)])
    spec = DatasetSpec("org/mix", "dev", "apache-2.0", exclude_tasks=("fold",))
    out = run_exam_a([spec], root, "dev", seed=0)
    assert out["n_episodes"] == 6 and out["datasets"][0]["tasks"] == ["pick"]
    assert out["datasets"][0]["excluded_tasks"] == ["fold"]


def test_config_reads_excluded_tasks(tmp_path):
    from jianjin.eval.exam_a import load_config

    cfg = tmp_path / "a.toml"
    cfg.write_text('[[dataset]]\nrepo = "o/m"\nsplit = "dev"\nlicense = "mit"\nexclude_tasks = ["fold"]\n',
                   encoding="utf-8")
    (spec,) = load_config(cfg)
    assert spec.exclude_tasks == ("fold",) and spec.camera is None


def test_fetch_downloads_only_the_named_camera():
    from jianjin.eval.fetch import patterns_for

    top = DatasetSpec("o/a", "dev", "mit", "observation.images.top")
    assert patterns_for(top, True) == ["meta/*", "data/*", "videos/observation.images.top/*"]
    assert patterns_for(top, False) == ["meta/*", "data/*"]
    assert patterns_for(DatasetSpec("o/b", "dev", "mit"), True) == ["meta/*", "data/*"]  # 没写相机：不下视频


def test_scores_broken_down_by_dataset(tmp_path):
    root = tmp_path / "data"
    for repo, task in [("org/a", "pick"), ("org/b", "place")]:
        make_lerobot_v3(local_dir(root, repo), [(task, moving6(300, i)) for i in range(6)])
    specs = [DatasetSpec("org/a", "dev", "mit"), DatasetSpec("org/b", "dev", "mit")]
    out = run_exam_a(specs, root, "dev", seed=0)
    assert set(out["by_dataset"]) == {"org/a", "org/b"}
    for i, k in enumerate(out["kinds"]):
        parts = [out["by_dataset"][r][i] for r in ("org/a", "org/b")]
        assert all(p["kind"] == k["kind"] and p["n_injected"] == 6 and p["n_clean"] == 6 for p in parts)
        for key in ("detected", "localized", "clean_flagged"):
            assert sum(p[key] for p in parts) == k[key]
