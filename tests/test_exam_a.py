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
    cfg = CheckConfig(still_rate=0.02, pause_s=3.0, end_moving_rate=0.05, end_window_s=0.5)
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
