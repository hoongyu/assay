import json

import numpy as np

from helpers import make_lerobot_v3
from jianjin.cli import main
from jianjin.ingest import load


def paused6():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(150), np.linspace(1, 2, 60), np.full(30, 2.0)])
    return np.tile(x[:, None], (1, 6))


def test_load_detects_lerobot(tmp_path):
    ds = make_lerobot_v3(tmp_path / "d", [("pick", paused6())])
    assert len(load(ds).episodes) == 1


def test_load_unknown_format(tmp_path):
    (tmp_path / "x.txt").write_text("hi")
    try:
        load(tmp_path)
    except ValueError as exc:
        assert "认不出" in str(exc)
    else:
        raise AssertionError("应该报错")


def test_check_command_writes_json(tmp_path, capsys):
    ds = make_lerobot_v3(tmp_path / "d", [("pick", paused6())])
    out = tmp_path / "findings.json"
    assert main(["check", str(ds), "--json", str(out)]) == 0
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows[0]["episode_id"] == "d/ep000000"
    assert rows[0]["findings"][0]["cause"] == "停顿过长"
    printed = capsys.readouterr().out
    assert "1 条示范" in printed and "停顿过长" in printed
