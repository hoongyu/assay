import json

import numpy as np

from helpers import make_lerobot_v3
from assay.cli import main
from assay.ingest import load


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
    assert rows[0]["label"] == "存疑"
    printed = capsys.readouterr().out
    assert "1 条示范" in printed and "停顿过长" in printed


def moving6(n, seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.uniform(0.5, 1.5, n))
    return np.tile(x[:, None], (1, 6)) * np.arange(1, 7)


def test_check_writes_report(tmp_path, capsys):
    ds = make_lerobot_v3(tmp_path / "d", [("pick", paused6())])
    out = tmp_path / "r.html"
    assert main(["check", str(ds), "--out", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    assert "甄数质检报告" in html and "没跑视频评判" in html
    assert "存疑 1" in capsys.readouterr().out


def test_check_with_cached_video(tmp_path, capsys):
    from helpers import write_cached_curve

    ds = make_lerobot_v3(tmp_path / "d", [("pick", moving6(300, i)) for i in range(6)])
    cache = tmp_path / "cache"
    cam = "observation.images.front"
    for i, ep in enumerate(load(ds).episodes):
        write_cached_curve(cache, ep, cam, [0.1, 0.5, 0.3 if i == 0 else 0.9])
    js = tmp_path / "v.json"
    assert main(["check", str(ds), "--judges", "robometer", "--cache", str(cache), "--camera", cam,
                 "--json", str(js)]) == 0
    rows = json.loads(js.read_text(encoding="utf-8"))
    assert rows[0]["label"] == "存疑" and rows[0]["findings"][0]["source"] == "robometer"
    assert "Robometer：6/6 条有缓存" in capsys.readouterr().out


def test_load_detects_robovad(tmp_path):
    from helpers import make_robovad

    root = make_robovad(tmp_path / "rv", {"h_stackcubes_ep1": (np.zeros((60, 6)), np.zeros((60, 6)))})
    assert [e.id for e in load(root).episodes] == ["h_stackcubes_ep1"]



def test_check_creates_output_folders(tmp_path):
    ds = make_lerobot_v3(tmp_path / "d", [("pick", paused6())])
    js, out = tmp_path / "new" / "v.json", tmp_path / "new" / "r.html"
    assert main(["check", str(ds), "--json", str(js), "--out", str(out)]) == 0
    assert js.exists() and out.exists()


def test_check_report_has_keyframes_when_video_is_local(tmp_path, capsys):
    from helpers import make_mp4

    ds = make_lerobot_v3(tmp_path / "d", [("pick", paused6())])
    make_mp4(ds / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4", n=300)
    out = tmp_path / "r.html"
    assert main(["check", str(ds), "--out", str(out)]) == 0
    assert '<div class="shots">' in out.read_text(encoding="utf-8")
    assert "1 条存疑或作废的示范配了关键帧" in capsys.readouterr().out
    assert main(["check", str(ds), "--out", str(out), "--no-frames"]) == 0
    assert '<div class="shots">' not in out.read_text(encoding="utf-8")


def test_fetch_robomind_parses():
    from assay.cli import build_parser

    args = build_parser().parse_args(["fetch", "robomind", "--workers", "2"])
    assert args.what == "robomind" and args.workers == 2


def test_check_summary_without_key_uses_template(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    ds = make_lerobot_v3(tmp_path / "d", [("pick", paused6())])
    out = tmp_path / "r.html"
    assert main(["check", str(ds), "--out", str(out), "--no-frames", "--summary", "--cache", str(tmp_path / "c")]) == 0
    html = out.read_text(encoding="utf-8")
    assert "本批小结" in html and "1 条示范" in html and "模板" in html
    assert "NEBIUS_API_KEY" in capsys.readouterr().out


def test_check_summary_without_key_uses_cached_model_text(tmp_path, monkeypatch, capsys):
    """没设密钥，但这批的小结已经缓存过：用缓存里模型写的，不说「用模板」。"""
    from assay import summary as sm

    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    ds = make_lerobot_v3(tmp_path / "d", [("pick", paused6())])
    cache = tmp_path / "c"
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    monkeypatch.setattr(sm, "token_factory_chat", lambda key: (lambda messages, model: "本批 {n_total} 条。"))
    assert main(["check", str(ds), "--out", str(tmp_path / "a.html"), "--no-frames", "--summary", "--cache", str(cache)]) == 0
    monkeypatch.delenv("NEBIUS_API_KEY")
    capsys.readouterr()
    assert main(["check", str(ds), "--out", str(tmp_path / "b.html"), "--no-frames", "--summary", "--cache", str(cache)]) == 0
    printed = capsys.readouterr().out
    assert "NEBIUS_API_KEY" not in printed and "模型写的" in printed
    assert "本批 1 条。" in (tmp_path / "b.html").read_text(encoding="utf-8")


def test_check_in_english(tmp_path, monkeypatch, capsys):
    """--lang en：报告和终端打印都是英文（录给 Nebius 评委看）；JSON 里证据带英文。"""
    from helpers import HAN

    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    ds = make_lerobot_v3(tmp_path / "d", [("pick", paused6())])
    out, js = tmp_path / "r.html", tmp_path / "v.json"
    assert main(["check", str(ds), "--out", str(out), "--json", str(js), "--lang", "en", "--summary",
                 "--cache", str(tmp_path / "c")]) == 0
    html = out.read_text(encoding="utf-8")
    assert '<html lang="en">' in html and "Assay QA report" in html
    printed = capsys.readouterr().out
    assert "DOUBT 1" in printed and "long pause: 1" in printed and "Report written to" in printed
    assert "Batch summary (en): template" in printed and not HAN.search(printed), printed
    row = json.loads(js.read_text(encoding="utf-8"))[0]
    assert row["findings"][0]["evidence_en"].startswith("The joints barely moved")
    assert row["reasons_en"][0].startswith("Only joints reported")


def test_summary_stats_command(tmp_path, capsys):
    from test_summary import write_cache_entry

    write_cache_entry(tmp_path, "a", "zh", "model", 1)
    write_cache_entry(tmp_path, "b", "en", "template", 2)
    assert main(["summary-stats", "--cache", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "zh：第一次就过 1，重试以后过 0，退回模板 0（共 1 条）" in printed
    assert "en：第一次就过 0，重试以后过 0，退回模板 1（共 1 条）" in printed
