import os
from pathlib import Path

import numpy as np

from helpers import make_episode
from jianjin.diagnose import diagnose
from jianjin.judges.base import Curve
from jianjin.report import render
from jianjin.types import Cause, Finding

SNAP = Path(__file__).parent / "snapshots" / "report_small.html"


def small():
    eps = [make_episode(np.zeros(90), ep_id=f"ds/ep{i}", task="pick <the> cube") for i in range(3)]
    pause = Finding("long_pause", Cause.LONG_PAUSE, 1.0, 2.0, 1.0, "关节几乎不动，停了 1.0 秒")
    drop = Finding("progress_drop", Cause.DROP, 1.5, 2.5, 0.4, "进度从 80% 掉到 40%", source="robometer")
    curve = Curve("robometer", "v", [0.0, 1.0, 2.0, 3.0], [0.1, 0.8, 0.4, 0.9])
    vs = [
        diagnose("ds/ep0", 3.0, [pause, drop], {}, curve),
        diagnose("ds/ep1", 3.0, [pause], {"robometer.wrong_task": "同任务只有 3 条"}, curve),
        diagnose("ds/ep2", 3.0, [], {}, None),
    ]
    return vs, eps


def test_snapshot():
    vs, eps = small()
    html = render(vs, eps, title="ds", judges=["robometer"], generated="2026-10-03 12:00")
    if os.environ.get("JIANJIN_UPDATE_SNAPSHOTS") or not SNAP.exists():
        SNAP.parent.mkdir(exist_ok=True)
        SNAP.write_text(html, encoding="utf-8")
    assert html == SNAP.read_text(encoding="utf-8")


def test_counts_order_and_escaping():
    vs, eps = small()
    html = render(vs, eps, title="ds <1>", judges=["robometer"])
    assert "pick &lt;the&gt; cube" in html and "<the>" not in html and "ds &lt;1&gt;" in html
    assert html.count('<article class="card ') == 3
    assert html.index('data-label="作废"') < html.index('data-label="存疑"') < html.index('data-label="合格"')
    assert "人工只需要看存疑的 1 条" in html


def test_curve_with_marked_findings():
    vs, eps = small()
    html = render(vs, eps, title="ds", judges=["robometer"])
    assert '<polyline points="0.0,48.8' in html and 'class="mark reject"' in html


def test_banner_only_without_video():
    vs, eps = small()
    assert "没跑视频评判" in render(vs, eps, title="ds", judges=[])
    assert "没跑视频评判" not in render(vs, eps, title="ds", judges=["robometer"])


def test_operator_table_only_with_ids():
    vs, eps = small()
    assert "没有采集员编号" in render(vs, eps, title="ds")
    eps[0].operator_id = eps[1].operator_id = "王五"
    html = render(vs, eps, title="ds")
    assert "<td>王五</td><td>2</td><td>1</td><td>1</td>" in html and "（未填）" in html


def test_read_failures_listed():
    vs, eps = small()
    assert "读取失败 1 个文件" in render(vs, eps, title="ds", failures=[("a.hdf5", "坏了")])


def test_whole_episode_finding_says_whole():
    vs, eps = small()
    whole = Finding("wrong_task", Cause.WRONG_TASK, 0.0, 3.0, 9.0, "做的可能不是这个任务", source="robometer")
    vs[2] = diagnose("ds/ep2", 3.0, [whole], {}, None)
    html = render(vs, eps, title="ds")
    assert '<span class="when">整条</span>' in html and "第 0.0 到 3.0 秒" not in html


def test_short_episode_duration_keeps_two_decimals():
    ep = make_episode(np.zeros(1), ep_id="ds/tiny")
    v = diagnose("ds/tiny", ep.duration, [], {}, None)
    html = render([v], [ep], title="ds")
    assert '<span class="dur">0.03 秒</span>' in html


def test_curve_drawn_only_within_joint_data():
    vs, eps = small()
    vs[2].progress = Curve("robometer", "v", [0.0, 1.5, 3.0, 5.0], [0.1, 0.5, 0.9, 0.2])  # 第 5 秒在示范以外
    html = render([vs[2]], [eps[2]], title="ds")
    assert html.count('<polyline points="') == 1
    pts = html.split('<polyline points="')[1].split('"')[0].split()
    assert len(pts) == 3


def test_cards_show_keyframes_when_given():
    vs, eps = small()
    shots = {"ds/ep0": [(1.0, "data:image/jpeg;base64,AAAA"), (2.5, "data:image/jpeg;base64,BBBB")]}
    html = render(vs, eps, title="ds", shots=shots)
    assert html.count('<div class="shots">') == 1
    assert '<img src="data:image/jpeg;base64,BBBB" alt="第 2.5 秒的画面"' in html


def test_frame_after_joint_data_is_labelled():
    vs, eps = small()
    html = render(vs, eps, title="ds", shots={"ds/ep0": [(3.5, "data:image/jpeg;base64,AAAA")]})
    assert "第 3.5 秒（关节数据之后）" in html


def test_note_without_rejects():
    vs, eps = small()
    html = render(vs[1:], eps[1:], title="ds")
    assert "没有作废的" in html and "作废的 0 条" not in html
