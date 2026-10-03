"""离线单文件网页报告：整批的判定漏斗、问题分布、按采集员统计，加上每条示范一张体检单。

不引用任何外部资源（样式、脚本都在文件里），断网也能打开，数据不出场。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from html import escape

from jianjin.diagnose import Label, Verdict
from jianjin.types import Cause, Episode, Finding, source_name

ORDER = [Label.REJECT, Label.DOUBT, Label.PASS]  # 体检单的排列顺序
TILES = [Label.PASS, Label.DOUBT, Label.REJECT]  # 漏斗的排列顺序
CSS_CLASS = {Label.PASS: "pass", Label.DOUBT: "doubt", Label.REJECT: "reject"}
CAUSE_ORDER = list(Cause)

CSS = """
:root{--bg:#f6f5f1;--fg:#1d1d1b;--muted:#6b6a64;--card:#fff;--line:#e2e0d9;--curve:#2b5cab;
--pass:#2f7d4f;--doubt:#a86b12;--reject:#c0392b;--pass-bg:#e5f1e9;--doubt-bg:#fbefd8;--reject-bg:#f9e2de}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#9a9993;--card:#1f1f1d;--line:#34342f;
--curve:#7aa7ef;--pass:#5cbf86;--doubt:#e0a640;--reject:#ef6f61;--pass-bg:#1c3125;--doubt-bg:#3a2d15;--reject-bg:#3d1f1b}}
*{box-sizing:border-box}
body{margin:0 auto;max-width:1080px;padding:28px 16px 40px;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:18px;margin:36px 0 12px}
.sub,.note{color:var(--muted);margin:2px 0}
.banner{background:var(--doubt-bg);border:1px solid var(--doubt);border-radius:8px;padding:8px 12px;margin:14px 0 0}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:22px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;display:flex;flex-direction:column}
.tile b{font-size:34px;line-height:1.1;font-variant-numeric:tabular-nums}
.tile.pass b{color:var(--pass)}.tile.doubt b{color:var(--doubt)}.tile.reject b{color:var(--reject)}
.tile em{font-style:normal;color:var(--muted);font-size:13px}
.bar{display:flex;height:10px;border-radius:5px;overflow:hidden;margin:14px 0 8px;background:var(--line)}
.bar i.pass{background:var(--pass)}.bar i.doubt{background:var(--doubt)}.bar i.reject{background:var(--reject)}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{color:var(--muted);font-weight:500;font-size:13px}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.filters button{font:inherit;font-size:13px;padding:4px 12px;border-radius:999px;border:1px solid var(--line);
background:var(--card);color:var(--fg);cursor:pointer}
.filters button.on{border-color:var(--fg)}
.cards{display:flex;flex-direction:column;gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:10px;padding:12px 16px}
.card.pass{border-left-color:var(--pass)}.card.doubt{border-left-color:var(--doubt)}.card.reject{border-left-color:var(--reject)}
.head{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 12px}
.badge{font-size:12px;padding:1px 9px;border-radius:999px;white-space:nowrap}
.badge.pass{background:var(--pass-bg);color:var(--pass)}.badge.doubt{background:var(--doubt-bg);color:var(--doubt)}
.badge.reject{background:var(--reject-bg);color:var(--reject)}
code{font-family:ui-monospace,Consolas,monospace;font-size:13px;overflow-wrap:anywhere}
.task{color:var(--muted)}.dur{color:var(--muted);font-size:13px;margin-left:auto}
.reasons{margin:8px 0 0;padding-left:18px}
.findings{list-style:none;margin:8px 0 0;padding:0}
.findings li{padding:5px 0;border-top:1px dashed var(--line)}
.cause{font-weight:600;margin-right:8px}.when{color:var(--muted);font-size:13px;margin-right:8px}
.src{color:var(--muted);font-size:12px;margin-right:8px;padding:0 6px;border:1px solid var(--line);border-radius:4px}
.spark{margin:10px 0 0}
.spark svg{width:100%;max-width:440px;height:56px;display:block;background:var(--bg);border-radius:6px}
.spark polyline{fill:none;stroke:var(--curve);stroke-width:2;vector-effect:non-scaling-stroke}
.spark rect.mark{opacity:.28}.spark rect.mark.doubt{fill:var(--doubt)}.spark rect.mark.reject{fill:var(--reject)}
figcaption{color:var(--muted);font-size:12px}
details{margin-top:8px;color:var(--muted);font-size:13px}
details ul{margin:4px 0 0;padding-left:18px}
footer{margin-top:44px;color:var(--muted);font-size:13px;border-top:1px solid var(--line);padding-top:12px}
@media (max-width:600px){.tiles{grid-template-columns:1fr}.dur{margin-left:0}}
"""

JS = """
document.querySelectorAll(".filters button").forEach(function (b) {
  b.addEventListener("click", function () {
    document.querySelectorAll(".filters button").forEach(function (x) { x.classList.toggle("on", x === b); });
    document.querySelectorAll(".card").forEach(function (c) {
      c.hidden = b.dataset.f !== "all" && c.dataset.label !== b.dataset.f;
    });
  });
});
"""

FOOTER = (
    "<footer><p>判定规则：作废是客观硬规则触发（读数缺失、丢帧、示范过短），或者两个独立来源指向同一处问题；"
    "存疑是有发现但够不上作废，要人看；合格是什么都没报。错杀一条好数据比漏掉一条坏数据贵，所以判得偏保守。</p>"
    "<p>没查的项目不当成通过。阈值都在开发数据上定，出处见拣金设计文档。</p></footer>"
)


def _pct(k: int, n: int) -> str:
    return f"{k / n:.0%}" if n else "—"


def _header(title: str, n: int, judges: list[str], generated: str) -> str:
    checked = "关节体检" + (f"；视频评判：{'、'.join(source_name(j) for j in judges)}" if judges else "")
    when = f" · 生成于 {escape(generated)}" if generated else ""
    banner = "" if judges else (
        '<p class="banner">没跑视频评判，只有关节体检。「指令不符」「物体掉落」这几类查不了，记在每条的「没查」里。</p>')
    return (f'<header><h1>拣金质检报告</h1><p class="sub">{escape(title)} · {n} 条示范{when}</p>'
            f'<p class="sub">查了什么：{checked}</p>{banner}</header>')


def _funnel(verdicts: list[Verdict]) -> str:
    n = len(verdicts)
    c = Counter(v.label for v in verdicts)
    tiles = "".join(f'<div class="tile {CSS_CLASS[lb]}"><b>{c[lb]}</b><span>{lb.value}</span>'
                    f'<em>{_pct(c[lb], n)}</em></div>' for lb in TILES)
    bar = "".join(f'<i class="{CSS_CLASS[lb]}" style="width:{c[lb] / n * 100:.2f}%"></i>'
                  for lb in TILES if n and c[lb])
    note = (f"人工只需要看存疑的 {c[Label.DOUBT]} 条（{_pct(c[Label.DOUBT], n)}）；"
            f"作废的 {c[Label.REJECT]} 条每条都写了依据。")
    return f'<section class="funnel"><div class="tiles">{tiles}</div><div class="bar">{bar}</div><p class="note">{note}</p></section>'


def _cause_set(v: Verdict) -> list[Cause]:
    return sorted({f.cause for f in v.causes}, key=CAUSE_ORDER.index)


def _causes(verdicts: list[Verdict]) -> str:
    counts: dict[Cause, Counter] = defaultdict(Counter)
    for v in verdicts:
        for cause in _cause_set(v):
            counts[cause][v.label] += 1
    if not counts:
        return '<section><h2>问题出在哪</h2><p class="note">这一批没有报出任何问题。</p></section>'
    rows = "".join(f"<tr><td>{c.value}</td><td>{counts[c][Label.REJECT]}</td><td>{counts[c][Label.DOUBT]}</td></tr>"
                   for c in CAUSE_ORDER if c in counts)
    return ('<section><h2>问题出在哪</h2><table class="causes"><thead><tr><th>原因</th><th>作废</th><th>存疑</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
            '<p class="note">按示范计；一条示范报了几类原因，每类各记一次。</p></section>')


def _operators(verdicts: list[Verdict], episodes: list[Episode]) -> str:
    ops = {ep.id: ep.operator_id for ep in episodes}
    if not any(ops.values()):
        return ('<section><h2>按采集员</h2><p class="note">数据里没有采集员编号（公开数据都没有）。'
                '试点数据带上编号后，这里按人统计问题分布。</p></section>')
    table: dict[str, list] = defaultdict(lambda: [0, Counter(), Counter()])  # 示范数、判定、原因
    for v in verdicts:
        row = table[ops.get(v.episode_id) or "（未填）"]
        row[0] += 1
        row[1][v.label] += 1
        row[2].update(_cause_set(v))
    rows = "".join(
        f"<tr><td>{escape(op)}</td><td>{r[0]}</td><td>{r[1][Label.DOUBT]}</td><td>{r[1][Label.REJECT]}</td>"
        f"<td>{r[2].most_common(1)[0][0].value if r[2] else '—'}</td></tr>"
        for op, r in sorted(table.items()))
    return ('<section><h2>按采集员</h2><table class="ops"><thead><tr><th>采集员</th><th>示范</th><th>存疑</th>'
            f'<th>作废</th><th>最常见的问题</th></tr></thead><tbody>{rows}</tbody></table></section>')


def _failures(failures: list[tuple[str, str]]) -> str:
    if not failures:
        return ""
    items = "".join(f"<li><code>{escape(name)}</code>：{escape(why)}</li>" for name, why in failures)
    return f'<section><h2>读取失败 {len(failures)} 个文件</h2><ul class="failures">{items}</ul></section>'


def _spark(v: Verdict, duration: float) -> str:
    c = v.progress
    if c is None or not c.times or duration <= 0:
        return ""
    w, h = 320, 56

    def x(t: float) -> float:
        return min(max(t / duration, 0.0), 1.0) * w

    def y(p: float) -> float:
        return (1.0 - min(max(p, 0.0), 1.0)) * (h - 4) + 2

    marks = "".join(
        f'<rect class="mark {CSS_CLASS[v.label]}" x="{x(f.t0):.1f}" y="0" width="{max(x(f.t1) - x(f.t0), 2.0):.1f}" height="{h}"/>'
        for f in v.causes)
    pts = " ".join(f"{x(t):.1f},{y(p):.1f}" for t, p in zip(c.times, c.progress))
    name = f"{source_name(c.model)} 的进度曲线"
    return (f'<figure class="spark"><svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" '
            f'aria-label="{escape(name)}">{marks}<polyline points="{pts}"/></svg>'
            f'<figcaption>{escape(name)}，横轴 0 到 {duration:.1f} 秒，色块是报出问题的时间段</figcaption></figure>')


def _finding(f: Finding, duration: float) -> str:
    whole = f.t0 <= 0.0 and f.t1 >= duration - 1e-9
    when = "整条" if whole else f"第 {f.t0:.1f} 到 {f.t1:.1f} 秒"
    return (f'<li><span class="cause">{f.cause.value}</span><span class="when">{when}</span>'
            f'<span class="src">{escape(source_name(f.source))}</span>{escape(f.evidence)}</li>')


def _card(v: Verdict, ep: Episode) -> str:
    cls = CSS_CLASS[v.label]
    parts = [f'<article class="card {cls}" data-label="{v.label.value}">',
             f'<div class="head"><span class="badge {cls}">{v.label.value}</span><code>{escape(v.episode_id)}</code>'
             f'<span class="task">{escape(ep.task)}</span><span class="dur">{ep.duration:.1f} 秒</span></div>']
    if v.reasons:
        parts.append('<ul class="reasons">' + "".join(f"<li>{escape(r)}</li>" for r in v.reasons) + "</ul>")
    parts.append(_spark(v, ep.duration))
    if v.causes:
        parts.append('<ul class="findings">' + "".join(_finding(f, ep.duration) for f in v.causes) + "</ul>")
    if v.not_checked:
        items = "".join(f"<li>{escape(k)}：{escape(why)}</li>" for k, why in sorted(v.not_checked.items()))
        parts.append(f"<details><summary>没查 {len(v.not_checked)} 项</summary><ul>{items}</ul></details>")
    parts.append("</article>")
    return "".join(parts)


def _cards(verdicts: list[Verdict], episodes: list[Episode]) -> str:
    by_id = {ep.id: ep for ep in episodes}
    c = Counter(v.label for v in verdicts)
    buttons = [f'<button type="button" data-f="all" class="on">全部 {len(verdicts)}</button>'] + [
        f'<button type="button" data-f="{lb.value}">{lb.value} {c[lb]}</button>' for lb in ORDER]
    ordered = sorted(verdicts, key=lambda v: ORDER.index(v.label))  # 稳定排序：同一类里保持原来的顺序
    cards = "".join(_card(v, by_id[v.episode_id]) for v in ordered)
    return (f'<section><h2>每条示范</h2><nav class="filters">{"".join(buttons)}</nav>'
            f'<div class="cards">{cards}</div></section>')


def render(verdicts: list[Verdict], episodes: list[Episode], *, title: str, judges=(), failures=(),
           generated: str = "") -> str:
    body = "".join([
        _header(title, len(verdicts), list(judges), generated),
        _funnel(verdicts),
        _causes(verdicts),
        _operators(verdicts, episodes),
        _failures(list(failures)),
        _cards(verdicts, episodes),
        FOOTER,
    ])
    return ('<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>拣金质检报告 · {escape(title)}</title><style>{CSS}</style></head>'
            f'<body>{body}<script>{JS}</script></body></html>\n')
