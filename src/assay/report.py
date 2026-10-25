"""离线单文件网页报告：整批的判定漏斗、问题分布、按采集员统计，加上每条示范一张体检单。

不引用任何外部资源（样式、脚本都在文件里），断网也能打开，数据不出场。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from html import escape

from assay.diagnose import LABEL_EN, Label, Verdict
from assay.i18n import check_name
from assay.types import CAUSE_EN, Cause, Episode, Finding, en, source_name, spaced

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
.review{margin-top:8px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;display:flex;flex-direction:column;gap:4px}
.review .a{font-weight:600}
.cause{font-weight:600;margin-right:8px}.when{color:var(--muted);font-size:13px;margin-right:8px}
.src{color:var(--muted);font-size:12px;margin-right:8px;padding:0 6px;border:1px solid var(--line);border-radius:4px}
.spark{margin:10px 0 0}
.spark svg{width:100%;max-width:440px;height:56px;display:block;background:var(--bg);border-radius:6px}
.spark polyline{fill:none;stroke:var(--curve);stroke-width:2;vector-effect:non-scaling-stroke}
.spark rect.mark{opacity:.28}.spark rect.mark.doubt{fill:var(--doubt)}.spark rect.mark.reject{fill:var(--reject)}
.shots{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(140px,45%),1fr));gap:6px;margin:10px 0 0;max-width:760px}
.shots figure{margin:0}.shots img{width:100%;height:auto;display:block;border-radius:6px;background:var(--line)}
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

UI = {  # 界面文字：键 -> (中文, 英文)
    "title": ("甄数质检报告", "Assay QA report"),
    "episodes": ("{n} 条示范", "{n} episodes"),
    "generated": ("生成于 {when}", "generated {when}"),
    "checked": ("查了什么：", "Checks: "),
    "joints": ("关节体检", "joint rules"),
    "judges": ("；视频评判：", "; video judges: "),
    "list_sep": ("、", ", "),
    "banner": ("没跑视频评判，只有关节体检。「指令不符」「物体掉落」这几类查不了，记在每条的「没查」里。",
               "The video judges did not run, only the joint rules. Wrong task and object dropped could not be "
               "checked; every card lists them under not checked."),
    "note": ("人工只需要看存疑的 {d} 条（{pct}）；", "A person only needs to review the {d} DOUBT episodes ({pct}); "),
    "note_reject": ("作废的 {k} 条每条都写了依据。", "each of the {k} REJECT episodes comes with its evidence."),
    "note_none": ("没有作废的。", "nothing was rejected."),
    "problems": ("问题出在哪", "Problems found"),
    "no_problems": ("这一批没有报出任何问题。", "No problems were reported in this batch."),
    "cause": ("原因", "Problem"),
    "per_episode": ("按示范计；一条示范报了几类原因，每类各记一次。",
                    "Counted per episode; an episode with several kinds of problem counts once for each."),
    "by_operator": ("按采集员", "By operator"),
    "no_operators": ("数据里没有采集员编号（公开数据都没有）。试点数据带上编号后，这里按人统计问题分布。",
                     "The data carries no operator ids (no public dataset does). With operator ids, this section "
                     "breaks the problems down by operator."),
    "operator": ("采集员", "Operator"),
    "n_episodes": ("示范", "Episodes"),
    "top_problem": ("最常见的问题", "Most frequent problem"),
    "unknown_op": ("（未填）", "(unknown)"),
    "failures": ("读取失败 {n} 个文件", "{n} files could not be read"),
    "curve": ("{model} 的进度曲线", "{model} progress curve"),
    "curve_caption": ("{name}，横轴 0 到 {d} 秒，色块是报出问题的时间段",
                      "{name}, 0 to {d} s; shaded spans mark the reported problems"),
    "at": ("第 {t} 秒", "at {t} s"),
    "after": ("（关节数据之后）", " (after the joint data)"),
    "frame": ("{when}的画面", "frame {when}"),
    "whole": ("整条", "whole episode"),
    "span": ("第 {t0} 到 {t1} 秒", "{t0} to {t1} s"),
    "seconds": ("{x} 秒", "{x} s"),
    "not_checked": ("没查 {n} 项", "Not checked: {n}"),
    "sep": ("：", ": "),
    "episodes_h": ("每条示范", "Episodes"),
    "all": ("全部 {n}", "All {n}"),
    "summary": ("本批小结", "Batch summary"),
    "review": ("视频复核（Nemotron 3.5 Super VL，经 Nebius Token Factory）", "Video review (Nemotron 3.5 Super VL on Nebius Token Factory)"),
    "review_q": ("问（英文原文）", "Question"),
    "review_a": ("答", "Answer"),
    "review_confirms": ("支持规则的发现", "agrees with the rule"),
    "review_clears": ("和规则的发现相反，要人看", "disagrees with the rule; a person should look"),
    "review_unclear": ("没给出能用的回答", "no usable answer"),
    "footer": (
        "<p>判定规则：作废是客观硬规则触发（读数缺失、丢帧、示范过短、视频比关节数据短），或者两个独立来源指向同一处问题；"
        "存疑是有发现但够不上作废，要人看；合格是什么都没报。判得偏保守是取舍：好示范很少被错杀，代价是有些坏示范会判合格。</p>"
        "<p>没查的项目不当成通过。阈值都在开发数据上定，出处见甄数设计文档。</p>",
        "<p>Verdicts: REJECT means an objective hard rule fired (missing readings, dropped frames, too short, video "
        "shorter than the joint data) or two independent sources point at the same problem; DOUBT means something "
        "was found but not enough to reject, so a person should look; PASS means nothing was found. Verdicts are "
        "conservative by choice: good episodes are rarely thrown away, at the price of letting some bad ones "
        "pass.</p><p>Checks that could not run are not counted as passed. Thresholds were chosen on development "
        "data; see the Assay design docs.</p>",
    ),
}


def _t(key: str, lang: str, **kw) -> str:
    zh, english = UI[key]
    text = english if lang == "en" else zh
    return text.format(**kw) if kw else text


def _label(lb: Label, lang: str) -> str:
    return LABEL_EN[lb] if lang == "en" else lb.value


def _cause(c: Cause, lang: str) -> str:
    return CAUSE_EN[c] if lang == "en" else c.value


def _text(text: str, lang: str) -> str:
    """证据、理由、没查的原因：英文报告取 types.Bi 的英文。"""
    return en(text) if lang == "en" else text


def _rule(key: str, lang: str) -> str:
    return check_name(key, "en") if lang == "en" else spaced(check_name(key))


def _pct(k: int, n: int) -> str:
    return f"{k / n:.0%}" if n else "—"


def _dur(seconds: float, lang: str) -> str:
    x = f"{seconds:.1f}" if seconds >= 1 else f"{seconds:.2f}"  # 过短的示范别显示成 0.0 秒
    return _t("seconds", lang, x=x)


def _header(title: str, n: int, judges: list[str], generated: str, lang: str) -> str:
    names = _t("list_sep", lang).join(source_name(j, lang) for j in judges)
    checked = _t("joints", lang) + (_t("judges", lang) + names if judges else "")
    when = f" · {escape(_t('generated', lang, when=generated))}" if generated else ""
    banner = "" if judges else f'<p class="banner">{_t("banner", lang)}</p>'
    return (f'<header><h1>{_t("title", lang)}</h1><p class="sub">{escape(title)} · {_t("episodes", lang, n=n)}'
            f'{when}</p><p class="sub">{_t("checked", lang)}{checked}</p>{banner}</header>')


def _funnel(verdicts: list[Verdict], lang: str) -> str:
    n = len(verdicts)
    c = Counter(v.label for v in verdicts)
    tiles = "".join(f'<div class="tile {CSS_CLASS[lb]}"><b>{c[lb]}</b><span>{_label(lb, lang)}</span>'
                    f'<em>{_pct(c[lb], n)}</em></div>' for lb in TILES)
    bar = "".join(f'<i class="{CSS_CLASS[lb]}" style="width:{c[lb] / n * 100:.2f}%"></i>'
                  for lb in TILES if n and c[lb])
    k = c[Label.REJECT]
    note = (_t("note", lang, d=c[Label.DOUBT], pct=_pct(c[Label.DOUBT], n))
            + (_t("note_reject", lang, k=k) if k else _t("note_none", lang)))
    return f'<section class="funnel"><div class="tiles">{tiles}</div><div class="bar">{bar}</div><p class="note">{note}</p></section>'


def _cause_set(v: Verdict) -> list[Cause]:
    return sorted({f.cause for f in v.causes}, key=CAUSE_ORDER.index)


def _causes(verdicts: list[Verdict], lang: str) -> str:
    counts: dict[Cause, Counter] = defaultdict(Counter)
    for v in verdicts:
        for cause in _cause_set(v):
            counts[cause][v.label] += 1
    if not counts:
        return f'<section><h2>{_t("problems", lang)}</h2><p class="note">{_t("no_problems", lang)}</p></section>'
    rows = "".join(f"<tr><td>{_cause(c, lang)}</td><td>{counts[c][Label.REJECT]}</td><td>{counts[c][Label.DOUBT]}</td></tr>"
                   for c in CAUSE_ORDER if c in counts)
    return (f'<section><h2>{_t("problems", lang)}</h2><table class="causes"><thead><tr><th>{_t("cause", lang)}</th>'
            f'<th>{_label(Label.REJECT, lang)}</th><th>{_label(Label.DOUBT, lang)}</th>'
            f'</tr></thead><tbody>{rows}</tbody></table><p class="note">{_t("per_episode", lang)}</p></section>')


def _operators(verdicts: list[Verdict], episodes: list[Episode], lang: str) -> str:
    ops = {ep.id: ep.operator_id for ep in episodes}
    if not any(ops.values()):
        return f'<section><h2>{_t("by_operator", lang)}</h2><p class="note">{_t("no_operators", lang)}</p></section>'
    table: dict[str, list] = defaultdict(lambda: [0, Counter(), Counter()])  # 示范数、判定、原因
    for v in verdicts:
        row = table[ops.get(v.episode_id) or _t("unknown_op", lang)]
        row[0] += 1
        row[1][v.label] += 1
        row[2].update(_cause_set(v))
    rows = "".join(
        f"<tr><td>{escape(op)}</td><td>{r[0]}</td><td>{r[1][Label.DOUBT]}</td><td>{r[1][Label.REJECT]}</td>"
        f"<td>{_cause(r[2].most_common(1)[0][0], lang) if r[2] else '—'}</td></tr>"
        for op, r in sorted(table.items()))
    return (f'<section><h2>{_t("by_operator", lang)}</h2><table class="ops"><thead><tr><th>{_t("operator", lang)}</th>'
            f'<th>{_t("n_episodes", lang)}</th><th>{_label(Label.DOUBT, lang)}</th><th>{_label(Label.REJECT, lang)}</th>'
            f'<th>{_t("top_problem", lang)}</th></tr></thead><tbody>{rows}</tbody></table></section>')


def _failures(failures: list[tuple[str, str]], lang: str) -> str:
    if not failures:
        return ""
    items = "".join(f"<li><code>{escape(name)}</code>{_t('sep', lang)}{escape(_text(why, lang))}</li>"
                    for name, why in failures)
    return f'<section><h2>{_t("failures", lang, n=len(failures))}</h2><ul class="failures">{items}</ul></section>'


def _spark(v: Verdict, duration: float, lang: str) -> str:
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
    pts = " ".join(f"{x(t):.1f},{y(p):.1f}" for t, p in zip(c.times, c.progress) if t <= duration + 1e-6)
    name = _t("curve", lang, model=source_name(c.model, lang))
    caption = _t("curve_caption", lang, name=name, d=f"{duration:.1f}")
    return (f'<figure class="spark"><svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" '
            f'aria-label="{escape(name)}">{marks}<polyline points="{pts}"/></svg>'
            f'<figcaption>{escape(caption)}</figcaption></figure>')


def _shots(shots: list[tuple[float, str]], duration: float, lang: str) -> str:
    if not shots:
        return ""

    def when(t: float) -> str:
        return _t("at", lang, t=f"{t:.1f}") + (_t("after", lang) if t > duration + 1e-6 else "")

    figs = "".join(f'<figure><img src="{uri}" alt="{_t("frame", lang, when=when(t))}" loading="lazy">'
                   f'<figcaption>{when(t)}</figcaption></figure>' for t, uri in shots)
    return f'<div class="shots">{figs}</div>'


def _finding(f: Finding, duration: float, lang: str) -> str:
    whole = f.t0 <= 0.0 and f.t1 >= duration - 1e-9
    when = _t("whole", lang) if whole else _t("span", lang, t0=f"{f.t0:.1f}", t1=f"{f.t1:.1f}")
    return (f'<li><span class="cause">{_cause(f.cause, lang)}</span><span class="when">{when}</span>'
            f'<span class="src">{escape(source_name(f.source, lang))}</span>{escape(_text(f.evidence, lang))}</li>')


ANSWER = {"yes": ("是", "yes"), "no": ("否", "no"), "unclear": ("看不清", "unclear"),
          "still_moving": ("还在动", "still moving"), "at_rest": ("已经停下", "at rest"),
          "robot_continues_task": ("机器人还在做任务", "the robot continues the task"),
          "person_in_scene": ("有人在画面里", "a person is in the scene"), "nothing_moves": ("画面里没有动作", "nothing moves")}


def _review(r, lang: str) -> str:
    """视觉模型的复核：问了什么、答了什么、和规则的发现是否一致。"""
    zh, english = ANSWER.get(r.answer, (r.answer or "—", r.answer or "—"))
    answer = english if lang == "en" else zh
    return (f'<div class="review"><span class="src">{_t("review", lang)}</span>'
            f'<span class="q">{_t("review_q", lang)}{_t("sep", lang)}{escape(r.question)}</span>'
            f'<span class="a">{_t("review_a", lang)}{_t("sep", lang)}{escape(answer)}{_t("sep", lang)}'
            f'{_t("review_" + r.outcome, lang)}</span></div>')


def _card(v: Verdict, ep: Episode, shots=(), lang: str = "zh") -> str:
    cls = CSS_CLASS[v.label]
    parts = [f'<article class="card {cls}" data-label="{cls}">',
             f'<div class="head"><span class="badge {cls}">{_label(v.label, lang)}</span><code>{escape(v.episode_id)}</code>'
             f'<span class="task">{escape(ep.task)}</span><span class="dur">{_dur(ep.duration, lang)}</span></div>']
    if v.reasons:
        parts.append('<ul class="reasons">' + "".join(f"<li>{escape(_text(r, lang))}</li>" for r in v.reasons) + "</ul>")
    parts.append(_spark(v, ep.duration, lang))
    parts.append(_shots(list(shots), ep.duration, lang))
    if v.causes:
        parts.append('<ul class="findings">' + "".join(_finding(f, ep.duration, lang) for f in v.causes) + "</ul>")
    parts += [_review(r, lang) for r in v.reviews]
    if v.not_checked:
        items = "".join(f"<li>{escape(_rule(k, lang))}{_t('sep', lang)}{escape(_text(why, lang))}</li>"
                        for k, why in sorted(v.not_checked.items()))
        parts.append(f"<details><summary>{_t('not_checked', lang, n=len(v.not_checked))}</summary><ul>{items}</ul></details>")
    parts.append("</article>")
    return "".join(parts)


def _cards(verdicts: list[Verdict], episodes: list[Episode], shots=None, lang: str = "zh") -> str:
    by_id = {ep.id: ep for ep in episodes}
    c = Counter(v.label for v in verdicts)
    buttons = [f'<button type="button" data-f="all" class="on">{_t("all", lang, n=len(verdicts))}</button>'] + [
        f'<button type="button" data-f="{CSS_CLASS[lb]}">{_label(lb, lang)} {c[lb]}</button>' for lb in ORDER]
    ordered = sorted(verdicts, key=lambda v: ORDER.index(v.label))  # 稳定排序：同一类里保持原来的顺序
    cards = "".join(_card(v, by_id[v.episode_id], (shots or {}).get(v.episode_id, ()), lang) for v in ordered)
    return (f'<section><h2>{_t("episodes_h", lang)}</h2><nav class="filters">{"".join(buttons)}</nav>'
            f'<div class="cards">{cards}</div></section>')


def _summary(s, lang: str) -> str:
    """本批小结（assay.summary.Summary）：中英各一段，注明是模型写的还是模板。英文报告英文在前。"""
    if s is None:
        return ""
    notes = {
        "zh": (f"Nemotron 写的（{s.model}，Nebius Token Factory），数字由甄数按统计填",
               "模板文字" + ("：模型的输出没过数字核对，或者调用失败" if s.problems.get("zh") else "")),
        "en": (f"Written by Nemotron ({s.model}, Nebius Token Factory); every number was filled in by Assay "
               "from its own statistics", "Template text" + (": the model's draft failed the number check, or the call "
                                                             "failed" if s.problems.get("en") else "")),
    }
    order = (("en", "en"), ("zh", "zh-CN")) if lang == "en" else (("zh", "zh-CN"), ("en", "en"))
    paras = []
    for code, html_lang in order:
        if code in s.texts:
            note = notes[code][0] if s.source[code] == "model" else notes[code][1]
            paras.append(f'<p lang="{html_lang}">{escape(s.texts[code])}</p><p class="note">{escape(note)}</p>')
    return f'<section class="summary"><h2>{_t("summary", lang)}</h2>{"".join(paras)}</section>'


def render(verdicts: list[Verdict], episodes: list[Episode], *, title: str, judges=(), failures=(),
           generated: str = "", shots=None, summary=None, lang: str = "zh") -> str:
    """shots：{示范 id: [(秒, 图片 data URI)]}，给存疑、作废的体检单配画面；不给就不配。
    summary：本批小结（assay.summary.Summary）；不给就没有这一节。lang：zh 或 en，界面用哪种文字。"""
    body = "".join([
        _header(title, len(verdicts), list(judges), generated, lang),
        _summary(summary, lang),
        _funnel(verdicts, lang),
        _causes(verdicts, lang),
        _operators(verdicts, episodes, lang),
        _failures(list(failures), lang),
        _cards(verdicts, episodes, shots, lang),
        f"<footer>{_t('footer', lang)}</footer>",
    ])
    html_lang = "en" if lang == "en" else "zh-CN"
    return (f'<!doctype html>\n<html lang="{html_lang}"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{_t("title", lang)} · {escape(title)}</title><style>{CSS}</style></head>'
            f'<body>{body}<script>{JS}</script></body></html>\n')
