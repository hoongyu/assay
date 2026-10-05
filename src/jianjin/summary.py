"""本批小结：Nemotron（Nebius Token Factory）写给质检组长和采集员的一段话，数字由拣金填。

模型只看拣金算出的统计（条数、原因、示范编号），不看原始数据。它写的文字里，所有数量、编号、原因名都是占位符，
由代码按统计填进去；它自己写的数字一律不收（见 check_text），不合格重试一次，还不行退回模板。
见 docs/superpowers/specs/2026-10-05-nemotron-summary-design.md。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from jianjin.diagnose import Label, Verdict
from jianjin.types import Cause, Episode, source_name

CAUSE_EN = {
    Cause.INCOMPLETE: "incomplete", Cause.EMPTY: "empty recording", Cause.MISPLACED: "mispositioned",
    Cause.GRASP_MISS: "missed grasp", Cause.REGRASP: "regrasp", Cause.DROP: "object dropped",
    Cause.EXTRA_MOTION: "extra motion", Cause.LONG_PAUSE: "long pause", Cause.JERKY: "unsmooth motion",
    Cause.WRONG_TASK: "wrong task", Cause.DATA_ERROR: "data error",
}
CHECKS = {  # 规则名：（中文，英文）
    "missing_values": ("读数缺失", "missing readings"), "too_short": ("示范过短", "too short"),
    "timestamp_gaps": ("时间戳", "timestamp gaps"), "video_length": ("视频时长", "video length"),
    "long_pause": ("停顿过长", "long pause"), "extra_motion": ("多余动作", "extra motion"),
    "end_pose": ("没做完", "incomplete"), "empty_recording": ("空录", "empty recording"),
    "regrasp": ("二次抓取", "regrasp"), "jerky": ("动作不流畅", "unsmooth motion"),
    "wrong_task": ("指令不符", "wrong task"), "progress_drop": ("进度回落", "progress drop"),
}
SOURCE_EN = {"joints": "joint", "meta": "metadata", "robometer": "Robometer", "topreward": "TOPReward"}
TOP_CAUSES = 5
EXAMPLES = 3
TOP_OPERATORS = 5


@dataclass(frozen=True)
class Facts:
    """给模型看的事实。meaning[占位符名] 告诉模型它是什么；values[语言][占位符名] 是填进去的文字。"""

    meaning: dict[str, str]
    values: dict[str, dict[str, str]]


def _check_name(key: str) -> tuple[str, str]:
    src, _, name = key.rpartition(".")
    zh, en = CHECKS.get(name, (name, name))
    if src:
        return f"{source_name(src)}的「{zh}」", f"{SOURCE_EN.get(src, src)} {en}"
    return zh, en


def _short(episode_id: str) -> str:
    return episode_id.rsplit("/", 1)[-1]


def batch_facts(verdicts: list[Verdict], episodes: list[Episode], title: str) -> Facts:
    """一批示范的统计：各判定条数，报得最多的原因（按示范计），没查的项目（没启用的不算），有采集员编号时按人分。"""
    meaning: dict[str, str] = {}
    zh: dict[str, str] = {}
    en: dict[str, str] = {}

    def put(key: str, what: str, z, e=None) -> None:
        meaning[key], zh[key], en[key] = what, str(z), str(z if e is None else e)

    order = list(Cause)
    n = len(verdicts)
    labels = Counter(v.label for v in verdicts)
    put("batch", "name of this batch of demonstrations", title)
    put("n_total", "number of episodes checked", n)
    put("n_pass", "episodes judged PASS", labels[Label.PASS])
    put("n_doubt", "episodes judged DOUBT (a human reviewer should look at them)", labels[Label.DOUBT])
    put("n_reject", "episodes judged REJECT", labels[Label.REJECT])

    by_cause: dict[Cause, list[str]] = defaultdict(list)
    for v in verdicts:
        for c in dict.fromkeys(f.cause for f in v.causes):
            by_cause[c].append(v.episode_id)
    ranked = sorted(by_cause.items(), key=lambda kv: (-len(kv[1]), order.index(kv[0])))[:TOP_CAUSES]
    for i, (cause, ids) in enumerate(ranked, 1):
        put(f"cause_{i}", f"name of the problem ranked #{i} by number of episodes", cause.value, CAUSE_EN[cause])
        put(f"cause_{i}_n", f"number of episodes with problem #{i}", len(ids))
        put(f"cause_{i}_pct", f"share of the batch with problem #{i}", f"{round(100 * len(ids) / n)}%")
        eg = [_short(x) for x in ids[:EXAMPLES]]
        put(f"cause_{i}_eg", f"example episode ids with problem #{i}", "、".join(eg), ", ".join(eg))

    unchecked = Counter(k for v in verdicts for k, why in v.not_checked.items() if "没启用" not in why)
    for i, (key, k) in enumerate(sorted(unchecked.items(), key=lambda kv: (-kv[1], kv[0]))[:3], 1):
        z, e = _check_name(key)
        put(f"unchecked_{i}", f"name of a check that could not run on some episodes (#{i})", z, e)
        put(f"unchecked_{i}_n", f"number of episodes where check #{i} could not run", k)

    ops = {ep.id: ep.operator_id for ep in episodes}
    if any(ops.values()):
        per: dict[str, list[Verdict]] = defaultdict(list)
        for v in verdicts:
            per[ops.get(v.episode_id) or ""].append(v)
        ranked_ops = sorted(per.items(), key=lambda kv: (-sum(x.label != Label.PASS for x in kv[1]), kv[0]))
        for j, (op, vs) in enumerate(ranked_ops[:TOP_OPERATORS], 1):
            put(f"op_{j}", f"id of operator #{j} (ranked by flagged episodes)", op or "（未填）", op or "(unknown)")
            put(f"op_{j}_n", f"episodes recorded by operator #{j}", len(vs))
            put(f"op_{j}_flagged", f"episodes of operator #{j} judged DOUBT or REJECT", sum(x.label != Label.PASS for x in vs))
            cc = Counter(c for x in vs for c in dict.fromkeys(f.cause for f in x.causes))
            if cc:
                c, k = sorted(cc.items(), key=lambda kv: (-kv[1], order.index(kv[0])))[0]
                put(f"op_{j}_cause", f"most frequent problem of operator #{j}", c.value, CAUSE_EN[c])
                put(f"op_{j}_cause_n", f"episodes of operator #{j} with that problem", k)
    return Facts(meaning, {"zh": zh, "en": en})
