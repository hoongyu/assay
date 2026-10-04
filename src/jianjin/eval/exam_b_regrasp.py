"""考卷 B 二次抓取：RoboVAD 按示范计，看「二次抓取」规则报没报；另外分异常类型报检出。

参照用 train 里同任务的正常示范：RoboVAD test 异常占六成多，拿这批自己当参照，常见上限会被异常抬高
（checks/batch.py 的约定：考卷里一整批都有问题，就传原样示范进来比）。train 里没有的任务（倒米、套环）
只能拿这批自己当参照，单独报。
读主臂还是从臂、余量取几，只在 train 上选：正常示范最多报出 5% 的前提下，抓空和掉落的检出最多；
打平看 AUC，再打平取余量大的。test 用 train 选定的跑一次。规则和 `jianjin check` 里的是同一个函数。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from jianjin.checks.batch import regrasp_scores
from jianjin.checks.config import CheckConfig
from jianjin.eval.auc import roc_auc
from jianjin.eval.exam_b import split_names
from jianjin.eval.metrics import wilson
from jianjin.ingest.robovad import read_robovad
from jianjin.types import Episode

TARGET = ("failed_grasp", "object_drop")
TYPES = ("failed_grasp", "object_drop", "collision", "placement_failure", "object_not_dropped")
NAMES = {"failed_grasp": "抓空", "object_drop": "掉落", "collision": "碰撞", "placement_failure": "放置失败",
         "object_not_dropped": "该掉没掉"}
SOURCES = ("action", "state")
MARGINS = tuple(range(7))
NORMAL_FLAG_MAX = 0.05


def _types(ep: Episode) -> set[str]:
    return {a["type"] for a in ep.meta["anomalies"]}


def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "ci": wilson(k, n)}


def _nan_low(x: float) -> float:
    return -1.0 if x != x else x


def _summary(episodes: list[Episode], scores: dict, margin: int) -> dict:
    rows = [ep for ep in episodes if ep.id in scores]
    flagged = {ep.id for ep in rows if scores[ep.id].excess > margin}
    normal = [ep for ep in rows if not ep.meta["anomalies"]]
    target = [ep for ep in rows if _types(ep) & set(TARGET)]
    anomalous = [ep for ep in rows if ep.meta["anomalies"]]

    def hit(eps):
        return sum(ep.id in flagged for ep in eps)

    y = np.array([1] * len(target) + [0] * len(normal), dtype=np.int8)
    s = np.array([scores[ep.id].excess for ep in target + normal], dtype=np.float64)
    return {
        "n_episodes": len(rows),
        "normal_flagged": _rate(hit(normal), len(normal)),
        "target_detected": _rate(hit(target), len(target)),
        "anomalous_detected": _rate(hit(anomalous), len(anomalous)),
        "by_type": {t: _rate(hit([e for e in anomalous if t in _types(e)]), sum(t in _types(e) for e in anomalous))
                    for t in TYPES},
        "auc": roc_auc(s, y) if len(target) and len(normal) else float("nan"),
    }


def _score(episodes: list[Episode], reference: list[Episode], cfg: CheckConfig):
    """见过的任务拿 reference 比；reference 里没有的任务拿这批自己比。"""
    seen_tasks = {ep.task for ep in reference}
    seen = [ep for ep in episodes if ep.task in seen_tasks]
    unseen = [ep for ep in episodes if ep.task not in seen_tasks]
    s1, k1 = regrasp_scores(seen, cfg, reference=reference)
    s2, k2 = regrasp_scores(unseen, cfg) if unseen else ({}, {})
    return seen, unseen, s1, s2, {**k1, **k2}


def run_regrasp_b(root: str | Path, split: str, source: str | None = None, margin: int | None = None) -> dict:
    """source、margin 不给就在这一批上选（只许在 train 上这么做）。"""
    root = Path(root)
    episodes = read_robovad(root, names=split_names(root, split)).episodes
    reference = [ep for ep in read_robovad(root, names=split_names(root, "train")).episodes if not ep.meta["anomalies"]]
    chosen_here = source is None or margin is None
    candidates = []
    if chosen_here:
        for src in SOURCES:
            seen, _, s1, _, _ = _score(episodes, reference, replace(CheckConfig(), regrasp_source=src))
            for m in MARGINS:
                r = _summary(seen, s1, m)
                candidates.append({"source": src, "margin": m, "target_detected": r["target_detected"],
                                   "normal_flagged": r["normal_flagged"], "auc": r["auc"]})
        ok = [c for c in candidates if c["normal_flagged"]["k"] <= NORMAL_FLAG_MAX * c["normal_flagged"]["n"]]
        best = max(ok or candidates, key=lambda c: (c["target_detected"]["k"], _nan_low(c["auc"]), c["margin"]))
        source, margin = best["source"], best["margin"]
    cfg = replace(CheckConfig(), regrasp_source=source)
    seen, unseen, s1, s2, skipped = _score(episodes, reference, cfg)
    return {
        "exam": "B·二次抓取（RoboVAD 按示范计；规则：夹爪开合次数超过同任务常见上限加余量）",
        "split": split,
        "source": source,
        "margin": margin,
        "quantile": cfg.regrasp_quantile,
        "band": cfg.grip_band,
        "chosen_on_this_split": chosen_here,
        "n_reference": len(reference),
        "candidates": candidates,
        "seen": _summary(seen, s1, margin),
        "unseen": _summary(unseen, s2, margin) if unseen else None,
        "unseen_tasks": sorted({ep.task for ep in unseen}),
        "skipped": dict(Counter(skipped.values())),
    }


def _fmt(r: dict) -> str:
    if not r["n"]:
        return "0/0"
    lo, hi = r["ci"]
    return f"{r['k']}/{r['n']} = {r['k'] / r['n']:.0%}（{lo:.0%} 到 {hi:.0%}）"


def format_regrasp_b(out: dict) -> str:
    how = "在这一批上选的" if out["chosen_on_this_split"] else "沿用 train 选定的"
    lines = [
        out["exam"],
        f"切分：{out['split']}；夹爪读{'主臂' if out['source'] == 'action' else '从臂'}；余量 {out['margin']}（{how}）；"
        f"常见上限取参照批开合次数的 {out['quantile']:.0%} 分位",
        f"参照：train 里同任务的正常示范，{out['n_reference']} 条",
    ]
    groups = (("seen", "train 里见过的任务（拿 train 的正常示范当参照）"),
              ("unseen", f"train 里没有的任务（{'、'.join(out['unseen_tasks'])}，拿这批自己当参照）"))
    for key, title in groups:
        r = out.get(key)
        if not r:
            continue
        lines += [
            f"\n{title}：{r['n_episodes']} 条",
            f"  抓空或掉落的示范报出：{_fmt(r['target_detected'])}",
            f"  有任何异常标注的报出：{_fmt(r['anomalous_detected'])}",
            f"  正常示范被报出：{_fmt(r['normal_flagged'])}",
            "  分异常类型：" + "；".join(f"{NAMES[t]} {_fmt(v)}" for t, v in r["by_type"].items() if v["n"]),
            f"  AUC（抓空或掉落对正常）：{r['auc']:.3f}",
        ]
    if out["candidates"]:
        lines.append("\n各组合（正常示范最多报出 5% 的前提下，抓空和掉落检出最多；打平看 AUC，再打平取余量大的）：")
        for c in out["candidates"]:
            lines.append(f"  {c['source']:6s} 余量 {c['margin']}：检出 {c['target_detected']['k']}/{c['target_detected']['n']}，"
                         f"正常报出 {c['normal_flagged']['k']}/{c['normal_flagged']['n']}，AUC {c['auc']:.3f}")
    if out["skipped"]:
        lines.append("\n没法比的：" + "；".join(f"{why}（{n} 条）" for why, n in out["skipped"].items()))
    return "\n".join(lines)
