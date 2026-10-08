"""考卷 B 按示范计：RoboVAD 的示范有没有异常标注，对上「进度回落」规则报没报、报的时间对不对。

阈值在 train 的正常示范上取 95 分位（正常的最多报出 5%）；test 只用 train 定的阈值跑一次。
test 用 annotations.json 里的标注，不用 evaluation/*.csv（那是逐帧评测的帧清单）。
规则和 `assay check` 里的「进度回落」是同一个函数（去掉回原位那段）。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from assay.checks.video import rule_drawdown
from assay.diagnose import overlaps
from assay.eval.auc import roc_auc
from assay.eval.exam_b import split_names
from assay.eval.metrics import wilson
from assay.ingest.robovad import read_robovad
from assay.judges.cache import load_curve
from assay.types import Episode

NORMAL_FLAG_MAX = 0.05


def annotated_seconds(ep: Episode) -> list[tuple[float, float]]:
    fi, ts = ep.meta["frame_index"], ep.timestamps
    return [(float(np.interp(a["start_frame"], fi, ts)), float(np.interp(a["end_frame"], fi, ts)))
            for a in ep.meta["anomalies"]]


def run_episode_b(root: str | Path, split: str, cache_dir: str | Path = "cache", judge: str = "robometer",
                  camera: str = "left", threshold: float | None = None) -> dict:
    """threshold 不给就在这一批的正常示范上定（只许在 train 上这么做）。"""
    root = Path(root)
    episodes = read_robovad(root, names=split_names(root, split)).episodes
    rows, missing = [], 0
    for ep in episodes:
        c = load_curve(Path(cache_dir), judge, ep, camera)
        if c is None:
            missing += 1
            continue
        d, t0, t1, _, _ = rule_drawdown(ep, c)
        rows.append((bool(ep.meta["anomalies"]), d, (t0, t1), annotated_seconds(ep)))
    y = np.array([r[0] for r in rows], dtype=bool)
    s = np.array([r[1] for r in rows], dtype=np.float64)
    chosen_here = threshold is None
    if chosen_here:
        normal = s[~y]
        threshold = float(np.quantile(normal, 1 - NORMAL_FLAG_MAX, method="higher")) if len(normal) else math.nan
    flagged = s > threshold
    pos, neg = int(y.sum()), int((~y).sum())
    detected = int((flagged & y).sum())
    normal_flagged = int((flagged & ~y).sum())
    localized = sum(1 for r, f in zip(rows, flagged) if f and r[0] and any(overlaps(r[2], a) for a in r[3]))
    return {
        "exam": "B·按示范计（RoboVAD：示范有没有异常标注；规则是进度从最高点回落）",
        "split": split,
        "judge": judge,
        "camera": camera,
        "n_episodes": len(rows),
        "missing": missing,
        "n_anomalous": pos,
        "n_normal": neg,
        "auc": roc_auc(s, y.astype(np.int8)),
        "threshold": threshold,
        "chosen_on_this_split": chosen_here,
        "detected": detected,
        "detect_ci": wilson(detected, pos),
        "localized": localized,
        "localize_ci": wilson(localized, pos),
        "normal_flagged": normal_flagged,
        "normal_flag_ci": wilson(normal_flagged, neg),
    }


def format_episode_b(out: dict) -> str:
    def pct(k: int, n: int, ci) -> str:
        return "—" if n == 0 else f"{k}/{n} = {k / n:.0%}（95% 区间 {ci[0]:.0%} 到 {ci[1]:.0%}）"

    how = "在这一批的正常示范上定的" if out["chosen_on_this_split"] else "沿用 train 定的"
    return "\n".join([
        f"考卷 {out['exam']}",
        f"切分：{out['split']}；评判：{out['judge']}（{out['camera']} 路相机）",
        f"分母：有缓存的 {out['n_episodes']} 条（缺 {out['missing']} 条），其中有异常标注 {out['n_anomalous']} 条、"
        f"正常 {out['n_normal']} 条",
        f"AUC {out['auc']:.3f}；阈值 {out['threshold']:.3f}（{how}）",
        f"  有异常的检出：{pct(out['detected'], out['n_anomalous'], out['detect_ci'])}",
        f"  报的时间落在标注区间：{pct(out['localized'], out['n_anomalous'], out['localize_ci'])}",
        f"  正常的被报出：{pct(out['normal_flagged'], out['n_normal'], out['normal_flag_ci'])}",
    ])
