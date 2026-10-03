"""考卷 A 的「换掉任务描述」：同一条视频，配上同一切分里别的数据集的任务描述，看视频评判能不能认出来。

每对是（原样示范，换了描述的示范），分数越高越像「做的不是这个任务」。
dev 上选分数、定阈值（原样示范报出不超过 5%）；test 只用 dev 定下的分数和阈值跑一次。
这是造出来的错配，比真实的「做错任务」容易：换进来的任务和原任务差得远。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from jianjin.checks import NotApplicable
from jianjin.eval.auc import roc_auc
from jianjin.eval.exam_a import MIN_INJECT_S, DatasetSpec, local_dir
from jianjin.eval.inject import swap_task
from jianjin.eval.metrics import wilson
from jianjin.ingest import load
from jianjin.judges.base import Curve
from jianjin.judges.cache import load_curve
from jianjin.types import Episode

CLEAN_FLAG_MAX = 0.05


def _tail(x: list[float], k: int = 3) -> float:
    return float(np.mean(x[-k:]))


def _success_end(c: Curve) -> float:
    if c.success is None:
        raise NotApplicable(f"{c.model} 不给成功概率")
    return 1.0 - _tail(c.success)


SWAP_SCORES = {
    "end": lambda c: 1.0 - _tail(c.progress),       # 结尾进度低
    "peak": lambda c: 1.0 - float(max(c.progress)),  # 全程最高进度低
    "mean": lambda c: 1.0 - float(np.mean(c.progress)),
    "success": _success_end,                         # 结尾成功概率低
}


def swap_pairs(specs: list[DatasetSpec], data_root: str | Path, split: str,
               seed: int = 0) -> list[tuple[Episode, Episode, str]]:
    """返回（原样，换了描述，相机）。短于 MIN_INJECT_S 秒或没有那路相机的示范不用。"""
    loaded = []
    for s in specs:
        if s.split != split:
            continue
        for ep in load(local_dir(data_root, s.repo)).episodes:
            cam = s.camera or next(iter(ep.cameras), None)
            if ep.duration >= MIN_INJECT_S and cam in ep.cameras:
                loaded.append((ep, cam))
    pool = sorted({ep.task for ep, _ in loaded})
    rng = np.random.default_rng(seed)
    out = []
    for ep, cam in loaded:
        if len(pool) < 2:
            break
        swapped, _ = swap_task(ep, rng, pool)
        out.append((ep, swapped, cam))
    return out


def run_swap(specs: list[DatasetSpec], data_root: str | Path, split: str, cache_dir: str | Path,
             judge: str = "robometer", score: str | None = None, threshold: float | None = None,
             seed: int = 0) -> dict:
    """score、threshold 不给就在这一批上选（只许在 dev 上这么做）。"""
    pairs = swap_pairs(specs, data_root, split, seed)
    curves, missing = [], 0
    for orig, swapped, cam in pairs:
        a = load_curve(Path(cache_dir), judge, orig, cam)
        b = load_curve(Path(cache_dir), judge, swapped, cam)
        if a is None or b is None:
            missing += 1
        else:
            curves.append((a, b))

    rows = []
    for name, fn in SWAP_SCORES.items():
        try:
            clean = np.array([fn(a) for a, _ in curves])
            bad = np.array([fn(b) for _, b in curves])
        except NotApplicable:
            continue
        y = np.r_[np.zeros(len(clean)), np.ones(len(bad))]
        rows.append({"name": name, "auc": roc_auc(np.r_[clean, bad], y), "clean": clean, "bad": bad})

    chosen_here = score is None
    if chosen_here:
        score = max(rows, key=lambda r: -1 if math.isnan(r["auc"]) else r["auc"])["name"] if rows else "end"
    row = next((r for r in rows if r["name"] == score), None)
    clean = row["clean"] if row else np.zeros(0)
    bad = row["bad"] if row else np.zeros(0)
    if threshold is None:
        threshold = float(np.quantile(clean, 1 - CLEAN_FLAG_MAX, method="higher")) if len(clean) else math.nan
    n = len(curves)
    detected = int((bad > threshold).sum())
    flagged = int((clean > threshold).sum())
    return {
        "exam": "A·换掉任务描述（同一条视频配别的数据集的任务；造出来的错配，不是真实做错任务）",
        "split": split,
        "judge": judge,
        "seed": seed,
        "n_candidates": len(pairs),
        "missing": missing,
        "n_pairs": n,
        "scores": [{"name": r["name"], "auc": r["auc"]} for r in rows],
        "score": score,
        "threshold": threshold,
        "chosen_on_this_split": chosen_here,
        "detected": detected,
        "detect_ci": wilson(detected, n),
        "clean_flagged": flagged,
        "clean_flag_ci": wilson(flagged, n),
        "tasks": sorted({o.task for o, _, _ in pairs}),
    }


def format_swap(out: dict) -> str:
    n = out["n_pairs"]

    def pct(k: int, ci) -> str:
        return "—" if n == 0 else f"{k}/{n} = {k / n:.0%}（95% 区间 {ci[0]:.0%} 到 {ci[1]:.0%}）"

    how = "在这一批上选的" if out["chosen_on_this_split"] else "沿用 dev 定的"
    lines = [
        f"考卷 {out['exam']}",
        f"切分：{out['split']}；评判：{out['judge']}；任务：{'; '.join(out['tasks'])}",
        f"分母：{out['n_candidates']} 对候选，缓存缺 {out['missing']} 对，实际评 {n} 对",
        "各分数的 AUC：" + "；".join(f"{r['name']} {r['auc']:.3f}" for r in out["scores"]),
        f"用分数 {out['score']}，阈值 {out['threshold']:.3f}（{how}）",
        f"  换了描述的检出：{pct(out['detected'], out['detect_ci'])}",
        f"  原样示范被报出：{pct(out['clean_flagged'], out['clean_flag_ci'])}",
    ]
    return "\n".join(lines)
