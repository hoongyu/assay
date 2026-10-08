"""考卷 A 的「换掉任务描述」：同一条视频，配上同一切分里别的数据集的任务描述，看视频评判能不能认出来。

每对是（原样示范，换了描述的示范），分数越高越像「做的不是这个任务」。
dev 上选分数、定阈值（原样示范报出不超过 5%）；test 只用 dev 定下的分数和阈值跑一次。
选分数看的是「原样报出不超过 5% 时能检出多少」，不看 AUC：我们只在这个工作点上用它。
只在 rel_ 开头的分数里选：甄数判「指令不符」只和同任务那批比，不带 rel_ 的分数规则用不上，只列出来参考
（2026-10-04 改：TOPReward 对话格式在 dev 上绝对的 success 检出最多，原来的选法会挑中它）。

rel_ 开头的分数和同一任务描述下那批原样示范比（中位数、四分位距，下限见 checks/video.py 的 SPREAD_FLOOR），
和考卷 A 的批次规则一个思路。分数定义和换算都在 checks/video.py，甄数判「指令不符」用的是同一套。
2026-10-03 在 dev 上看到：不同任务的进度高低本来就差很多（「放进盒子」最高进度挤在 0.92 到 0.94，
「叠方块」散在 0.72 到 0.91），换了描述只让进度低 0.07 左右，一个绝对阈值只能迁就散的那批。
这是造出来的错配，不是真实的「做错任务」。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from assay.checks import CheckConfig, NotApplicable
from assay.checks.video import BASE_SCORES, relative
from assay.eval.auc import roc_auc
from assay.eval.exam_a import MIN_INJECT_S, DatasetSpec, local_dir
from assay.eval.inject import swap_task
from assay.eval.metrics import wilson
from assay.ingest import load
from assay.judges.cache import load_curve
from assay.types import Episode

CLEAN_FLAG_MAX = 0.05
MIN_BATCH = CheckConfig().min_task_episodes  # 同任务比较至少要这么多条原样示范

SCORE_NAMES = [*BASE_SCORES, *(f"rel_{k}" for k in BASE_SCORES)]


def _relative(clean: np.ndarray, bad: np.ndarray, clean_tasks: np.ndarray,
              bad_tasks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """每个任务描述下，用那批原样示范的中位数和四分位距换算：比同任务的常态差多少。"""
    rc, rb = np.empty(len(clean)), np.empty(len(bad))
    for t in set(clean_tasks) | set(bad_tasks):
        ref = clean[clean_tasks == t]
        if len(ref) < MIN_BATCH:
            raise NotApplicable(f"任务「{t}」只有 {len(ref)} 条原样示范，不够同任务比较")
        rc[clean_tasks == t] = relative(clean[clean_tasks == t], ref)
        rb[bad_tasks == t] = relative(bad[bad_tasks == t], ref)
    return rc, rb


def _row(name: str, clean: np.ndarray, bad: np.ndarray) -> dict:
    """AUC，以及原样报出不超过 CLEAN_FLAG_MAX 时的阈值和检出。"""
    y = np.r_[np.zeros(len(clean)), np.ones(len(bad))]
    thr = float(np.quantile(clean, 1 - CLEAN_FLAG_MAX, method="higher")) if len(clean) else math.nan
    return {"name": name, "auc": roc_auc(np.r_[clean, bad], y), "clean": clean, "bad": bad,
            "threshold_at_max": thr, "detected_at_max": int((bad > thr).sum())}


def _choose(rows: list[dict]) -> str:
    """在规则能用的分数（rel_ 开头）里，先比原样报出不超过 5% 时的检出，再比 AUC。一个都没有就返回 rel_end，
    它不在 rows 里，阈值记 nan、检出 0：这一批规则查不了。"""
    usable = [r for r in rows if r["name"].startswith("rel_")]
    if not usable:
        return "rel_end"

    def rank(r: dict) -> tuple:
        return (r["detected_at_max"], -1.0 if math.isnan(r["auc"]) else r["auc"])

    return max(usable, key=rank)["name"]


def swap_pairs(specs: list[DatasetSpec], data_root: str | Path, split: str,
               seed: int = 0) -> list[tuple[Episode, Episode, str]]:
    """返回（原样，换了描述，相机）。短于 MIN_INJECT_S 秒或没有那路相机的示范不用，配置里排除的任务也不用。

    没写相机的数据集（只用关节数据的双臂开发集）整个跳过：它们的元数据里列着相机，但本地没有视频，
    混进来会改变别的示范换到哪个描述，和已有缓存对不上（2026-10-03 加双臂开发集时漏了这里，同日修）。
    """
    loaded = []
    for s in specs:
        if s.split != split or not s.camera:
            continue
        for ep in load(local_dir(data_root, s.repo)).episodes:
            if ep.task not in s.exclude_tasks and ep.duration >= MIN_INJECT_S and s.camera in ep.cameras:
                loaded.append((ep, s.camera))
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
            curves.append((a, b, orig.task, swapped.task))
    clean_tasks = np.array([c[2] for c in curves])
    bad_tasks = np.array([c[3] for c in curves])

    rows = []
    for name, fn in BASE_SCORES.items():
        try:
            clean = np.array([fn(c[0]) for c in curves])
            bad = np.array([fn(c[1]) for c in curves])
        except NotApplicable:
            continue
        rows.append(_row(name, clean, bad))
        try:
            rows.append(_row(f"rel_{name}", *_relative(clean, bad, clean_tasks, bad_tasks)))
        except NotApplicable:
            pass

    chosen_here = score is None
    if chosen_here:
        score = _choose(rows)
    row = next((r for r in rows if r["name"] == score), None)
    clean = row["clean"] if row else np.zeros(0)
    bad = row["bad"] if row else np.zeros(0)
    if threshold is None:
        threshold = row["threshold_at_max"] if row else math.nan
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
        "scores": [{"name": r["name"], "auc": r["auc"], "detected_at_max": r["detected_at_max"]} for r in rows],
        "clean_flag_max": CLEAN_FLAG_MAX,
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
    ]
    if out["chosen_on_this_split"]:
        lines.append(f"各分数：AUC，以及在这一批上按原样报出不超过 {out['clean_flag_max']:.0%} 定阈值时的检出"
                     "（只在 rel_ 开头的里选：「指令不符」规则只和同任务那批比）")
        lines += [f"  {r['name']}：AUC {r['auc']:.3f}，检出 {r['detected_at_max']}/{n}" for r in out["scores"]]
    else:  # 在这一批上现定阈值的检出不能当成绩，不打印
        lines.append("各分数的 AUC（只供参考，成绩只看 dev 选中的那个）：" +
                     "；".join(f"{r['name']} {r['auc']:.3f}" for r in out["scores"]))
    lines += [
        f"用分数 {out['score']}，阈值 {out['threshold']:.3f}（{how}）",
        f"  换了描述的检出：{pct(out['detected'], out['detect_ci'])}",
        f"  原样示范被报出：{pct(out['clean_flagged'], out['clean_flag_ci'])}",
    ]
    return "\n".join(lines)
