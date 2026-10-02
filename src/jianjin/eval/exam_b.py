"""考卷 B：RoboVAD 帧级异常检测。按论文协议只评 csv 里列出的帧；micro AUC 把所有被评的帧合起来算。

拣金零样本：不在 RoboVAD 上训练；打分器在 validation 上选，test 只跑一次。
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from jianjin.checks import NotApplicable
from jianjin.eval.auc import roc_auc
from jianjin.ingest.robovad import read_robovad
from jianjin.scores import SCORERS

SETTINGS = ["in_domain", "cross_anomaly", "cross_task", "cross_task+anomaly"]

# 论文 arXiv 2609.17843 表 II 全表（2026-10-02 从 HTML 原文解析），帧级 AUC（micro, macro），百分数。
# 都在 RoboVAD 上训练过（有监督、弱监督或只用正常数据自监督）；只有 FIDeL 用关节数据，其余只看视频。
PAPER_TABLE_II = {
    "DSANet": {"in_domain": (47.3, 40.7), "cross_anomaly": (52.1, 51.0), "cross_task": (54.4, 49.8), "cross_task+anomaly": (59.7, 47.8)},
    "FIDeL（关节）": {"in_domain": (56.3, 54.0), "cross_anomaly": (52.1, 53.4), "cross_task": (58.1, 53.7), "cross_task+anomaly": (63.2, 51.2)},
    "FrameShield": {"in_domain": (52.6, 54.8), "cross_anomaly": (47.8, 44.0), "cross_task": (54.3, 48.6), "cross_task+anomaly": (55.1, 42.0)},
    "Jigsaw-VAD": {"in_domain": (64.1, 65.2), "cross_anomaly": (51.7, 51.1), "cross_task": (66.9, 70.9), "cross_task+anomaly": (57.6, 57.9)},
    "LaGOVAD": {"in_domain": (45.8, 44.3), "cross_anomaly": (45.2, 46.6), "cross_task": (47.9, 46.1), "cross_task+anomaly": (50.6, 45.4)},
    "VadCLIP": {"in_domain": (47.1, 42.0), "cross_anomaly": (51.5, 54.1), "cross_task": (48.0, 55.1), "cross_task+anomaly": (48.0, 51.9)},
    "UR-DMU": {"in_domain": (88.5, 94.4), "cross_anomaly": (61.3, 71.0), "cross_task": (71.8, 77.5), "cross_task+anomaly": (63.9, 71.9)},
    "FlowJigsaw-SSL": {"in_domain": (73.6, 72.8), "cross_anomaly": (79.6, 82.5), "cross_task": (65.6, 64.9), "cross_task+anomaly": (67.4, 64.5)},
    "FlowJigsaw-Sup.": {"in_domain": (94.6, 94.4), "cross_anomaly": (69.0, 71.9), "cross_task": (72.0, 73.8), "cross_task+anomaly": (66.7, 66.8)},
}

Segments = dict[str, list[tuple[int, int, int]]]  # 示范名 -> [(起, 止, 标签)]


def read_segments(path: Path) -> Segments:
    out: Segments = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["episode_name"]].append(
                (int(row["start_frame"]), int(row["end_frame_exclusive"]), int(row["label"]))
            )
    return dict(out)


def labels_from_annotations(n: int, anomalies: list[dict]) -> np.ndarray:
    y = np.zeros(n, dtype=np.int8)
    for a in anomalies:
        y[max(0, a["start_frame"]) : min(n, a["end_frame"])] = 1
    return y


def _validation_segments(root: Path, episodes: dict) -> Segments:
    with open(root / "splits" / "validation.csv", encoding="utf-8", newline="") as f:
        names = [r["episode_name"] for r in csv.DictReader(f)]
    out: Segments = {}
    for name in names:
        ep = episodes.get(name)
        if ep is None:
            out[name] = []
            continue
        n = int(ep.meta["frame_index"].max()) + 1
        y = labels_from_annotations(n, ep.meta["anomalies"])
        out[name] = [(i, i + 1, int(v)) for i, v in enumerate(y)]
    return out


def score_population(segments: Segments, episodes: dict, scorer) -> dict:
    all_s, all_y, per_ep = [], [], []
    missing, not_applicable, missing_frames = [], [], 0
    for name, segs in segments.items():
        ep = episodes.get(name)
        if ep is None:
            missing.append(name)
            continue
        try:
            s = scorer(ep)
        except NotApplicable:
            not_applicable.append(name)
            continue
        fi = ep.meta["frame_index"]
        by_frame = np.full(int(max(fi.max() + 1, max((e for _, e, _ in segs), default=0))), np.nan)
        by_frame[fi] = s
        ss, ys = [], []
        for start, end, label in segs:
            chunk = by_frame[start:end]
            ss.append(chunk)
            ys.append(np.full(len(chunk), label, dtype=np.int8))
        ss = np.concatenate(ss) if ss else np.zeros(0)
        ys = np.concatenate(ys) if ys else np.zeros(0, dtype=np.int8)
        ok = np.isfinite(ss)
        missing_frames += int((~ok).sum())
        all_s.append(ss[ok])
        all_y.append(ys[ok])
        per_ep.append(roc_auc(ss[ok], ys[ok]))
    s = np.concatenate(all_s) if all_s else np.zeros(0)
    y = np.concatenate(all_y) if all_y else np.zeros(0, dtype=np.int8)
    macro = [a for a in per_ep if not math.isnan(a)]
    return {
        "n_episodes": len(all_s),
        "n_frames": int(len(y)),
        "n_anomalous": int(y.sum()),
        "micro_auc": roc_auc(s, y),
        "macro_auc": float(np.mean(macro)) if macro else math.nan,
        "n_macro_episodes": len(macro),
        "missing_episodes": missing,
        "not_applicable": not_applicable,
        "missing_frames": missing_frames,
    }


def run_exam_b(root: str | Path, split: str, scorer_name: str) -> dict:
    root = Path(root)
    res = read_robovad(root)
    episodes = {e.id: e for e in res.episodes}
    if split == "validation":
        populations = {"validation": _validation_segments(root, episodes)}
    else:
        populations = {s: read_segments(root / "evaluation" / f"{s}.csv") for s in SETTINGS}
    scorer = SCORERS[scorer_name]
    results = [{"setting": k, **score_population(v, episodes, scorer)} for k, v in populations.items()]
    return {
        "exam": "B（RoboVAD 帧级异常检测；拣金零样本，只用关节数据）",
        "split": split,
        "scorer": scorer_name,
        "read_failures": res.failures,
        "results": results,
    }


def _pct(x: float) -> str:
    return "—" if math.isnan(x) else f"{x * 100:.1f}"


def format_table(out: dict) -> str:
    lines = [f"考卷 {out['exam']}", f"切分：{out['split']}；打分器：{out['scorer']}"]
    for r in out["results"]:
        lines += [
            f"[{r['setting']}] micro AUC {_pct(r['micro_auc'])}，macro AUC {_pct(r['macro_auc'])}"
            f"（macro 只算同时有正常和异常帧的 {r['n_macro_episodes']} 条）",
            f"  分母：{r['n_episodes']} 条示范、{r['n_frames']} 帧，其中异常 {r['n_anomalous']} 帧；"
            f"缺关节数据 {len(r['missing_episodes'])} 条，打不了分 {len(r['not_applicable'])} 条，"
            f"对不上帧号 {r['missing_frames']} 帧",
        ]
        if r["setting"] in SETTINGS:
            paper = "；".join(f"{m} {v[r['setting']][0]}/{v[r['setting']][1]}" for m, v in PAPER_TABLE_II.items())
            lines.append(f"  论文表 II（micro/macro，都训练过）：{paper}")
    return "\n".join(lines)
