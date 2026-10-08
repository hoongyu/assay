"""考卷 C：RoboMIND 公开的失败示范（只读关节），按失败原因量检出。

「动作不流畅」「速度过快」：每个候选的阈值在成功示范的开发集上定（每个数据集报出都不超过 5% 的最小格点，
各自拿自己那批当参照）；在 RoboMIND 开发集上选（目标类检出减去同任务其他原因被报出的比例，大的胜；打平看 AUC；
时长以外的候选要严格大于时长基线才选；时长基线的差也不大于 0，这一类不加规则）；测试集用开发集选定的跑一次。
参照是同任务里原因不是这两类的示范，不够 5 条的没法比，单独记数。
「二次抓取」的参数已在 RoboVAD train 上定，RoboMIND 全部任务对它都是没见过的，只量一次；参照是同任务原因不是
「二次抓取」的示范。见 docs/superpowers/specs/2026-10-05-motion-quality-design.md 第 3 节。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from assay.checks.batch import MOTION_CANDIDATES, motion_scores, regrasp_scores
from assay.checks.config import CheckConfig
from assay.eval.auc import roc_auc
from assay.eval.metrics import wilson
from assay.ingest.robomind import read_robomind
from assay.types import Episode

SPLIT_FILE = Path("configs/exam_c_split.json")
TARGET = {"jerky": "动作不流畅", "fast": "速度过快"}
BASELINE = {"jerky": "duration", "fast": "duration_short"}
MOTION_CAUSES = set(TARGET.values())
RATIO_GRID = tuple(round(1.0 + 0.1 * k, 1) for k in range(1, 41))  # 1.1 到 5.0
MARGIN_GRID = tuple(range(9))  # 0 到 8
FLAG_MAX = 0.05
NAMES = {"duration": "时长（基线）", "stops": "中途停顿", "bursts": "动作段数", "jerk": "jerk",
         "duration_short": "时长（基线）", "peak_speed": "最快速度", "mean_speed": "平均速度"}
LAYOUTS = {("arm_hand", 7): "真机单臂 6 关节加夹爪（UR）", ("arm_hand", 8): "真机单臂 7 关节加夹爪（Franka）",
           ("arm_hand", 16): "真机双臂各 7 关节加夹爪", ("tiangong", 38): "天工仿真（灵巧手）",
           ("franka_sim", 8): "Franka 仿真"}


def _cause(ep: Episode) -> str | None:
    return ep.meta.get("robomind_failure_cause")


def _layout(ep: Episode) -> str:
    key = (ep.meta.get("layout", "?"), ep.state.shape[1])
    return LAYOUTS.get(key, f"{key[0]}，{key[1]} 维")


def _rate(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "ci": wilson(k, n)}


def _frac(r: dict) -> float:
    return r["k"] / r["n"] if r["n"] else 0.0


def _nan_low(x: float) -> float:
    return -1.0 if x != x else x


def pick_threshold(scores_by_set: dict[str, list[float]], kind: str) -> tuple[float | None, dict]:
    """每个数据集报出（分数大于阈值）都不超过 5% 的最小格点；没有就返回 None。"""
    grid = RATIO_GRID if kind == "ratio" else MARGIN_GRID
    for t in grid:
        flags = {name: {"k": int(sum(s > t for s in xs)), "n": len(xs)} for name, xs in scores_by_set.items() if xs}
        if all(f["k"] <= FLAG_MAX * f["n"] for f in flags.values()):
            return t, flags
    return None, {}


def choose(cands: dict[str, dict]) -> dict[str, dict | None]:
    """cands：候选名 → {threshold, gap, auc}。每一类按「差」选；时长以外的要严格大于时长基线；差不大于 0 不加规则。"""
    chosen: dict[str, dict | None] = {}
    for cls, base in BASELINE.items():
        ok = {c: m for c, m in cands.items() if MOTION_CANDIDATES[c][0] == cls and m["threshold"] is not None}
        name = base if base in ok else None
        rest = sorted((c for c in ok if c != base), key=lambda c: (ok[c]["gap"], _nan_low(ok[c]["auc"])), reverse=True)
        if rest and (name is None or ok[rest[0]]["gap"] > ok[base]["gap"]):
            name = rest[0]
        chosen[cls] = {"candidate": name, "threshold": ok[name]["threshold"]} if name and ok[name]["gap"] > 0 else None
    return chosen


def success_sets(config: str | Path, data_root: str | Path, robovad_root: str | Path, split: str) -> dict[str, list[Episode]]:
    """成功示范：dev 用考卷 A 开发集和 RoboVAD train 的正常示范，test 用考卷 A 测试集和 RoboVAD test 的正常示范。"""
    from assay.eval.exam_a import load_config, local_dir
    from assay.eval.exam_b import split_names
    from assay.ingest import load
    from assay.ingest.robovad import read_robovad

    out: dict[str, list[Episode]] = {}
    for s in load_config(config):
        if s.split == split:
            out[s.repo] = [e for e in load(local_dir(data_root, s.repo)).episodes if e.task not in s.exclude_tasks]
    vad = "train" if split == "dev" else "test"
    root = Path(robovad_root)
    out[f"RoboVAD {vad} 正常示范"] = [e for e in read_robovad(root, names=split_names(root, vad)).episodes
                                      if not e.meta["anomalies"]]
    return out


def _evaluate(episodes: list[Episode], scores: dict, cand: str, threshold: float) -> dict:
    """目标类检出、同任务其他原因被报出、AUC。只数能比的示范。"""
    target_cause = TARGET[MOTION_CANDIDATES[cand][0]]
    targets = [e for e in episodes if _cause(e) == target_cause and e.id in scores]
    others = [e for e in episodes if _cause(e) and _cause(e) not in MOTION_CAUSES and e.id in scores]

    def s(e):
        return scores[e.id].scores[cand]

    t = _rate(sum(s(e) > threshold for e in targets), len(targets))
    o = _rate(sum(s(e) > threshold for e in others), len(others))
    y = np.array([1] * len(targets) + [0] * len(others), dtype=np.int8)
    x = np.array([s(e) if s(e) == s(e) else -1e9 for e in targets + others], dtype=np.float64)  # NaN 排最后
    return {"target": t, "others": o, "gap": _frac(t) - _frac(o),
            "auc": roc_auc(x, y) if len(targets) and len(others) else float("nan")}


def _by_layout(episodes: list[Episode], scores: dict, cand: str, threshold: float) -> dict:
    out = {}
    for lay in sorted({_layout(e) for e in episodes}):
        sub = [e for e in episodes if _layout(e) == lay]
        r = _evaluate(sub, scores, cand, threshold)
        if r["target"]["n"] or r["others"]["n"]:
            out[lay] = {"target": r["target"], "others": r["others"]}
    return out


def _robomind(root: str | Path, split: str, split_file: str | Path) -> tuple[list[Episode], int]:
    tasks = json.loads(Path(split_file).read_text(encoding="utf-8"))["tasks"]
    res = read_robomind(Path(root))
    return [e for e in res.episodes if tasks.get(e.meta["task_dir"]) == split], len(res.failures)


def run_exam_c(root: str | Path, split: str, success: dict[str, list[Episode]], split_file: str | Path = SPLIT_FILE,
               cfg: CheckConfig | None = None, dev_result: dict | None = None) -> dict:
    """dev 上定阈值、选候选；test 必须带 dev_result，用它选定的跑一次。"""
    cfg = cfg or CheckConfig()
    if split != "dev" and dev_result is None:
        raise ValueError("test 要用 dev 选定的：带上 dev 的结果")
    episodes, unreadable = _robomind(root, split, split_file)
    reference = [e for e in episodes if _cause(e) and _cause(e) not in MOTION_CAUSES]
    scores, skipped = motion_scores(episodes, cfg, reference=reference)
    success_scores = {name: motion_scores(eps, cfg)[0] for name, eps in success.items()}

    def success_flags(cand, t):
        return {name: _rate(sum(s.scores[cand] > t for s in sc.values()), len(sc))
                for name, sc in success_scores.items() if sc}

    counts = {"episodes": len(episodes), "unreadable": unreadable,
              "by_cause": dict(Counter(_cause(e) or "（没有原因目录）" for e in episodes)),
              "comparable": dict(Counter(_cause(e) or "（没有原因目录）" for e in episodes if e.id in scores)),
              "skipped": dict(Counter(skipped.values()))}
    out = {"exam": "C·RoboMIND 失败示范（只读关节）：动作不流畅、速度过快", "split": split,
           "chosen_on_this_split": dev_result is None, "counts": counts}
    if dev_result is None:
        cands = {}
        for cand, (_, kind) in MOTION_CANDIDATES.items():
            t, flags = pick_threshold({n: [s.scores[cand] for s in sc.values()] for n, sc in success_scores.items()}, kind)
            m = _evaluate(episodes, scores, cand, t) if t is not None else {"target": None, "others": None,
                                                                          "gap": float("nan"), "auc": float("nan")}
            cands[cand] = {"class": MOTION_CANDIDATES[cand][0], "kind": kind, "threshold": t, "success": flags, **m}
        out["candidates"] = cands
        out["chosen"] = choose(cands)
    else:
        out["chosen"] = dev_result["chosen"]
        out["dev_thresholds"] = {c: m["threshold"] for c, m in dev_result["candidates"].items()}
        out["candidates"] = {c: {"class": MOTION_CANDIDATES[c][0], "threshold": t,
                                 **(_evaluate(episodes, scores, c, t) if t is not None else {})}
                             for c, t in out["dev_thresholds"].items()}
    self_scores, _ = motion_scores(episodes, cfg)
    result = {}
    for cls, ch in out["chosen"].items():
        if ch is None:
            result[cls] = None
            continue
        cand, t = ch["candidate"], ch["threshold"]
        result[cls] = {**_evaluate(episodes, scores, cand, t), "candidate": cand, "threshold": t,
                       "success": success_flags(cand, t), "by_layout": _by_layout(episodes, scores, cand, t),
                       "self_reference": {k: v for k, v in _evaluate(episodes, self_scores, cand, t).items()
                                          if k in ("target", "others")}}
    out["result"] = result
    return out


def run_regrasp_c(root: str | Path, cfg: CheckConfig | None = None) -> dict:
    """二次抓取规则（参数在 RoboVAD train 上定）在 RoboMIND 全部任务上量一次。"""
    cfg = cfg or CheckConfig()
    res = read_robomind(Path(root))
    episodes = res.episodes
    reference = [e for e in episodes if _cause(e) and _cause(e) != "二次抓取"]
    scores, skipped = regrasp_scores(episodes, cfg, reference=reference)
    targets = [e for e in episodes if _cause(e) == "二次抓取" and e.id in scores]
    others = [e for e in episodes if _cause(e) and _cause(e) != "二次抓取" and e.id in scores]

    def hit(es):
        return sum(scores[e.id].excess > cfg.regrasp_margin for e in es)

    y = np.array([1] * len(targets) + [0] * len(others), dtype=np.int8)
    x = np.array([scores[e.id].excess for e in targets + others], dtype=np.float64)
    by_layout = {}
    for lay in sorted({_layout(e) for e in targets + others}):
        t = [e for e in targets if _layout(e) == lay]
        o = [e for e in others if _layout(e) == lay]
        by_layout[lay] = {"target": _rate(hit(t), len(t)), "others": _rate(hit(o), len(o))}
    return {
        "exam": "C·RoboMIND 失败示范（只读关节）：二次抓取（参数沿用 RoboVAD train 选定的，没在 RoboMIND 上调）",
        "source": cfg.regrasp_source, "margin": cfg.regrasp_margin,
        "episodes": len(episodes), "unreadable": len(res.failures),
        "n_target_all": sum(_cause(e) == "二次抓取" for e in episodes),
        "target": _rate(hit(targets), len(targets)), "others": _rate(hit(others), len(others)),
        "auc": roc_auc(x, y) if len(targets) and len(others) else float("nan"),
        "by_layout": by_layout, "skipped": dict(Counter(skipped.values())),
    }


def _fmt(r: dict | None) -> str:
    if not r or not r["n"]:
        return "0/0"
    lo, hi = r["ci"]
    return f"{r['k']}/{r['n']} = {r['k'] / r['n']:.0%}（{lo:.0%} 到 {hi:.0%}）"


def format_exam_c(out: dict) -> str:
    how = "在这一批上选的" if out["chosen_on_this_split"] else "沿用 dev 选定的"
    c = out["counts"]
    lines = [out["exam"], f"切分：{out['split']}（{how}）；{c['episodes']} 条示范，读不了 {c['unreadable']} 条",
             "各原因条数（能比的）：" + "；".join(f"{k} {v}（{c['comparable'].get(k, 0)}）" for k, v in c["by_cause"].items())]
    lines.append("\n各候选（阈值在成功示范开发集上定；检出减其他原因报出叫「差」）：")
    for cand, m in out["candidates"].items():
        if not m.get("target"):
            lines.append(f"  {TARGET[m['class']]}·{NAMES[cand]}：成功示范上压不到 5%，出局")
            continue
        lines.append(f"  {TARGET[m['class']]}·{NAMES[cand]}（阈值 {m['threshold']}）：检出 {_fmt(m['target'])}，"
                     f"其他原因报出 {_fmt(m['others'])}，差 {m['gap']:+.1%}，AUC {m['auc']:.3f}")
    for cls, r in out["result"].items():
        if r is None:
            lines.append(f"\n{TARGET[cls]}：没有候选比时长基线强、差也不大于 0，不加规则")
            continue
        lines += [f"\n{TARGET[cls]}：选中「{NAMES[r['candidate']]}」，阈值 {r['threshold']}",
                  f"  检出 {_fmt(r['target'])}；同任务其他原因报出 {_fmt(r['others'])}；AUC {r['auc']:.3f}",
                  "  成功示范报出：" + "；".join(f"{n} {_fmt(v)}" for n, v in r["success"].items()),
                  f"  拿整批自己当参照：检出 {_fmt(r['self_reference']['target'])}，"
                  f"其他原因报出 {_fmt(r['self_reference']['others'])}",
                  "  按机型：" + "；".join(f"{k} 检出 {_fmt(v['target'])}、其他 {_fmt(v['others'])}"
                                        for k, v in r["by_layout"].items())]
    if c["skipped"]:
        lines.append("\n没法比的：" + "；".join(f"{why}（{n} 条）" for why, n in c["skipped"].items()))
    return "\n".join(lines)


def format_regrasp_c(out: dict) -> str:
    lines = [out["exam"], f"夹爪读{'从臂' if out['source'] == 'state' else '主臂'}，余量 {out['margin']}；"
             f"{out['episodes']} 条示范，读不了 {out['unreadable']} 条；二次抓取共 {out['n_target_all']} 条",
             f"二次抓取检出：{_fmt(out['target'])}；同任务其他原因报出：{_fmt(out['others'])}；AUC {out['auc']:.3f}",
             "按机型：" + "；".join(f"{k} 检出 {_fmt(v['target'])}、其他 {_fmt(v['others'])}"
                                  for k, v in out["by_layout"].items())]
    if out["skipped"]:
        lines.append("没法比的：" + "；".join(f"{why}（{n} 条）" for why, n in out["skipped"].items()))
    return "\n".join(lines)
