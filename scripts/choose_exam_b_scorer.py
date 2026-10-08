"""在 RoboVAD 的 validation 划分上比打分器，按 micro AUC 选一个；test 只用选中的那个跑一次。"""

from __future__ import annotations

import sys

from assay.eval.exam_b import run_exam_b
from assay.scores import SCORERS


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    best = None
    for name in SCORERS:
        (r,) = run_exam_b("data/robovad", "validation", name)["results"]
        print(f"{name}: micro {r['micro_auc']:.3f}  macro {r['macro_auc']:.3f}  ({r['n_episodes']} 条，{r['n_frames']} 帧，异常 {r['n_anomalous']} 帧)")
        if best is None or r["micro_auc"] > best[1]:
            best = (name, r["micro_auc"])
    print(f"选：{best}")


if __name__ == "__main__":
    main()
