"""在考卷 A 开发集上扫关节体检的阈值。

原则：每个开发数据集的原样报出率都不超过 5%；在这个前提下定位对的最多（各数据集加总）；再打平就选原样报出最少的。
数据集大小差很多（armnet 一家 869 条），合起来算 5% 会被大的那家主导，所以按数据集分别卡（2026-10-03 改）。
每组参数只跑一次，按数据集拆开的数来自同一次运行（run_exam_a 的 by_dataset）。

用法：uv run python scripts/tune_exam_a.py [insert_pause|truncate_end]，不给就全扫。
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import replace

from assay.checks import CheckConfig
from assay.eval.exam_a import load_config, run_exam_a

GRID = {
    "insert_pause": {"still_rate": [0.005, 0.01, 0.02, 0.04], "pause_s": [2.0, 3.0, 4.0]},
    # 2026-10-03 改：end_pose_max 的单位从「活动幅度」换成「收尾四分位距的倍数」，加了下限 end_pose_floor
    "truncate_end": {"end_pose_max": [2.0, 3.0, 4.0, 6.0, 8.0], "end_pose_floor": [0.02, 0.05, 0.1]},
}
MAX_CLEAN_FLAG = 0.05


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    specs = load_config("configs/exam_a.toml")
    for kind in sys.argv[1:] or list(GRID):
        grid = GRID[kind]
        print(f"== {kind}")
        best = None
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid, values))
            out = run_exam_a(specs, "data", "dev", cfg=replace(CheckConfig(), **params))
            i = next(n for n, x in enumerate(out["kinds"]) if x["kind"] == kind)
            k = out["kinds"][i]
            parts, ok = [], True
            for repo, scores in out["by_dataset"].items():
                d = scores[i]
                ok &= d["clean_flagged"] <= MAX_CLEAN_FLAG * d["n_clean"]
                parts.append(f"{repo.split('/')[-1]} 原样 {d['clean_flagged']}/{d['n_clean']} 定位 {d['localized']}/{d['n_injected']}")
            print(f"{params}  定位 {k['localized']}/{k['n_injected']}  原样 {k['clean_flagged']}/{k['n_clean']}"
                  f"{'' if ok else '  超限'}\n    " + " | ".join(parts))
            rank = (k["localized"], -k["clean_flagged"])
            if ok and (best is None or rank > best[1]):
                best = (params, rank)
        print(f"选：{best}")


if __name__ == "__main__":
    main()
