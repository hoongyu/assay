"""在考卷 A 开发集上扫关节体检的阈值。

原则：原样示范上的报出率不超过 5%；在这个前提下定位对的最多；再打平就选原样报出最少的。
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import replace

from jianjin.checks import CheckConfig
from jianjin.eval.exam_a import load_config, run_exam_a

GRID = {
    "insert_pause": {"still_rate": [0.005, 0.01, 0.02, 0.04], "pause_s": [2.0, 3.0, 4.0]},
    "truncate_end": {"end_pose_max": [0.2, 0.3, 0.4, 0.5, 0.6]},
}
MAX_CLEAN_FLAG = 0.05


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    specs = load_config("configs/exam_a.toml")
    for kind, grid in GRID.items():
        print(f"== {kind}")
        best = None
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid, values))
            out = run_exam_a(specs, "data", "dev", cfg=replace(CheckConfig(), **params))
            k = next(x for x in out["kinds"] if x["kind"] == kind)
            ok = k["clean_flag_rate"] <= MAX_CLEAN_FLAG
            print(
                f"{params}  检出 {k['detected']}/{k['n_injected']}  定位 {k['localized']}/{k['n_injected']}"
                f"  原样报出 {k['clean_flagged']}/{k['n_clean']}{'' if ok else '  超限'}"
            )
            rank = (k["localized"], -k["clean_flagged"])
            if ok and (best is None or rank > best[1]):
                best = (params, rank)
        print(f"选：{best}")


if __name__ == "__main__":
    main()
