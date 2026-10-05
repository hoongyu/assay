"""给几批数据带上 --summary 重出英文报告，再数本批小结的通过率（设计见
docs/superpowers/specs/2026-10-05-english-report-design.md 第 4 节）。

要先在同一个终端里设好 NEBIUS_API_KEY。叠毛巾那批不在这里，留到录视频时现场跑。
小结按事实缓存在 cache/summary/，中文报告之后不用密钥就能从缓存重出。

    uv run python scripts/run_summaries.py                  # 放方块、叠方块、分拣、乐高块 4 批
    uv run python scripts/run_summaries.py --with-robovad   # 再加 RoboVAD（1,078 条，不配关键帧）
"""

from __future__ import annotations

import os
import sys

from jianjin.cli import main

BATCHES = [  # 数据目录、视频评判缓存用的相机、报告名
    ("data/lerobot__svla_so100_pickplace", "observation.images.top", "svla_so100_pickplace"),
    ("data/lerobot__svla_so100_stacking", "observation.images.top", "svla_so100_stacking"),
    ("data/lerobot__svla_so100_sorting", "observation.images.top", "svla_so100_sorting"),
    ("data/lerobot__svla_so101_pickplace", "observation.images.up", "svla_so101_pickplace"),
]
ROBOVAD = ("data/robovad", "left", "robovad")


def run(argv: list[str]) -> int:
    if not os.environ.get("NEBIUS_API_KEY"):
        print("先在这个终端里设好 NEBIUS_API_KEY 再跑")
        return 1
    batches = BATCHES + ([ROBOVAD] if "--with-robovad" in argv else [])
    for path, camera, name in batches:
        print(f"== {name}")
        main(["check", path, "--judges", "robometer,topreward", "--camera", camera, "--summary", "--lang", "en",
              "--out", f"reports/{name}_en.html"] + (["--no-frames"] if name == "robovad" else []))
    print("== 本批小结的通过率（缓存里当前提示版本的全部条目）")
    return main(["summary-stats"])


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
