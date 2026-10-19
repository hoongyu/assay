"""Summarize the downstream experiment from out/results/square_*.json.

Pre-registered (notebook cell 9): per condition, success at the last evaluation and the best evaluation,
mean and population sd over seeds. Added here: the last evaluation pooled over seeds (n rollouts in total)
with a Wilson 95% interval, used to say whether a difference is larger than evaluation noise.
    python summarize_ds.py [results_dir] [out.json]
"""

import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "out" / "results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
CONDS = ["exp_all", "exp_assay", "exp_random", "exp_oracle"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


runs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(str(RES / "square_exp_*.json")))]
summary = {}
for cond in CONDS:
    rs = sorted((r for r in runs if r["name"].startswith(f"square_{cond}_")), key=lambda r: r["seed"])
    if not rs:
        continue
    last_ep = [max(map(int, r["success_by_epoch"])) for r in rs]
    last = [r["success_by_epoch"][str(e)] for r, e in zip(rs, last_ep)]
    best = [max(r["success_by_epoch"].values()) for r in rs]
    n_each = [r["rollouts_per_eval"] for r in rs]
    k = sum(round(s * n) for s, n in zip(last, n_each))
    n = sum(n_each)
    lo, hi = wilson(k, n)
    summary[cond] = {
        "seeds": [r["seed"] for r in rs], "n_demos": [r.get("n_demos") for r in rs],
        "last_epoch": last_ep, "last": last, "best": best,
        "last_mean": round(st.mean(last), 3), "last_sd": round(st.pstdev(last), 3),
        "best_mean": round(st.mean(best), 3), "best_sd": round(st.pstdev(best), 3),
        "last_pooled": f"{k}/{n}", "last_pooled_rate": round(k / n, 3), "last_wilson95": [round(lo, 3), round(hi, 3)],
        "curve_mean": {e: round(st.mean(r["success_by_epoch"].get(e, float("nan")) for r in rs), 3)
                       for e in sorted(rs[0]["success_by_epoch"], key=int)},
        "gpu_minutes": round(sum(r["wall_seconds"] for r in rs) / 60, 1),
    }
for cond, s in summary.items():
    print(f"{cond:11s} demos {s['n_demos']}  last {s['last']} mean {s['last_mean']:.2f}±{s['last_sd']:.2f}  "
          f"pooled {s['last_pooled']} = {s['last_pooled_rate']:.2f} [{s['last_wilson95'][0]:.2f}, {s['last_wilson95'][1]:.2f}]  "
          f"best mean {s['best_mean']:.2f}±{s['best_sd']:.2f}  curve {s['curve_mean']}")
if OUT:
    OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("saved", OUT)
