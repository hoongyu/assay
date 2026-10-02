"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from jianjin.checks import run_checks
from jianjin.ingest import load


def _cmd_check(args: argparse.Namespace) -> int:
    res = load(args.path)
    results = run_checks(res.episodes)
    tasks = {ep.id: ep.task for ep in res.episodes}
    counts: Counter[str] = Counter()
    for r in results:
        for f in r.findings:
            counts[f.cause.value] += 1
    print(f"{len(res.episodes)} 条示范，读取失败 {len(res.failures)} 个文件")
    for cause, n in counts.most_common():
        print(f"  {cause}：{n}")
    for name, why in res.failures:
        print(f"  读取失败 {name}：{why}")
    if args.json:
        rows = [
            {
                "episode_id": r.episode_id,
                "task": tasks[r.episode_id],
                "findings": [{**asdict(f), "cause": f.cause.value} for f in r.findings],
                "not_checked": r.not_checked,
            }
            for r in results
        ]
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细写到 {args.json}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from jianjin.eval.exam_a import format_table, load_config, run_exam_a

    args.out = args.out or f"results/exam_a_{args.split}.json"
    out = run_exam_a(load_config(args.config), args.data_root, args.split, seed=args.seed)
    print(format_table(out))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果写到 {args.out}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from jianjin.eval.exam_a import load_config
    from jianjin.eval.fetch import fetch

    for p in fetch(load_config(args.config), args.data_root, with_videos=args.videos):
        print(f"已下载到 {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jianjin", description="数采场示范数据的自动质检")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="对一个数据集做关节体检")
    c.add_argument("path")
    c.add_argument("--json", help="把每条示范的发现写成 JSON")
    c.set_defaults(func=_cmd_check)

    e = sub.add_parser("eval", help="跑考卷")
    e.add_argument("exam", choices=["a"])
    e.add_argument("--config", default="configs/exam_a.toml")
    e.add_argument("--data-root", default="data")
    e.add_argument("--split", choices=["dev", "test"], default="dev")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=None)
    e.set_defaults(func=_cmd_eval)

    f = sub.add_parser("fetch", help="按考卷配置下载公开数据集")
    f.add_argument("--config", default="configs/exam_a.toml")
    f.add_argument("--data-root", default="data")
    f.add_argument("--videos", action="store_true", help="连视频一起下")
    f.set_defaults(func=_cmd_fetch)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台管道默认不是 UTF-8
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
