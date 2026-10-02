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
    if args.exam == "a":
        from jianjin.eval.exam_a import format_table, load_config, run_exam_a

        split = args.split or "dev"
        out = run_exam_a(load_config(args.config), args.data_root, split, seed=args.seed)
    else:
        from jianjin.eval.exam_b import format_table, run_exam_b

        split = args.split or "validation"
        out = run_exam_b(args.robovad_root, split, args.scorer, cache_dir=args.cache, camera=args.camera)
    print(format_table(out))
    path = Path(args.out or f"results/exam_{args.exam}_{split}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果写到 {path}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    if args.what == "robovad":
        from jianjin.eval.robovad_fetch import fetch_robovad

        print(f"已取到 {fetch_robovad(args.data_root, videos=args.videos)}")
        return 0
    from jianjin.eval.exam_a import load_config
    from jianjin.eval.fetch import fetch

    for p in fetch(load_config(args.config), args.data_root, with_videos=args.videos):
        print(f"已下载到 {p}")
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    from jianjin.checks import NotApplicable
    from jianjin.judges import JUDGES
    from jianjin.judges.cache import judge_episode

    judge = JUDGES[args.judge]()
    if args.what == "robovad":
        from jianjin.eval.exam_b import split_names
        from jianjin.ingest.robovad import read_robovad

        names = [n for s in args.split.split(",") for n in split_names(Path(args.robovad_root), s)]
        episodes = read_robovad(args.robovad_root, names=names).episodes
    else:
        episodes = load(args.what).episodes
    if args.limit:
        episodes = episodes[: args.limit]
    done = skipped = 0
    for i, ep in enumerate(episodes, 1):
        try:
            judge_episode(judge, ep, args.camera, Path(args.cache), sample_fps=args.sample_fps)
            done += 1
        except NotApplicable as exc:
            skipped += 1
            print(f"  跳过 {ep.id}：{exc}")
        if i % 10 == 0 or i == len(episodes):
            print(f"{i}/{len(episodes)}，完成 {done}，跳过 {skipped}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jianjin", description="数采场示范数据的自动质检")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="对一个数据集做关节体检")
    c.add_argument("path")
    c.add_argument("--json", help="把每条示范的发现写成 JSON")
    c.set_defaults(func=_cmd_check)

    e = sub.add_parser("eval", help="跑考卷")
    e.add_argument("exam", choices=["a", "b"])
    e.add_argument("--config", default="configs/exam_a.toml")
    e.add_argument("--data-root", default="data")
    e.add_argument("--robovad-root", default="data/robovad")
    e.add_argument("--split", choices=["dev", "test", "train", "validation"], default=None)
    e.add_argument("--scorer", default="tracking")
    e.add_argument("--cache", default="cache")
    e.add_argument("--camera", default=None)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=None)
    e.set_defaults(func=_cmd_eval)

    f = sub.add_parser("fetch", help="下载考卷数据")
    f.add_argument("what", nargs="?", choices=["exam_a", "robovad"], default="exam_a")
    f.add_argument("--config", default="configs/exam_a.toml")
    f.add_argument("--data-root", default="data")
    f.add_argument("--videos", action="store_true", help="连视频一起下")
    f.set_defaults(func=_cmd_fetch)

    j = sub.add_parser("judge", help="在有 GPU 的机器上跑视频评判，结果写进缓存")
    j.add_argument("what", help="robovad，或者一个数据集路径")
    j.add_argument("--judge", default="robometer")
    j.add_argument("--robovad-root", default="data/robovad")
    j.add_argument("--split", default="train", help="robovad 的划分，逗号分隔，如 train,test")
    j.add_argument("--camera", default=None)
    j.add_argument("--cache", default="cache")
    j.add_argument("--sample-fps", type=float, default=2.0)
    j.add_argument("--limit", type=int, default=0)
    j.set_defaults(func=_cmd_judge)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台管道默认不是 UTF-8
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
