"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from jianjin.ingest import load


def _cmd_check(args: argparse.Namespace) -> int:
    from datetime import datetime

    from jianjin.diagnose import Label, diagnose_batch
    from jianjin.judges.cache import load_curves
    from jianjin.report import render
    from jianjin.types import source_name

    res = load(args.path)
    judges = [j for j in (args.judges or "").split(",") if j]
    curves = {j: load_curves(res.episodes, Path(args.cache), j, args.camera) for j in judges}
    verdicts = diagnose_batch(res.episodes, curves)
    labels = Counter(v.label for v in verdicts)
    print(f"{len(res.episodes)} 条示范：合格 {labels[Label.PASS]}，存疑 {labels[Label.DOUBT]}，"
          f"作废 {labels[Label.REJECT]}；读取失败 {len(res.failures)} 个文件")
    causes = Counter(f.cause.value for v in verdicts for f in v.causes)
    for cause, n in causes.most_common():
        print(f"  {cause}：{n}")
    for j in judges:
        print(f"  {source_name(j)}：{len(curves[j])}/{len(res.episodes)} 条有缓存")
    for name, why in res.failures:
        print(f"  读取失败 {name}：{why}")
    if args.json:
        tasks = {ep.id: ep.task for ep in res.episodes}
        rows = [
            {
                "episode_id": v.episode_id,
                "task": tasks[v.episode_id],
                "label": v.label.value,
                "reasons": v.reasons,
                "findings": [{**asdict(f), "cause": f.cause.value} for f in v.causes],
                "not_checked": v.not_checked,
            }
            for v in verdicts
        ]
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细写到 {args.json}")
    if args.out:
        shots = None
        if not args.no_frames:
            from jianjin.keyframes import shots_for

            shots, missing = shots_for(verdicts, res.episodes, args.camera)
            print(f"  画面：{len(shots)} 条存疑或作废的示范配了关键帧"
                  + (f"，{missing} 条没有本地视频或读不出画面" if missing else ""))
        summary = None
        if args.summary:
            import os

            from jianjin.summary import batch_facts, summarize, token_factory_chat

            key = os.environ.get("NEBIUS_API_KEY")
            if not key:
                print("  本批小结：没设 NEBIUS_API_KEY，用模板文字")
            summary = summarize(batch_facts(verdicts, res.episodes, Path(args.path).name),
                                chat=token_factory_chat(key) if key else None, model=args.summary_model,
                                cache_dir=args.cache)
            for lang, src in summary.source.items():
                print(f"  本批小结（{lang}）：{'模型写的' if src == 'model' else '模板'}"
                      + (f"；{'；'.join(summary.problems[lang])}" if summary.problems.get(lang) else ""))
        html = render(verdicts, res.episodes, title=Path(args.path).name, judges=judges,
                      failures=res.failures, generated=datetime.now().strftime("%Y-%m-%d %H:%M"), shots=shots,
                      summary=summary)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(html, encoding="utf-8")
        print(f"报告写到 {args.out}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    # 读视频评判缓存的考卷，不是 Robometer 的成绩文件名带上评判名，免得盖掉 Robometer 的
    suffix = f"_{args.judge}" if args.exam in ("a-swap", "b-episode") and args.judge != "robometer" else ""
    if args.exam == "c":
        from jianjin.eval.exam_c import format_exam_c, run_exam_c, success_sets

        split = args.split or "dev"
        dev = None
        if args.dev_result:
            dev = json.loads(Path(args.dev_result).read_text(encoding="utf-8"))
        elif split != "dev":
            raise SystemExit("test 要用 dev 选定的：加 --dev-result results/exam_c_dev.json")
        succ = success_sets(args.config, args.data_root, args.robovad_root, split)
        out = run_exam_c(args.robomind_root, split, succ, split_file=args.split_file, dev_result=dev)
        text = format_exam_c(out)
    elif args.exam == "c-regrasp":
        from jianjin.eval.exam_c import format_regrasp_c, run_regrasp_c

        split = "all"
        out = run_regrasp_c(args.robomind_root)
        text = format_regrasp_c(out)
    elif args.exam == "b-regrasp":
        from jianjin.eval.exam_b_regrasp import format_regrasp_b, run_regrasp_b

        split = args.split or "train"
        source = margin = None
        if args.dev_result:
            dev = json.loads(Path(args.dev_result).read_text(encoding="utf-8"))
            source, margin = dev["source"], dev["margin"]
        elif split != "train":
            raise SystemExit("test 要用 train 选定的：加 --dev-result results/exam_b_regrasp_train.json")
        out = run_regrasp_b(args.robovad_root, split, source=source, margin=margin)
        text = format_regrasp_b(out)
    elif args.exam == "b-episode":
        from jianjin.eval.exam_b_episode import format_episode_b, run_episode_b

        split = args.split or "train"
        threshold = None
        if args.dev_result:
            threshold = json.loads(Path(args.dev_result).read_text(encoding="utf-8"))["threshold"]
        elif split != "train":
            raise SystemExit(f"test 要用 train 定的阈值：加 --dev-result results/exam_b_episode_train{suffix}.json")
        out = run_episode_b(args.robovad_root, split, args.cache, judge=args.judge,
                            camera=args.camera or "left", threshold=threshold)
        text = format_episode_b(out)
    elif args.exam == "a-swap":
        from jianjin.eval.exam_a import load_config
        from jianjin.eval.task_swap import format_swap, run_swap

        split = args.split or "dev"
        score = threshold = None
        if args.dev_result:
            dev = json.loads(Path(args.dev_result).read_text(encoding="utf-8"))
            score, threshold = dev["score"], dev["threshold"]
        elif split != "dev":
            raise SystemExit(f"test 要用 dev 定的分数和阈值：加 --dev-result results/exam_a_swap_dev{suffix}.json")
        out = run_swap(load_config(args.config), args.data_root, split, args.cache,
                       judge=args.judge, score=score, threshold=threshold, seed=args.seed)
        text = format_swap(out)
    elif args.exam == "a":
        from jianjin.eval.exam_a import format_table, load_config, run_exam_a

        split = args.split or "dev"
        out = run_exam_a(load_config(args.config), args.data_root, split, seed=args.seed)
        text = format_table(out)
    else:
        from jianjin.eval.exam_b import format_table, run_exam_b

        split = args.split or "validation"
        out = run_exam_b(args.robovad_root, split, args.scorer, cache_dir=args.cache, camera=args.camera)
        text = format_table(out)
    print(text)
    path = Path(args.out or f"results/exam_{args.exam.replace('-', '_')}_{split}{suffix}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果写到 {path}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    if args.what == "robomind":
        from jianjin.eval.robomind_fetch import fetch_robomind_failure

        stat = fetch_robomind_failure(args.data_root, workers=args.workers)
        print("、".join(f"{k} {v} 条" for k, v in stat.items()))
        return 0
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
    jobs: list[tuple] = []  # (示范, 相机)
    if args.what == "exam_a":
        from jianjin.eval.exam_a import load_config
        from jianjin.eval.task_swap import swap_pairs

        for split in args.split.split(","):
            for orig, swapped, cam in swap_pairs(load_config(args.config), args.data_root, split, seed=args.seed):
                jobs += [(orig, cam), (swapped, cam)]
    elif args.what == "robovad":
        from jianjin.eval.exam_b import split_names
        from jianjin.ingest.robovad import read_robovad

        names = [n for s in args.split.split(",") for n in split_names(Path(args.robovad_root), s)]
        jobs = [(ep, args.camera) for ep in read_robovad(args.robovad_root, names=names).episodes]
    else:
        jobs = [(ep, args.camera) for ep in load(args.what).episodes]
    if args.limit:
        jobs = jobs[: args.limit]
    done = skipped = 0
    for i, (ep, cam) in enumerate(jobs, 1):
        try:
            judge_episode(judge, ep, cam, Path(args.cache), sample_fps=args.sample_fps, max_side=args.max_side)
            done += 1
        except NotApplicable as exc:
            skipped += 1
            print(f"  跳过 {ep.id}：{exc}")
        if i % 10 == 0 or i == len(jobs):
            print(f"{i}/{len(jobs)}，完成 {done}，跳过 {skipped}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jianjin", description="数采场示范数据的自动质检")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="对一个数据集做关节体检")
    c.add_argument("path")
    c.add_argument("--json", help="把每条示范的判定和发现写成 JSON")
    c.add_argument("--out", help="把报告写成一个 HTML 文件")
    c.add_argument("--judges", default="", help="读哪些视频评判的缓存，逗号分隔，如 robometer,topreward")
    c.add_argument("--cache", default="cache")
    c.add_argument("--camera", default=None, help="视频评判用的那一路相机；不写就用第一路")
    c.add_argument("--no-frames", action="store_true", help="报告里不配关键帧（默认有本地视频就配）")
    c.add_argument("--summary", action="store_true",
                   help="报告顶部加本批小结：Nemotron（Nebius Token Factory）写文字，数字由拣金填；密钥读环境变量 "
                        "NEBIUS_API_KEY，没设就用模板")
    c.add_argument("--summary-model", default="nvidia/Nemotron-3_5-Lightning", help="本批小结用哪个模型")
    c.set_defaults(func=_cmd_check)

    e = sub.add_parser("eval", help="跑考卷")
    e.add_argument("exam", choices=["a", "a-swap", "b", "b-episode", "b-regrasp", "c", "c-regrasp"])
    e.add_argument("--config", default="configs/exam_a.toml")
    e.add_argument("--data-root", default="data")
    e.add_argument("--robovad-root", default="data/robovad")
    e.add_argument("--robomind-root", default="data/robomind_failure", help="考卷 C：jianjin fetch robomind 存的地方")
    e.add_argument("--split-file", default="configs/exam_c_split.json", help="考卷 C 的任务切分")
    e.add_argument("--split", choices=["dev", "test", "train", "validation"], default=None)
    e.add_argument("--scorer", default="tracking")
    e.add_argument("--cache", default="cache")
    e.add_argument("--camera", default=None)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=None)
    e.add_argument("--judge", default="robometer", help="a-swap、b-episode 读哪个视频评判的缓存")
    e.add_argument("--dev-result", default=None, help="沿用这个结果里的选择：a-swap、c 用 dev 的，b-episode、b-regrasp 用 train 的")
    e.set_defaults(func=_cmd_eval)

    f = sub.add_parser("fetch", help="下载考卷数据")
    f.add_argument("what", nargs="?", choices=["exam_a", "robovad", "robomind"], default="exam_a",
                   help="robomind：RoboMIND 公开的失败示范，只读关节（约 2 GB 流量，存下 200 多 MB）")
    f.add_argument("--config", default="configs/exam_a.toml")
    f.add_argument("--data-root", default="data")
    f.add_argument("--videos", action="store_true", help="连视频一起下")
    f.add_argument("--workers", type=int, default=4, help="robomind 同时下几条")
    f.set_defaults(func=_cmd_fetch)

    j = sub.add_parser("judge", help="在有 GPU 的机器上跑视频评判，结果写进缓存")
    j.add_argument("what", help="robovad、exam_a（考卷 A 换描述的原样和换后各一份），或者一个数据集路径")
    j.add_argument("--config", default="configs/exam_a.toml")
    j.add_argument("--data-root", default="data")
    j.add_argument("--seed", type=int, default=0)
    j.add_argument("--judge", default="robometer")
    j.add_argument("--robovad-root", default="data/robovad")
    j.add_argument("--split", default="train", help="划分，逗号分隔，如 train,test 或 dev,test")
    j.add_argument("--camera", default=None)
    j.add_argument("--cache", default="cache")
    j.add_argument("--sample-fps", type=float, default=2.0)
    j.add_argument("--max-side", type=int, default=448)
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
