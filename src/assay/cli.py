"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from assay.ingest import load


def _cmd_check(args: argparse.Namespace) -> int:
    from datetime import datetime

    from assay.diagnose import LABEL_EN, Label, diagnose_batch
    from assay.judges.cache import load_curves
    from assay.report import render
    from assay.types import CAUSE_EN, en, source_name

    english = args.lang == "en"

    def say(zh: str, english_text: str) -> None:
        print(english_text if english else zh)

    res = load(args.path)
    judges = [j for j in (args.judges or "").split(",") if j]
    curves = {j: load_curves(res.episodes, Path(args.cache), j, args.camera) for j in judges}
    verdicts = diagnose_batch(res.episodes, curves)
    labels = Counter(v.label for v in verdicts)
    n, p, d, r, k = (len(res.episodes), labels[Label.PASS], labels[Label.DOUBT], labels[Label.REJECT],
                     len(res.failures))
    say(f"{n} 条示范：合格 {p}，存疑 {d}，作废 {r}；读取失败 {k} 个文件",
        f"{n} episodes: {LABEL_EN[Label.PASS]} {p}, {LABEL_EN[Label.DOUBT]} {d}, {LABEL_EN[Label.REJECT]} {r}; "
        f"{k} files could not be read")
    causes = Counter(f.cause for v in verdicts for f in v.causes)
    for cause, c in causes.most_common():
        say(f"  {cause.value}：{c}", f"  {CAUSE_EN[cause]}: {c}")
    for j in judges:
        say(f"  {source_name(j)}：{len(curves[j])}/{n} 条有缓存", f"  {source_name(j, 'en')}: {len(curves[j])}/{n} cached")
    for name, why in res.failures:
        say(f"  读取失败 {name}：{why}", f"  Could not read {name}: {en(why)}")
    if args.review:
        import os

        from assay.review_run import CachedChat, run_reviews

        chat = CachedChat(os.environ.get("NEBIUS_API_KEY"), args.cache, read_cache=not args.review_no_cache)
        say(f"视频复核：{args.review_model}（Nebius Token Factory），输入{'整段视频' if args.review_input == 'video' else '关键帧'}",
            f"Video review: {args.review_model} on Nebius Token Factory, input: {args.review_input}")
        verdict_zh = {"confirms": "支持规则", "clears": "和规则相反，要人看", "unclear": "没给出能用的回答"}
        verdict_en = {"confirms": "agrees with the rule", "clears": "disagrees with the rule, a person should look",
                      "unclear": "no usable answer"}

        def show(r) -> None:
            short = r.episode_id.rsplit("/", 1)[-1]  # 终端字号调大录屏时不折行
            say(f"  {short}  {r.check} → {r.answer}（{verdict_zh[r.outcome]}）",
                f"  {short}  {r.check} -> {r.answer} ({verdict_en[r.outcome]})")

        skipped: list[str] = []
        try:
            verdicts, skipped = run_reviews(verdicts, res.episodes, chat, camera=args.camera, mode=args.review_input,
                                            model=args.review_model, on_result=show)
        except (RuntimeError, OSError) as exc:  # 没设密钥又没有缓存、网络错误
            say(f"  视频复核没跑完：{exc}", f"  Video review stopped: {exc}")
        if skipped:
            one = len(skipped) == 1
            say(f"  {len(skipped)} 条没有本地视频，没复核",
                f"  {len(skipped)} {'episode has' if one else 'episodes have'} no local video and "
                f"{'was' if one else 'were'} not reviewed")
        fresh = [u for u in chat.usage if not u.get("cached")]
        tokens = sum(u.get("prompt_tokens", 0) for u in fresh)
        say(f"  新调用 {len(fresh)} 次，输入 {tokens} 个 token；{len(chat.usage) - len(fresh)} 次用了缓存",
            f"  {len(fresh)} new calls, {tokens} prompt tokens; {len(chat.usage) - len(fresh)} answered from cache")
    if args.json:
        tasks = {ep.id: ep.task for ep in res.episodes}
        rows = [
            {
                "episode_id": v.episode_id,
                "task": tasks[v.episode_id],
                "label": v.label.value,
                "reasons": v.reasons,
                "reasons_en": [en(x) for x in v.reasons],
                "findings": [{**asdict(f), "cause": f.cause.value, "evidence_en": en(f.evidence)} for f in v.causes],
                "not_checked": v.not_checked,
                "not_checked_en": {key: en(why) for key, why in v.not_checked.items()},
                "reviews": [{k: getattr(r, k) for k in ("model", "check", "question", "answer", "outcome")}
                            for r in v.reviews],
            }
            for v in verdicts
        ]
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        say(f"明细写到 {args.json}", f"Details written to {args.json}")
    if args.out:
        shots = None
        if not args.no_frames:
            from assay.keyframes import shots_for

            shots, missing = shots_for(verdicts, res.episodes, args.camera)
            say(f"  画面：{len(shots)} 条存疑或作废的示范配了关键帧" + (f"，{missing} 条没有本地视频或读不出画面" if missing else ""),
                f"  Frames: keyframes for {len(shots)} DOUBT or REJECT episodes"
                + (f"; {missing} {'has' if missing == 1 else 'have'} no local video or no readable frames" if missing else ""))
        summary = None
        if args.summary:
            import os

            from assay.summary import batch_facts, summarize, token_factory_chat

            key = os.environ.get("NEBIUS_API_KEY")
            summary = summarize(batch_facts(verdicts, res.episodes, Path(args.path).name),
                                chat=token_factory_chat(key) if key else None, model=args.summary_model,
                                cache_dir=args.cache)
            if not key and "template" in summary.source.values():
                say("  本批小结：没设 NEBIUS_API_KEY，缓存里也没有这批的，用模板文字",
                    "  Batch summary: NEBIUS_API_KEY is not set and this batch is not cached, so the template is used")
            for lang, src in summary.source.items():
                problems = summary.problems.get(lang) or []
                say(f"  本批小结（{lang}）：{'模型写的' if src == 'model' else '模板'}"
                    + (f"；{'；'.join(problems)}" if problems else ""),
                    f"  Batch summary ({lang}): {'written by the model' if src == 'model' else 'template'}"
                    + (f"; {'; '.join(en(x) for x in problems)}" if problems else ""))
        html = render(verdicts, res.episodes, title=Path(args.path).name, judges=judges,
                      failures=res.failures, generated=datetime.now().strftime("%Y-%m-%d %H:%M"), shots=shots,
                      summary=summary, lang=args.lang)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(html, encoding="utf-8")
        say(f"报告写到 {args.out}", f"Report written to {args.out}")
    return 0


def _cmd_summary_stats(args: argparse.Namespace) -> int:
    from assay.summary import PROMPT_VERSION, cache_stats

    stats = cache_stats(args.cache)
    if not stats:
        print(f"{args.cache} 里没有第 {PROMPT_VERSION} 版提示的小结")
    for lang, s in stats.items():
        print(f"{lang}：第一次就过 {s['first_try']}，重试以后过 {s['after_retry']}，退回模板 {s['template']}"
              f"（共 {sum(s.values())} 条）")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    # 读视频评判缓存的考卷，不是 Robometer 的成绩文件名带上评判名，免得盖掉 Robometer 的
    suffix = f"_{args.judge}" if args.exam in ("a-swap", "b-episode") and args.judge != "robometer" else ""
    if args.exam == "c":
        from assay.eval.exam_c import format_exam_c, run_exam_c, success_sets

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
        from assay.eval.exam_c import format_regrasp_c, run_regrasp_c

        split = "all"
        out = run_regrasp_c(args.robomind_root)
        text = format_regrasp_c(out)
    elif args.exam == "b-regrasp":
        from assay.eval.exam_b_regrasp import format_regrasp_b, run_regrasp_b

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
        from assay.eval.exam_b_episode import format_episode_b, run_episode_b

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
        from assay.eval.exam_a import load_config
        from assay.eval.task_swap import format_swap, run_swap

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
        from assay.eval.exam_a import format_table, load_config, run_exam_a

        split = args.split or "dev"
        out = run_exam_a(load_config(args.config), args.data_root, split, seed=args.seed)
        text = format_table(out)
    else:
        from assay.eval.exam_b import format_table, run_exam_b

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
        from assay.eval.robomind_fetch import fetch_robomind_failure

        stat = fetch_robomind_failure(args.data_root, workers=args.workers)
        print("、".join(f"{k} {v} 条" for k, v in stat.items()))
        return 0
    if args.what == "robovad":
        from assay.eval.robovad_fetch import fetch_robovad

        print(f"已取到 {fetch_robovad(args.data_root, videos=args.videos)}")
        return 0
    from assay.eval.exam_a import load_config
    from assay.eval.fetch import fetch

    for p in fetch(load_config(args.config), args.data_root, with_videos=args.videos):
        print(f"已下载到 {p}")
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    from assay.checks import NotApplicable
    from assay.judges import JUDGES
    from assay.judges.cache import judge_episode

    judge = JUDGES[args.judge]()
    jobs: list[tuple] = []  # (示范, 相机)
    if args.what == "exam_a":
        from assay.eval.exam_a import load_config
        from assay.eval.task_swap import swap_pairs

        for split in args.split.split(","):
            for orig, swapped, cam in swap_pairs(load_config(args.config), args.data_root, split, seed=args.seed):
                jobs += [(orig, cam), (swapped, cam)]
    elif args.what == "robovad":
        from assay.eval.exam_b import split_names
        from assay.ingest.robovad import read_robovad

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
    p = argparse.ArgumentParser(prog="assay", description="数采场示范数据的自动质检")
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
                   help="报告顶部加本批小结：Nemotron（Nebius Token Factory）写文字，数字由甄数填；密钥读环境变量 "
                        "NEBIUS_API_KEY，没设就用模板")
    c.add_argument("--summary-model", default="nvidia/Nemotron-3_5-Lightning", help="本批小结用哪个模型")
    c.add_argument("--review", action="store_true",
                   help="存疑和作废的示范各问 Nemotron 3.5 Super VL（Nebius Token Factory）一个由规则生成的问题，"
                        "结论写进报告，不改判定；要本地视频，密钥读环境变量 NEBIUS_API_KEY，回答缓存在 --cache 下")
    c.add_argument("--review-input", choices=["video", "frames"], default="video", help="复核时发整段视频还是 8 张关键帧")
    c.add_argument("--review-model", default="nvidia/Nemotron-3.5-Super-VL", help="视频复核用哪个模型")
    c.add_argument("--review-no-cache", action="store_true", help="视频复核不读缓存，每条都现场调一次（回答照样写进缓存）")
    c.add_argument("--lang", choices=["zh", "en"], default="zh", help="报告和终端打印用中文还是英文")
    c.set_defaults(func=_cmd_check)

    st = sub.add_parser("summary-stats", help="数缓存里的本批小结：第一次就过、重试以后过、退回模板各几条")
    st.add_argument("--cache", default="cache")
    st.set_defaults(func=_cmd_summary_stats)

    e = sub.add_parser("eval", help="跑考卷")
    e.add_argument("exam", choices=["a", "a-swap", "b", "b-episode", "b-regrasp", "c", "c-regrasp"])
    e.add_argument("--config", default="configs/exam_a.toml")
    e.add_argument("--data-root", default="data")
    e.add_argument("--robovad-root", default="data/robovad")
    e.add_argument("--robomind-root", default="data/robomind_failure", help="考卷 C：assay fetch robomind 存的地方")
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
