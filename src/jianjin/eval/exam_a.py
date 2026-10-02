"""考卷 A：注入缺陷。按数据集切开发集和测试集（一个数据集一个任务），不按示范随机切。"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from jianjin.checks import CheckConfig, run_checks
from jianjin.eval.inject import INJECTORS
from jianjin.eval.metrics import score_kind
from jianjin.ingest import load

MIN_INJECT_S = 3.0  # 短于 3 秒的示范不注入，数量照实打印


@dataclass(frozen=True)
class DatasetSpec:
    repo: str
    split: str
    license: str


def load_config(path: str | Path) -> list[DatasetSpec]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return [DatasetSpec(d["repo"], d["split"], d["license"]) for d in data["dataset"]]


def local_dir(data_root: str | Path, repo: str) -> Path:
    return Path(data_root) / repo.replace("/", "__")


def run_exam_a(
    specs: list[DatasetSpec],
    data_root: str | Path,
    split: str,
    cfg: CheckConfig | None = None,
    seed: int = 0,
) -> dict:
    cfg = cfg or CheckConfig()
    episodes, datasets = [], []
    for s in specs:
        if s.split != split:
            continue
        res = load(local_dir(data_root, s.repo))
        episodes += res.episodes
        datasets.append(
            {
                "repo": s.repo,
                "license": s.license,
                "n_episodes": len(res.episodes),
                "tasks": sorted({e.task for e in res.episodes}),
                "robot_types": sorted({str(e.meta.get("robot_type")) for e in res.episodes}),
                "read_failures": res.failures,
            }
        )
    clean = run_checks(episodes, cfg)
    eligible = [i for i, e in enumerate(episodes) if e.duration >= MIN_INJECT_S]

    kinds = []
    for kind, inject in INJECTORS.items():
        rng = np.random.default_rng(seed)
        pairs = [inject(episodes[i], rng) for i in eligible]
        # 同任务比较拿原样示范当参照，不然整批都注入了缺陷，中位数本身就偏了
        injected = run_checks([p[0] for p in pairs], cfg, reference=episodes)
        kinds.append(score_kind(kind, [p[1] for p in pairs], injected, clean))

    return {
        "exam": "A（注入缺陷：在真机成功示范上人工造的失败，不是真实失败）",
        "split": split,
        "seed": seed,
        "config": asdict(cfg),
        "datasets": datasets,
        "n_episodes": len(episodes),
        "n_eligible": len(eligible),
        "kinds": kinds,
    }


def _pct(k: int, n: int, ci: tuple[float, float]) -> str:
    if n == 0:
        return "—"
    return f"{k}/{n} = {k / n:.0%}（95% 区间 {ci[0]:.0%} 到 {ci[1]:.0%}）"


def format_table(out: dict) -> str:
    lines = [
        f"考卷 {out['exam']}",
        f"切分：{out['split']}；随机种子 {out['seed']}",
        f"示范 {out['n_episodes']} 条，其中 {out['n_eligible']} 条长于 {MIN_INJECT_S:.0f} 秒、用来注入",
    ]
    for d in out["datasets"]:
        lines.append(
            f"  {d['repo']}（{d['license']}）：{d['n_episodes']} 条；机型 {', '.join(d['robot_types'])}；"
            f"任务 {'; '.join(d['tasks'])}；读取失败 {len(d['read_failures'])} 个文件"
        )
    for k in out["kinds"]:
        lines += [
            f"[{k['kind']} → {k['cause']}]",
            f"  检出：{_pct(k['detected'], k['n_injected'], k['detect_ci'])}",
            f"  时间点落在注入区间：{_pct(k['localized'], k['n_injected'], k['localize_ci'])}",
            f"  原样示范上也报了这一类：{_pct(k['clean_flagged'], k['n_clean'], k['clean_flag_ci'])}"
            "（不全是误报：原示范本身可能就有这个问题）",
        ]
    return "\n".join(lines)
