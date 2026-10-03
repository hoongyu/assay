# 拣金第三周：判定和网页报告 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把关节规则和视频评判的发现合成「合格、存疑、作废」三种判定，生成离线单文件网页报告，`jianjin check <数据集> --out report.html` 一条命令出结果。

**Architecture:** `checks/video.py` 从缓存的进度曲线里找问题（指令不符、进度回落），每个视频模型算一个独立来源；`diagnose.py` 按设计文档 4.3 合成判定；`report.py` 把判定渲染成一个不引用外部资源的 HTML；`cli.py` 的 `check` 命令串起来。视频结果只从缓存读，本机不需要 GPU。

**Tech Stack:** 只用现有依赖（numpy、pyarrow、h5py、av）；报告是手写 HTML、CSS 和几行原生 JS，不引外部库。

**范围：** 对应设计文档第 3 节必做的第 4、5 项（诊断、离线报告）。TOPReward 的阈值等它的缓存回来、在 dev 上定好后，往 `WRONG_TASK`、`PROGRESS_DROP` 两张表里各加一行，不在本计划里。关键帧缩略图、夹爪规则不做。

**事实依据（2026-10-03 核）：**

- 「指令不符」的阈值：考卷 A 换描述 dev 上选出 `rel_peak`，阈值 1.5477136091635564（`results/exam_a_swap_dev.json`），test 检出 184/201、原样报出 10/201。
- 「进度回落」的探针（RoboVAD train，367 条，60 条有异常标注）：进度从此前最高点往下掉的最大值，按示范算 AUC 0.797；正常示范的 95 分位约 0.344，这时有异常的检出 17/60。
- RoboVAD 的 `annotations.json` 覆盖 train、validation、test 三个划分（test 629 条里 387 条有异常标注）；`read_robovad` 的时间戳已经换算成从 0 开始。
- `load()` 现在认不出 RoboVAD 的目录。
- 设计文档 4.3 的「同一处问题」只按时间重叠定义；但「指令不符」「多余动作」这类发现覆盖整条示范，和什么都重叠。本计划改成：两个不同来源，时间重叠，而且原因相同，或者两条都只占示范的一段。改动写回设计文档。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `src/jianjin/types.py` | `Finding` 加 `source`；`SOURCE_NAMES`、`source_name` |
| `src/jianjin/diagnose.py` | `Label`、`Verdict`、`overlaps`、`label_of`、`diagnose`、`diagnose_batch` |
| `src/jianjin/checks/video.py` | 视频规则：`BASE_SCORES`、`relative`、`wrong_task`、`drawdown`、`progress_drop`、`run_video_checks`，以及阈值表 |
| `src/jianjin/eval/task_swap.py` | 改成用 `checks/video.py` 的分数和换算，结果不变 |
| `src/jianjin/eval/metrics.py` | `overlaps` 改从 `diagnose` 引 |
| `src/jianjin/eval/exam_b_episode.py` | 考卷 B 按示范计：有没有异常 vs 进度回落规则 |
| `src/jianjin/judges/cache.py` | `load_curves`：一批示范的曲线 |
| `src/jianjin/report.py` | 离线单文件网页报告 |
| `src/jianjin/ingest/__init__.py` | `load()` 认出 RoboVAD |
| `src/jianjin/cli.py` | `check` 出判定和报告；`eval b-episode` |
| `tests/test_diagnose.py`、`tests/test_video_checks.py`、`tests/test_exam_b_episode.py`、`tests/test_report.py` | 新测试 |
| `tests/helpers.py` | `write_cached_curve` |
| `tests/snapshots/report_small.html` | 报告快照 |

---

### Task 1：判定核心

**Files:**
- Modify: `src/jianjin/types.py`
- Create: `src/jianjin/diagnose.py`
- Modify: `src/jianjin/eval/metrics.py:21-22`
- Test: `tests/test_diagnose.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_diagnose.py`：

```python
from jianjin.diagnose import Label, diagnose, label_of, overlaps
from jianjin.judges.base import Curve
from jianjin.types import Cause, Finding

DUR = 10.0


def f(cause, t0, t1, source="joints", hard=False, check="x"):
    return Finding(check, cause, t0, t1, 1.0, f"{cause.value} {t0}-{t1}", hard=hard, source=source)


def test_overlaps():
    assert overlaps((1, 3), (2, 4)) and not overlaps((1, 2), (2.5, 4))


def test_nothing_found_is_pass():
    assert label_of([], DUR) == (Label.PASS, [])


def test_hard_rule_rejects_alone():
    label, reasons = label_of([f(Cause.DATA_ERROR, 0, 0.5, hard=True)], DUR)
    assert label == Label.REJECT and reasons[0].startswith("客观硬规则")


def test_one_soft_signal_is_doubt():
    label, reasons = label_of([f(Cause.LONG_PAUSE, 2, 6)], DUR)
    assert label == Label.DOUBT and "只有关节" in reasons[0]


def test_same_source_twice_is_still_doubt():
    assert label_of([f(Cause.LONG_PAUSE, 2, 6), f(Cause.LONG_PAUSE, 5, 8)], DUR)[0] == Label.DOUBT


def test_two_sources_localized_and_overlapping_reject():
    label, reasons = label_of([f(Cause.LONG_PAUSE, 2, 6), f(Cause.DROP, 5, 7, source="robometer")], DUR)
    assert label == Label.REJECT and "Robometer" in reasons[0] and "第 5.0 到 6.0 秒" in reasons[0]


def test_two_sources_disjoint_is_doubt():
    label, reasons = label_of([f(Cause.LONG_PAUSE, 2, 4), f(Cause.DROP, 6, 7, source="robometer")], DUR)
    assert label == Label.DOUBT and "对不上" in reasons[0]


def test_whole_episode_finding_needs_same_cause():
    whole = f(Cause.WRONG_TASK, 0, DUR, source="robometer")
    assert label_of([whole, f(Cause.LONG_PAUSE, 2, 6)], DUR)[0] == Label.DOUBT
    assert label_of([whole, f(Cause.WRONG_TASK, 0, DUR, source="topreward")], DUR)[0] == Label.REJECT


def test_diagnose_sorts_causes_and_keeps_extras():
    late, early = f(Cause.LONG_PAUSE, 6, 8), f(Cause.LONG_PAUSE, 1, 3)
    c = Curve("robometer", "v", [0.0, 10.0], [0.0, 1.0])
    v = diagnose("e1", DUR, [late, early], {"end_pose": "同任务只有 3 条"}, c)
    assert v.episode_id == "e1" and v.label == Label.DOUBT
    assert v.causes == [early, late] and v.not_checked == {"end_pose": "同任务只有 3 条"} and v.progress is c
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `uv run pytest -q tests/test_diagnose.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'jianjin.diagnose'`

- [ ] **Step 3: `Finding` 加来源**

`src/jianjin/types.py`，`Finding` 的 `hard` 后面加一个字段，类末尾加来源名表：

```python
@dataclass(frozen=True)
class Finding:
    check: str
    cause: Cause
    t0: float
    t1: float
    score: float
    evidence: str
    hard: bool = False
    source: str = "joints"  # 哪个独立来源：关节规则记 joints，视频评判记模型名


SOURCE_NAMES = {"joints": "关节", "robometer": "Robometer", "topreward": "TOPReward"}


def source_name(source: str) -> str:
    return SOURCE_NAMES.get(source, source)
```

- [ ] **Step 4: 写 `diagnose.py`**

```python
"""诊断：把关节规则和视频评判的发现合成一个判定（设计文档 4.3）。

- 作废：客观硬规则触发；或者两个不同来源指向同一处问题。
  「同一处」是时间区间重叠，而且原因相同，或者两条都只占示范的一段。
  只看重叠不够：「指令不符」这类发现覆盖整条示范，和什么都重叠（2026-10-03 改）。
- 存疑：有发现，但够不上作废，要人看。
- 合格：什么都没报。没查的项目记在 not_checked，不当成通过。
错杀一条好数据比漏掉一条坏数据贵，所以判得偏保守。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations

from jianjin.judges.base import Curve
from jianjin.types import Finding, source_name


class Label(StrEnum):
    PASS = "合格"
    DOUBT = "存疑"
    REJECT = "作废"


@dataclass
class Verdict:
    episode_id: str
    label: Label
    reasons: list[str]
    causes: list[Finding]
    not_checked: dict[str, str] = field(default_factory=dict)
    progress: Curve | None = None


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _whole(f: Finding, duration: float) -> bool:
    return f.t0 <= 0.0 and f.t1 >= duration - 1e-9


def _agree(a: Finding, b: Finding, duration: float) -> bool:
    if a.source == b.source or not overlaps((a.t0, a.t1), (b.t0, b.t1)):
        return False
    return a.cause == b.cause or not (_whole(a, duration) or _whole(b, duration))


def label_of(findings: list[Finding], duration: float) -> tuple[Label, list[str]]:
    hard = [f for f in findings if f.hard]
    if hard:
        return Label.REJECT, [f"客观硬规则：{f.evidence}" for f in hard]
    agree = []
    for a, b in combinations(findings, 2):
        if not _agree(a, b, duration):
            continue
        when = f"第 {max(a.t0, b.t0):.1f} 到 {min(a.t1, b.t1):.1f} 秒"
        if a.cause == b.cause:
            agree.append(f"{source_name(a.source)}和{source_name(b.source)}都报了「{a.cause.value}」，{when}重叠")
        else:
            agree.append(f"{source_name(a.source)}报「{a.cause.value}」、{source_name(b.source)}报「{b.cause.value}」，"
                         f"{when}重叠，指向同一处问题")
    if agree:
        return Label.REJECT, agree
    if findings:
        names = "、".join(sorted({source_name(f.source) for f in findings}))
        if len({f.source for f in findings}) == 1:
            return Label.DOUBT, [f"只有{names}报了问题，没有第二个来源印证，要人看"]
        return Label.DOUBT, [f"{names}报的问题对不上（时间不重叠，或者原因不同），要人看"]
    return Label.PASS, []


def diagnose(episode_id: str, duration: float, findings: list[Finding],
             not_checked: dict[str, str] | None = None, progress: Curve | None = None) -> Verdict:
    label, reasons = label_of(findings, duration)
    causes = sorted(findings, key=lambda f: (f.t0, f.t1))
    return Verdict(episode_id, label, reasons, causes, dict(not_checked or {}), progress)
```

- [ ] **Step 5: `metrics.py` 的 `overlaps` 改从 `diagnose` 引**

删掉 `src/jianjin/eval/metrics.py` 里的 `def overlaps`（两行），在 import 区加：

```python
from jianjin.diagnose import overlaps
```

`tests/test_metrics.py` 仍从 `jianjin.eval.metrics` 引 `overlaps`，不用改。

- [ ] **Step 6: 跑全部测试**

Run: `uv run pytest -q`
Expected: 全部通过（108 + 9 = 117 个）

- [ ] **Step 7: 提交**

```bash
git add src/jianjin/types.py src/jianjin/diagnose.py src/jianjin/eval/metrics.py tests/test_diagnose.py
git commit -m "feat: 判定核心（硬规则、两个来源印证、存疑）；Finding 记来源"
```

---

### Task 2：视频规则「指令不符」

**Files:**
- Create: `src/jianjin/checks/video.py`
- Modify: `src/jianjin/eval/task_swap.py`（分数和换算改用 `checks/video.py`）
- Modify: `tests/helpers.py`（加 `write_cached_curve`）
- Test: `tests/test_video_checks.py`

- [ ] **Step 1: 测试辅助函数**

`tests/helpers.py` 末尾：

```python
# ---- 缓存的进度曲线 ----


def write_cached_curve(cache: Path, ep: Episode, camera: str, progress: list[float],
                       judge: str = "robometer", times: list[float] | None = None,
                       success: list[float] | None = None) -> Path:
    """直接写一份缓存 JSON，形状和 judge_episode 写的一样（本机没视频时按示范名和任务描述认）。"""
    from jianjin.judges.cache import _path

    path = _path(cache, judge, ep, camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    times = times if times is not None else list(np.linspace(0, ep.duration, len(progress)))
    path.write_text(json.dumps({"episode": ep.id, "camera": camera, "task": ep.task, "model": judge,
                                "version": "test", "times": [float(t) for t in times],
                                "progress": progress, "success": success}), encoding="utf-8")
    return path
```

- [ ] **Step 2: 写失败的测试**

`tests/test_video_checks.py`：

```python
import numpy as np
import pytest

from helpers import make_episode
from jianjin.checks import CheckConfig
from jianjin.checks.video import WRONG_TASK, relative, wrong_task
from jianjin.judges.base import Curve
from jianjin.types import Cause


def batch(n=6, task="pick"):
    return [make_episode(np.zeros(300), ep_id=f"e{i}", task=task) for i in range(n)]


def curve(peak):
    return Curve("robometer", "v", [0.0, 5.0, 10.0], [0.1, peak / 2, peak])


def test_relative_uses_median_and_iqr_with_floor():
    assert relative([0.5], [0.1, 0.2, 0.3, 0.4, 0.5]).tolist() == pytest.approx([1.0])
    assert relative([0.2], [0.1, 0.1, 0.1]).tolist() == pytest.approx([10.0])  # 四分位距 0，按 0.01 算


def test_wrong_task_compares_with_same_task_batch():
    eps = batch()
    curves = {ep.id: curve(0.9) for ep in eps}
    curves["e0"] = curve(0.3)
    found, skipped = wrong_task(eps, curves, "robometer", CheckConfig())
    assert list(found) == ["e0"] and not skipped
    (f,) = found["e0"]
    assert f.cause == Cause.WRONG_TASK and f.source == "robometer"
    assert (f.t0, f.t1) == (0.0, eps[0].duration) and f.score > WRONG_TASK["robometer"][1]
    assert "30%" in f.evidence and "90%" in f.evidence and "pick" in f.evidence


def test_wrong_task_skips_small_batches_and_missing_curves():
    eps = batch(4) + [make_episode(np.zeros(300), ep_id="p0", task="place")]
    curves = {ep.id: curve(0.9) for ep in eps[:3]}
    found, skipped = wrong_task(eps, curves, "robometer", CheckConfig())
    assert not found
    assert "只有 3 条" in skipped["e0"] and "缓存里没有" in skipped["e3"]


def test_wrong_task_uses_reference_when_given():
    clean = batch()
    odd = make_episode(np.zeros(300), ep_id="odd", task="pick")
    curves = {ep.id: curve(0.9) for ep in clean} | {"odd": curve(0.3)}
    found, _ = wrong_task([odd], curves, "robometer", CheckConfig(), reference=clean)
    assert list(found) == ["odd"]


def test_wrong_task_without_threshold_is_not_checked():
    eps = batch()
    found, skipped = wrong_task(eps, {ep.id: curve(0.9) for ep in eps}, "topreward", CheckConfig())
    assert not found and "还没在开发数据上给 TOPReward 定" in skipped["e0"]
```

- [ ] **Step 3: 跑测试，确认失败**

Run: `uv run pytest -q tests/test_video_checks.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'jianjin.checks.video'`

- [ ] **Step 4: 写 `checks/video.py`**

```python
"""视频评判的规则：从缓存的进度曲线里找问题。每个视频模型算一个独立来源（Finding.source 记模型名）。

阈值都在开发数据上定，出处写在阈值表旁边；某个模型还没定阈值的规则，记进 not_checked。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np

from jianjin.checks.config import CheckConfig, NotApplicable
from jianjin.judges.base import Curve
from jianjin.types import Cause, Episode, Finding, source_name

Found = dict[str, list[Finding]]
Skipped = dict[str, str]
Curves = dict[str, Curve]  # 示范 id -> 进度曲线

SPREAD_FLOOR = 0.01  # 进度的 1 个百分点：一批太齐时，别把一点小差别放大成异常


def _tail(x: list[float], k: int = 3) -> float:
    return float(np.mean(x[-k:]))


def _success_end(c: Curve) -> float:
    if c.success is None:
        raise NotApplicable(f"{c.model} 不给成功概率")
    return 1.0 - _tail(c.success)


# 分数越高越像「做的不是这个任务」；都是 1 减去某个 0 到 1 的量
BASE_SCORES: dict[str, Callable[[Curve], float]] = {
    "end": lambda c: 1.0 - _tail(c.progress),       # 结尾进度低
    "peak": lambda c: 1.0 - float(max(c.progress)),  # 全程最高进度低
    "mean": lambda c: 1.0 - float(np.mean(c.progress)),
    "success": _success_end,                         # 结尾成功概率低
}
SCORE_LABELS = {"end": "结尾进度", "peak": "全程最高进度", "mean": "平均进度", "success": "结尾的完成概率"}

# 指令不符：模型 -> (分数, 阈值)。和同任务那批比，差出阈值个四分位距以上就报。
WRONG_TASK: dict[str, tuple[str, float]] = {
    # 2026-10-03 考卷 A「换掉任务描述」dev 上定，原样报出不超过 5%（results/exam_a_swap_dev.json）
    "robometer": ("peak", 1.5477136091635564),
}


def relative(values, ref) -> np.ndarray:
    """按参照那批的中位数和四分位距换算：比常态差多少个四分位距。"""
    ref = np.asarray(ref, dtype=np.float64)
    med = float(np.median(ref))
    spread = max(float(np.subtract(*np.percentile(ref, [75, 25]))), SPREAD_FLOOR)
    return (np.asarray(values, dtype=np.float64) - med) / spread


def _all_skipped(episodes: list[Episode], why: str) -> tuple[Found, Skipped]:
    return {}, {ep.id: why for ep in episodes}


def wrong_task(episodes: list[Episode], curves: Curves, judge: str, cfg: CheckConfig,
               reference: list[Episode] | None = None) -> tuple[Found, Skipped]:
    """进度比同一任务描述下那批示范差得多：做的可能不是这个任务。参照默认是这一批自己。"""
    name = source_name(judge)
    if judge not in WRONG_TASK:
        return _all_skipped(episodes, f"还没在开发数据上给 {name} 定「指令不符」的阈值")
    score_name, thr = WRONG_TASK[judge]
    fn = BASE_SCORES[score_name]
    groups: dict[str, list[float]] = defaultdict(list)
    for ep in reference if reference is not None else episodes:
        if ep.id in curves:
            groups[ep.task].append(fn(curves[ep.id]))
    found: Found = {}
    skipped: Skipped = {}
    for ep in episodes:
        c = curves.get(ep.id)
        if c is None:
            skipped[ep.id] = f"缓存里没有 {name} 对这条示范的结果"
            continue
        ref = groups.get(ep.task, [])
        if len(ref) < cfg.min_task_episodes:
            skipped[ep.id] = f"同任务有 {name} 结果的只有 {len(ref)} 条，少于 {cfg.min_task_episodes} 条，没法比"
            continue
        z = float(relative([fn(c)], ref)[0])
        if z <= thr:
            continue
        value, med = 1.0 - fn(c), 1.0 - float(np.median(ref))
        found[ep.id] = [Finding(
            check="wrong_task", cause=Cause.WRONG_TASK, t0=0.0, t1=ep.duration, score=z,
            evidence=(f"{name}看视频，{SCORE_LABELS[score_name]} {value:.0%}，同任务 {len(ref)} 条示范的中位数是 "
                      f"{med:.0%}；做的可能不是「{ep.task}」"),
            source=judge,
        )]
    return found, skipped
```

- [ ] **Step 5: `task_swap.py` 改用这里的分数和换算**

`src/jianjin/eval/task_swap.py`：删掉 `REL_SPREAD_FLOOR`、`_tail`、`_success_end`、`BASE_SCORES` 的定义，改成从 `checks/video.py` 引；`_relative` 里的换算改调 `relative`：

```python
from jianjin.checks.video import BASE_SCORES, SPREAD_FLOOR, relative
```

```python
def _relative(clean: np.ndarray, bad: np.ndarray, clean_tasks: np.ndarray,
              bad_tasks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """每个任务描述下，用那批原样示范的中位数和四分位距换算：比同任务的常态差多少。"""
    rc, rb = np.empty(len(clean)), np.empty(len(bad))
    for t in set(clean_tasks) | set(bad_tasks):
        ref = clean[clean_tasks == t]
        if len(ref) < MIN_BATCH:
            raise NotApplicable(f"任务「{t}」只有 {len(ref)} 条原样示范，不够同任务比较")
        rc[clean_tasks == t] = relative(clean[clean_tasks == t], ref)
        rb[bad_tasks == t] = relative(bad[bad_tasks == t], ref)
    return rc, rb
```

模块文档字符串里提到 `REL_SPREAD_FLOOR` 的地方改成 `SPREAD_FLOOR`（定义在 `checks/video.py`）。`SPREAD_FLOOR` 只为让读代码的人看到下限在哪定，`task_swap.py` 里不直接用也可以不引；`ruff` 报未使用就去掉。

- [ ] **Step 6: 跑全部测试，再确认考卷 A 换描述的成绩没变**

Run: `uv run pytest -q`
Expected: 全部通过

Run: `uv run jianjin eval a-swap --split dev --out "$TEMP/swap_dev_check.json"`
Expected: 和 `results/exam_a_swap_dev.json` 一样：`rel_peak`，阈值 1.548，检出 57/103，原样报出 5/103

- [ ] **Step 7: 提交**

```bash
git add src/jianjin/checks/video.py src/jianjin/eval/task_swap.py tests/helpers.py tests/test_video_checks.py
git commit -m "feat: 视频规则「指令不符」（同任务比较，阈值来自考卷 A 换描述 dev）"
```

---

### Task 3：视频规则「进度回落」，考卷 B 按示范计

**Files:**
- Modify: `src/jianjin/checks/video.py`（`drawdown`、`PROGRESS_DROP`、`progress_drop`）
- Create: `src/jianjin/eval/exam_b_episode.py`
- Modify: `src/jianjin/cli.py`（`eval b-episode`）
- Test: `tests/test_video_checks.py`、`tests/test_exam_b_episode.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_video_checks.py` 末尾：

```python
from jianjin.checks.video import drawdown, progress_drop  # noqa: E402


def test_drawdown_finds_biggest_fall_from_running_peak():
    c = Curve("robometer", "v", [0, 1, 2, 3, 4], [0.1, 0.6, 0.3, 0.7, 0.65])
    assert drawdown(c) == pytest.approx((0.3, 1.0, 2.0, 0.6, 0.3))


def test_drawdown_of_rising_curve_is_zero():
    assert drawdown(Curve("robometer", "v", [0, 1, 2], [0.1, 0.2, 0.3]))[0] == 0.0


def test_progress_drop_reports_where_progress_fell():
    eps = [make_episode(np.zeros(90), ep_id=f"e{i}") for i in range(2)]
    curves = {"e0": Curve("robometer", "v", [0, 1, 2], [0.2, 0.8, 0.3]),
              "e1": Curve("robometer", "v", [0, 1, 2], [0.2, 0.5, 0.9])}
    found, skipped = progress_drop(eps, curves, "robometer", CheckConfig(), threshold=0.3)
    (f,) = found["e0"]
    assert f.cause == Cause.DROP and f.source == "robometer" and (f.t0, f.t1) == (1.0, 2.0)
    assert "80%" in f.evidence and "30%" in f.evidence
    assert "e1" not in found and not skipped


def test_progress_drop_without_threshold_or_curve():
    eps = [make_episode(np.zeros(90), ep_id="e0")]
    _, skipped = progress_drop(eps, {}, "topreward", CheckConfig())
    assert "还没在开发数据上给 TOPReward 定" in skipped["e0"]
    _, skipped = progress_drop(eps, {}, "robometer", CheckConfig(), threshold=0.3)
    assert "缓存里没有" in skipped["e0"]
```

`tests/test_exam_b_episode.py`：

```python
import numpy as np
import pytest

from helpers import make_robovad, write_cached_curve
from jianjin.eval.exam_b_episode import format_episode_b, run_episode_b
from jianjin.ingest.robovad import read_robovad

NORMAL = {f"h_stackcubes_ep{i}": d for i, d in enumerate([0.0, 0.05, 0.1, 0.2])}


def setup(tmp_path):
    names = [*NORMAL, "h_stackcubes_ep10", "h_stackcubes_ep11"]
    eps = {n: (np.zeros((150, 6)), np.zeros((150, 6))) for n in names}
    drop = [{"start_frame": 60, "end_frame": 90, "type": "object_drop", "categories": {}}]  # 第 2 到 3 秒
    root = make_robovad(tmp_path / "rv", eps, {"h_stackcubes_ep10": drop, "h_stackcubes_ep11": drop},
                        validation=names)
    cache = tmp_path / "cache"
    for ep in read_robovad(root).episodes:
        if ep.id in NORMAL:
            d = NORMAL[ep.id]
            write_cached_curve(cache, ep, "left", [0.1, 0.5, 0.5 - d, 0.9], times=[0, 1, 2, 4])
        elif ep.id == "h_stackcubes_ep10":  # 第 2 到 2.5 秒掉了 0.5，落在标注里
            write_cached_curve(cache, ep, "left", [0.1, 0.3, 0.8, 0.3, 0.6], times=[0, 1, 2, 2.5, 4])
        else:  # 只掉了 0.15，抓不到
            write_cached_curve(cache, ep, "left", [0.1, 0.6, 0.45, 0.9], times=[0, 1, 2, 4])
    return root, cache


def test_threshold_from_normal_episodes(tmp_path):
    root, cache = setup(tmp_path)
    out = run_episode_b(root, "validation", cache)
    assert out["threshold"] == pytest.approx(0.2) and out["chosen_on_this_split"]
    assert (out["n_episodes"], out["n_anomalous"], out["missing"]) == (6, 2, 0)
    assert out["detected"] == 1 and out["localized"] == 1 and out["normal_flagged"] == 0
    assert "分母" in format_episode_b(out)


def test_given_threshold_is_used_as_is(tmp_path):
    root, cache = setup(tmp_path)
    out = run_episode_b(root, "validation", cache, threshold=0.6)
    assert out["detected"] == 0 and not out["chosen_on_this_split"]
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `uv run pytest -q tests/test_video_checks.py tests/test_exam_b_episode.py`
Expected: FAIL，`ImportError: cannot import name 'drawdown'` 和 `No module named 'jianjin.eval.exam_b_episode'`

- [ ] **Step 3: `checks/video.py` 加进度回落**

`WRONG_TASK` 下面加阈值表（先空着，Step 7 填）：

```python
# 进度回落：模型 -> 阈值。进度从此前最高点往下掉超过这么多就报。
PROGRESS_DROP: dict[str, float] = {}
```

文件末尾加：

```python
def drawdown(c: Curve) -> tuple[float, float, float, float, float]:
    """进度从此前最高点往下掉得最多的一次：(掉了多少, 最高点时刻, 最低点时刻, 最高, 最低)。"""
    p = np.asarray(c.progress, dtype=np.float64)
    t = np.asarray(c.times, dtype=np.float64)
    run = np.maximum.accumulate(p)
    k = int(np.argmax(run - p))
    i = int(np.argmax(p[: k + 1]))
    return float(run[k] - p[k]), float(t[i]), float(t[k]), float(p[i]), float(p[k])


def progress_drop(episodes: list[Episode], curves: Curves, judge: str, cfg: CheckConfig,
                  reference: list[Episode] | None = None, threshold: float | None = None) -> tuple[Found, Skipped]:
    """进度突然往回掉：东西掉了，或者退回去重做。不和同任务比，cfg、reference 只为和别的规则同一个签名。"""
    name = source_name(judge)
    thr = threshold if threshold is not None else PROGRESS_DROP.get(judge)
    if thr is None:
        return _all_skipped(episodes, f"还没在开发数据上给 {name} 定「进度回落」的阈值")
    found: Found = {}
    skipped: Skipped = {}
    for ep in episodes:
        c = curves.get(ep.id)
        if c is None:
            skipped[ep.id] = f"缓存里没有 {name} 对这条示范的结果"
            continue
        d, t0, t1, hi, lo = drawdown(c)
        if d > thr:
            found[ep.id] = [Finding(
                check="progress_drop", cause=Cause.DROP, t0=t0, t1=t1, score=d,
                evidence=f"{name}看视频，进度在第 {t0:.1f} 到 {t1:.1f} 秒从 {hi:.0%} 掉到 {lo:.0%}，可能东西掉了或者退回去重做",
                source=judge,
            )]
    return found, skipped
```

- [ ] **Step 4: 写 `eval/exam_b_episode.py`**

```python
"""考卷 B 按示范计：RoboVAD 的示范有没有异常标注，对上「进度回落」规则报没报、报的时间对不对。

阈值在 train 的正常示范上取 95 分位（正常的最多报出 5%）；test 只用 train 定的阈值跑一次。
test 用 annotations.json 里的标注，不用 evaluation/*.csv（那是逐帧评测的帧清单）。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from jianjin.checks.video import drawdown
from jianjin.diagnose import overlaps
from jianjin.eval.auc import roc_auc
from jianjin.eval.exam_b import split_names
from jianjin.eval.metrics import wilson
from jianjin.ingest.robovad import read_robovad
from jianjin.judges.cache import load_curve
from jianjin.types import Episode

NORMAL_FLAG_MAX = 0.05


def annotated_seconds(ep: Episode) -> list[tuple[float, float]]:
    fi, ts = ep.meta["frame_index"], ep.timestamps
    return [(float(np.interp(a["start_frame"], fi, ts)), float(np.interp(a["end_frame"], fi, ts)))
            for a in ep.meta["anomalies"]]


def run_episode_b(root: str | Path, split: str, cache_dir: str | Path = "cache", judge: str = "robometer",
                  camera: str = "left", threshold: float | None = None) -> dict:
    """threshold 不给就在这一批的正常示范上定（只许在 train 上这么做）。"""
    root = Path(root)
    episodes = read_robovad(root, names=split_names(root, split)).episodes
    rows, missing = [], 0
    for ep in episodes:
        c = load_curve(Path(cache_dir), judge, ep, camera)
        if c is None:
            missing += 1
            continue
        d, t0, t1, _, _ = drawdown(c)
        rows.append((bool(ep.meta["anomalies"]), d, (t0, t1), annotated_seconds(ep)))
    y = np.array([r[0] for r in rows], dtype=bool)
    s = np.array([r[1] for r in rows], dtype=np.float64)
    chosen_here = threshold is None
    if chosen_here:
        normal = s[~y]
        threshold = float(np.quantile(normal, 1 - NORMAL_FLAG_MAX, method="higher")) if len(normal) else math.nan
    flagged = s > threshold
    pos, neg = int(y.sum()), int((~y).sum())
    detected = int((flagged & y).sum())
    normal_flagged = int((flagged & ~y).sum())
    localized = sum(1 for r, f in zip(rows, flagged) if f and r[0] and any(overlaps(r[2], a) for a in r[3]))
    return {
        "exam": "B·按示范计（RoboVAD：示范有没有异常标注；规则是进度从最高点回落）",
        "split": split,
        "judge": judge,
        "camera": camera,
        "n_episodes": len(rows),
        "missing": missing,
        "n_anomalous": pos,
        "n_normal": neg,
        "auc": roc_auc(s, y.astype(np.int8)),
        "threshold": threshold,
        "chosen_on_this_split": chosen_here,
        "detected": detected,
        "detect_ci": wilson(detected, pos),
        "localized": localized,
        "localize_ci": wilson(localized, pos),
        "normal_flagged": normal_flagged,
        "normal_flag_ci": wilson(normal_flagged, neg),
    }


def format_episode_b(out: dict) -> str:
    def pct(k: int, n: int, ci) -> str:
        return "—" if n == 0 else f"{k}/{n} = {k / n:.0%}（95% 区间 {ci[0]:.0%} 到 {ci[1]:.0%}）"

    how = "在这一批的正常示范上定的" if out["chosen_on_this_split"] else "沿用 train 定的"
    return "\n".join([
        f"考卷 {out['exam']}",
        f"切分：{out['split']}；评判：{out['judge']}（{out['camera']} 路相机）",
        f"分母：有缓存的 {out['n_episodes']} 条（缺 {out['missing']} 条），其中有异常标注 {out['n_anomalous']} 条、"
        f"正常 {out['n_normal']} 条",
        f"AUC {out['auc']:.3f}；阈值 {out['threshold']:.3f}（{how}）",
        f"  有异常的检出：{pct(out['detected'], out['n_anomalous'], out['detect_ci'])}",
        f"  报的时间落在标注区间：{pct(out['localized'], out['n_anomalous'], out['localize_ci'])}",
        f"  正常的被报出：{pct(out['normal_flagged'], out['n_normal'], out['normal_flag_ci'])}",
    ])
```

- [ ] **Step 5: `cli.py` 加 `eval b-episode`**

`_cmd_eval` 开头加一个分支（放在 `a-swap` 分支前面）：

```python
    if args.exam == "b-episode":
        from jianjin.eval.exam_b_episode import format_episode_b, run_episode_b

        split = args.split or "train"
        threshold = None
        if args.dev_result:
            threshold = json.loads(Path(args.dev_result).read_text(encoding="utf-8"))["threshold"]
        elif split != "train":
            raise SystemExit("test 要用 train 定的阈值：加 --dev-result results/exam_b_episode_train.json")
        out = run_episode_b(args.robovad_root, split, args.cache, judge=args.judge,
                            camera=args.camera or "left", threshold=threshold)
        text = format_episode_b(out)
    elif args.exam == "a-swap":
```

`e.add_argument("exam", choices=["a", "a-swap", "b"])` 改成 `choices=["a", "a-swap", "b", "b-episode"]`；`--dev-result` 的 help 改成「沿用这个结果里的阈值：a-swap 用 dev 的，b-episode 用 train 的」。输出文件名沿用 `results/exam_{exam 里的 - 换成 _}_{split}.json`，即 `results/exam_b_episode_train.json`。

- [ ] **Step 6: 跑测试**

Run: `uv run pytest -q`
Expected: 全部通过

- [ ] **Step 7: 在 RoboVAD train 上定阈值，写进阈值表**

Run: `uv run jianjin eval b-episode --split train --camera left`
Expected: 367 条、有异常 60 条；AUC 约 0.797；阈值约 0.344；检出约 17/60；正常报出不超过 5%。结果写到 `results/exam_b_episode_train.json`。

把打印出的阈值（JSON 里 `threshold` 的完整小数）填进 `checks/video.py`：

```python
PROGRESS_DROP: dict[str, float] = {
    # 2026-10-03 RoboVAD train 上定：正常示范（307 条）进度回落的 95 分位（results/exam_b_episode_train.json）
    "robometer": <results/exam_b_episode_train.json 里的 threshold>,
}
```

- [ ] **Step 8: test 只跑一次**

Run: `uv run jianjin eval b-episode --split test --camera left --dev-result results/exam_b_episode_train.json`
Expected: 629 条、有异常 387 条；打印检出、时间对上、正常报出，写到 `results/exam_b_episode_test.json`。数字照实记，不回头改阈值。

- [ ] **Step 9: 提交**

```bash
git add src/jianjin/checks/video.py src/jianjin/eval/exam_b_episode.py src/jianjin/cli.py tests/ results/exam_b_episode_train.json results/exam_b_episode_test.json
git commit -m "feat: 视频规则「进度回落」，阈值在 RoboVAD train 上定；考卷 B 按示范计的成绩"
```

---

### Task 4：一批示范出判定

**Files:**
- Modify: `src/jianjin/checks/video.py`（`VIDEO_RULES`、`run_video_checks`）
- Modify: `src/jianjin/judges/cache.py`（`load_curves`）
- Modify: `src/jianjin/diagnose.py`（`diagnose_batch`）
- Test: `tests/test_diagnose.py`、`tests/test_judges.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_diagnose.py` 末尾：

```python
import numpy as np  # noqa: E402

from helpers import make_episode  # noqa: E402
from jianjin.diagnose import diagnose_batch  # noqa: E402


def smooth_batch(n=6):
    x = np.linspace(0, 1, 300)
    return [make_episode(np.tile(x[:, None], (1, 6)) * (1 + 0.01 * i), ep_id=f"e{i}") for i in range(n)]


def rising(top):
    return Curve("robometer", "v", [0.0, 5.0, 10.0], [0.1, top / 2, top])


def test_batch_without_video_says_so():
    vs = diagnose_batch(smooth_batch())
    assert [v.label for v in vs] == [Label.PASS] * 6
    assert all("没跑视频评判" in v.not_checked["视频评判"] for v in vs)


def test_batch_with_video_rules_and_progress():
    eps = smooth_batch()
    curves = {ep.id: rising(0.9) for ep in eps}
    curves["e0"] = rising(0.3)
    vs = diagnose_batch(eps, {"robometer": curves, "topreward": {}})
    v0 = vs[0]
    assert v0.label == Label.DOUBT and [f.cause for f in v0.causes] == [Cause.WRONG_TASK]
    assert v0.progress is curves["e0"]
    assert vs[1].label == Label.PASS and "视频评判" not in vs[1].not_checked
    assert "还没在开发数据上给 TOPReward 定" in vs[1].not_checked["topreward.wrong_task"]
```

`tests/test_judges.py` 末尾：

```python
def test_load_curves_reads_a_batch(tmp_path):
    import numpy as np

    from helpers import make_episode, write_cached_curve
    from jianjin.judges.cache import load_curves

    eps = [make_episode(np.zeros(90), ep_id=f"e{i}") for i in range(3)]
    for ep in eps:
        ep.cameras = {"front": VideoRef(tmp_path / "none.mp4", 0.0, 3.0)}
    write_cached_curve(tmp_path / "cache", eps[0], "front", [0.1, 0.9])
    got = load_curves(eps, tmp_path / "cache", "robometer", "front")
    assert list(got) == ["e0"] and got["e0"].progress == [0.1, 0.9]
    assert load_curves([make_episode(np.zeros(90))], tmp_path / "cache", "robometer") == {}  # 没视频也不报错
```

（`tests/test_judges.py` 顶部已有 `from jianjin.types import VideoRef`；没有就加上。）

- [ ] **Step 2: 跑测试，确认失败**

Run: `uv run pytest -q tests/test_diagnose.py tests/test_judges.py`
Expected: FAIL，`ImportError: cannot import name 'diagnose_batch'`、`cannot import name 'load_curves'`

- [ ] **Step 3: `judges/cache.py` 加 `load_curves`**

```python
def load_curves(episodes: list[Episode], cache_dir: Path, judge_name: str,
                camera: str | None = None) -> dict[str, Curve]:
    """一批示范从缓存里读出来的曲线；没缓存、没视频的跳过。"""
    out = {}
    for ep in episodes:
        try:
            c = load_curve(Path(cache_dir), judge_name, ep, camera)
        except NotApplicable:
            continue
        if c is not None:
            out[ep.id] = c
    return out
```

- [ ] **Step 4: `checks/video.py` 加规则表和总入口**

文件末尾：

```python
VIDEO_RULES = {"wrong_task": wrong_task, "progress_drop": progress_drop}


def run_video_checks(episodes: list[Episode], curves: dict[str, Curves], cfg: CheckConfig | None = None,
                     reference: list[Episode] | None = None) -> dict[str, tuple[list[Finding], dict[str, str]]]:
    """curves：{模型名: {示范 id: 曲线}}。返回 {示范 id: (发现, 没查的)}，没查的键是「模型.规则」。"""
    cfg = cfg or CheckConfig()
    out: dict[str, tuple[list[Finding], dict[str, str]]] = {ep.id: ([], {}) for ep in episodes}
    for judge, cs in curves.items():
        for name, rule in VIDEO_RULES.items():
            found, skipped = rule(episodes, cs, judge, cfg, reference=reference)
            for eid, fs in found.items():
                out[eid][0].extend(fs)
            for eid, why in skipped.items():
                out[eid][1][f"{judge}.{name}"] = why
    return out
```

- [ ] **Step 5: `diagnose.py` 加 `diagnose_batch`**

import 区加：

```python
from jianjin.checks import CheckConfig, run_checks
from jianjin.checks.video import Curves, run_video_checks
from jianjin.types import Episode, Finding, source_name
```

（替换原来那行 `from jianjin.types import Finding, source_name`。）文件末尾：

```python
PROGRESS_FROM = ("robometer", "topreward")  # 报告里画谁的进度曲线，按这个顺序找


def diagnose_batch(episodes: list[Episode], curves: dict[str, Curves] | None = None,
                   cfg: CheckConfig | None = None, reference: list[Episode] | None = None) -> list[Verdict]:
    """关节规则加视频规则，逐条出判定。curves 是 {模型名: {示范 id: 曲线}}，没有就只跑关节体检。"""
    curves = curves or {}
    checks = {r.episode_id: r for r in run_checks(episodes, cfg, reference=reference)}
    video = run_video_checks(episodes, curves, cfg, reference)
    order = [j for j in PROGRESS_FROM if j in curves] + [j for j in curves if j not in PROGRESS_FROM]
    out = []
    for ep in episodes:
        r = checks[ep.id]
        vf, vn = video[ep.id]
        not_checked = {**r.not_checked, **vn}
        if not curves:
            not_checked["视频评判"] = "没跑视频评判（没有 GPU，或者缓存里没有结果）"
        progress = next((curves[j][ep.id] for j in order if ep.id in curves[j]), None)
        out.append(diagnose(ep.id, ep.duration, r.findings + vf, not_checked, progress))
    return out
```

- [ ] **Step 6: 跑全部测试**

Run: `uv run pytest -q`
Expected: 全部通过

- [ ] **Step 7: 提交**

```bash
git add src/jianjin/checks/video.py src/jianjin/judges/cache.py src/jianjin/diagnose.py tests/test_diagnose.py tests/test_judges.py
git commit -m "feat: 一批示范出判定（关节规则加视频规则）；从缓存成批读曲线"
```

---

### Task 5：离线单文件网页报告

**Files:**
- Create: `src/jianjin/report.py`
- Test: `tests/test_report.py`、`tests/snapshots/report_small.html`（第一次跑测试时生成）

- [ ] **Step 1: 写失败的测试**

`tests/test_report.py`：

```python
import os
from pathlib import Path

import numpy as np

from helpers import make_episode
from jianjin.diagnose import diagnose
from jianjin.judges.base import Curve
from jianjin.report import render
from jianjin.types import Cause, Finding

SNAP = Path(__file__).parent / "snapshots" / "report_small.html"


def small():
    eps = [make_episode(np.zeros(90), ep_id=f"ds/ep{i}", task="pick <the> cube") for i in range(3)]
    pause = Finding("long_pause", Cause.LONG_PAUSE, 1.0, 2.0, 1.0, "第 1.0 到 2.0 秒关节几乎不动")
    drop = Finding("progress_drop", Cause.DROP, 1.5, 2.5, 0.4, "进度从 80% 掉到 40%", source="robometer")
    curve = Curve("robometer", "v", [0.0, 1.0, 2.0, 3.0], [0.1, 0.8, 0.4, 0.9])
    vs = [
        diagnose("ds/ep0", 3.0, [pause, drop], {}, curve),
        diagnose("ds/ep1", 3.0, [pause], {"robometer.wrong_task": "同任务只有 3 条"}, curve),
        diagnose("ds/ep2", 3.0, [], {}, None),
    ]
    return vs, eps


def test_snapshot():
    vs, eps = small()
    html = render(vs, eps, title="ds", judges=["robometer"], generated="2026-10-03 12:00")
    if os.environ.get("JIANJIN_UPDATE_SNAPSHOTS") or not SNAP.exists():
        SNAP.parent.mkdir(exist_ok=True)
        SNAP.write_text(html, encoding="utf-8")
    assert html == SNAP.read_text(encoding="utf-8")


def test_counts_order_and_escaping():
    vs, eps = small()
    html = render(vs, eps, title="ds <1>", judges=["robometer"])
    assert "pick &lt;the&gt; cube" in html and "<the>" not in html and "ds &lt;1&gt;" in html
    assert html.count('<article class="card ') == 3
    assert html.index('data-label="作废"') < html.index('data-label="存疑"') < html.index('data-label="合格"')
    assert "人工只需要看存疑的 1 条" in html


def test_curve_with_marked_findings():
    vs, eps = small()
    html = render(vs, eps, title="ds", judges=["robometer"])
    assert '<polyline points="0.0,48.8' in html and 'class="mark reject"' in html


def test_banner_only_without_video():
    vs, eps = small()
    assert "没跑视频评判" in render(vs, eps, title="ds", judges=[])
    assert "没跑视频评判" not in render(vs, eps, title="ds", judges=["robometer"])


def test_operator_table_only_with_ids():
    vs, eps = small()
    assert "没有采集员编号" in render(vs, eps, title="ds")
    eps[0].operator_id = eps[1].operator_id = "王五"
    html = render(vs, eps, title="ds")
    assert "<td>王五</td><td>2</td><td>1</td><td>1</td>" in html and "（未填）" in html


def test_read_failures_listed():
    vs, eps = small()
    assert "读取失败 1 个文件" in render(vs, eps, title="ds", failures=[("a.hdf5", "坏了")])
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `uv run pytest -q tests/test_report.py`
Expected: FAIL，`ModuleNotFoundError: No module named 'jianjin.report'`

- [ ] **Step 3: 写 `report.py`**

```python
"""离线单文件网页报告：整批的判定漏斗、问题分布、按采集员统计，加上每条示范一张体检单。

不引用任何外部资源（样式、脚本都在文件里），断网也能打开，数据不出场。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from html import escape

from jianjin.diagnose import Label, Verdict
from jianjin.types import Cause, Episode, Finding, source_name

ORDER = [Label.REJECT, Label.DOUBT, Label.PASS]  # 体检单的排列顺序
TILES = [Label.PASS, Label.DOUBT, Label.REJECT]  # 漏斗的排列顺序
CSS_CLASS = {Label.PASS: "pass", Label.DOUBT: "doubt", Label.REJECT: "reject"}
CAUSE_ORDER = list(Cause)

CSS = """
:root{--bg:#f6f5f1;--fg:#1d1d1b;--muted:#6b6a64;--card:#fff;--line:#e2e0d9;--curve:#2b5cab;
--pass:#2f7d4f;--doubt:#a86b12;--reject:#c0392b;--pass-bg:#e5f1e9;--doubt-bg:#fbefd8;--reject-bg:#f9e2de}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#9a9993;--card:#1f1f1d;--line:#34342f;
--curve:#7aa7ef;--pass:#5cbf86;--doubt:#e0a640;--reject:#ef6f61;--pass-bg:#1c3125;--doubt-bg:#3a2d15;--reject-bg:#3d1f1b}}
*{box-sizing:border-box}
body{margin:0 auto;max-width:1080px;padding:28px 16px 40px;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:18px;margin:36px 0 12px}
.sub,.note{color:var(--muted);margin:2px 0}
.banner{background:var(--doubt-bg);border:1px solid var(--doubt);border-radius:8px;padding:8px 12px;margin:14px 0 0}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:22px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;display:flex;flex-direction:column}
.tile b{font-size:34px;line-height:1.1;font-variant-numeric:tabular-nums}
.tile.pass b{color:var(--pass)}.tile.doubt b{color:var(--doubt)}.tile.reject b{color:var(--reject)}
.tile em{font-style:normal;color:var(--muted);font-size:13px}
.bar{display:flex;height:10px;border-radius:5px;overflow:hidden;margin:14px 0 8px;background:var(--line)}
.bar i.pass{background:var(--pass)}.bar i.doubt{background:var(--doubt)}.bar i.reject{background:var(--reject)}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{color:var(--muted);font-weight:500;font-size:13px}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.filters button{font:inherit;font-size:13px;padding:4px 12px;border-radius:999px;border:1px solid var(--line);
background:var(--card);color:var(--fg);cursor:pointer}
.filters button.on{border-color:var(--fg)}
.cards{display:flex;flex-direction:column;gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:10px;padding:12px 16px}
.card.pass{border-left-color:var(--pass)}.card.doubt{border-left-color:var(--doubt)}.card.reject{border-left-color:var(--reject)}
.head{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 12px}
.badge{font-size:12px;padding:1px 9px;border-radius:999px;white-space:nowrap}
.badge.pass{background:var(--pass-bg);color:var(--pass)}.badge.doubt{background:var(--doubt-bg);color:var(--doubt)}
.badge.reject{background:var(--reject-bg);color:var(--reject)}
code{font-family:ui-monospace,Consolas,monospace;font-size:13px;overflow-wrap:anywhere}
.task{color:var(--muted)}.dur{color:var(--muted);font-size:13px;margin-left:auto}
.reasons{margin:8px 0 0;padding-left:18px}
.findings{list-style:none;margin:8px 0 0;padding:0}
.findings li{padding:5px 0;border-top:1px dashed var(--line)}
.cause{font-weight:600;margin-right:8px}.when,.src{color:var(--muted);font-size:13px;margin-right:8px}
.spark{margin:10px 0 0}
.spark svg{width:100%;max-width:440px;height:56px;display:block;background:var(--bg);border-radius:6px}
.spark polyline{fill:none;stroke:var(--curve);stroke-width:2;vector-effect:non-scaling-stroke}
.spark rect.mark{opacity:.28}.spark rect.mark.doubt{fill:var(--doubt)}.spark rect.mark.reject{fill:var(--reject)}
figcaption{color:var(--muted);font-size:12px}
details{margin-top:8px;color:var(--muted);font-size:13px}
details ul{margin:4px 0 0;padding-left:18px}
footer{margin-top:44px;color:var(--muted);font-size:13px;border-top:1px solid var(--line);padding-top:12px}
@media (max-width:600px){.tiles{grid-template-columns:1fr}.dur{margin-left:0}}
"""

JS = """
document.querySelectorAll(".filters button").forEach(function (b) {
  b.addEventListener("click", function () {
    document.querySelectorAll(".filters button").forEach(function (x) { x.classList.toggle("on", x === b); });
    document.querySelectorAll(".card").forEach(function (c) {
      c.hidden = b.dataset.f !== "all" && c.dataset.label !== b.dataset.f;
    });
  });
});
"""

FOOTER = (
    "<footer><p>判定规则：作废是客观硬规则触发（读数缺失、丢帧、示范过短），或者两个独立来源指向同一处问题；"
    "存疑是有发现但够不上作废，要人看；合格是什么都没报。错杀一条好数据比漏掉一条坏数据贵，所以判得偏保守。</p>"
    "<p>没查的项目不当成通过。阈值都在开发数据上定，出处见拣金设计文档。</p></footer>"
)


def _pct(k: int, n: int) -> str:
    return f"{k / n:.0%}" if n else "—"


def _header(title: str, n: int, judges: list[str], generated: str) -> str:
    checked = "关节体检" + (f"；视频评判：{'、'.join(source_name(j) for j in judges)}" if judges else "")
    when = f" · 生成于 {escape(generated)}" if generated else ""
    banner = "" if judges else (
        '<p class="banner">没跑视频评判，只有关节体检。「指令不符」「物体掉落」这几类查不了，记在每条的「没查」里。</p>')
    return (f'<header><h1>拣金质检报告</h1><p class="sub">{escape(title)} · {n} 条示范{when}</p>'
            f'<p class="sub">查了什么：{checked}</p>{banner}</header>')


def _funnel(verdicts: list[Verdict]) -> str:
    n = len(verdicts)
    c = Counter(v.label for v in verdicts)
    tiles = "".join(f'<div class="tile {CSS_CLASS[lb]}"><b>{c[lb]}</b><span>{lb.value}</span>'
                    f'<em>{_pct(c[lb], n)}</em></div>' for lb in TILES)
    bar = "".join(f'<i class="{CSS_CLASS[lb]}" style="width:{c[lb] / n * 100:.2f}%"></i>'
                  for lb in TILES if n and c[lb])
    note = (f"人工只需要看存疑的 {c[Label.DOUBT]} 条（{_pct(c[Label.DOUBT], n)}）；"
            f"作废的 {c[Label.REJECT]} 条每条都写了依据。")
    return f'<section class="funnel"><div class="tiles">{tiles}</div><div class="bar">{bar}</div><p class="note">{note}</p></section>'


def _cause_set(v: Verdict) -> list[Cause]:
    return sorted({f.cause for f in v.causes}, key=CAUSE_ORDER.index)


def _causes(verdicts: list[Verdict]) -> str:
    counts: dict[Cause, Counter] = defaultdict(Counter)
    for v in verdicts:
        for cause in _cause_set(v):
            counts[cause][v.label] += 1
    if not counts:
        return '<section><h2>问题出在哪</h2><p class="note">这一批没有报出任何问题。</p></section>'
    rows = "".join(f"<tr><td>{c.value}</td><td>{counts[c][Label.REJECT]}</td><td>{counts[c][Label.DOUBT]}</td></tr>"
                   for c in CAUSE_ORDER if c in counts)
    return ('<section><h2>问题出在哪</h2><table class="causes"><thead><tr><th>原因</th><th>作废</th><th>存疑</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
            '<p class="note">按示范计；一条示范报了几类原因，每类各记一次。</p></section>')


def _operators(verdicts: list[Verdict], episodes: list[Episode]) -> str:
    ops = {ep.id: ep.operator_id for ep in episodes}
    if not any(ops.values()):
        return ('<section><h2>按采集员</h2><p class="note">数据里没有采集员编号（公开数据都没有）。'
                '试点数据带上编号后，这里按人统计问题分布。</p></section>')
    table: dict[str, list] = defaultdict(lambda: [0, Counter(), Counter()])  # 示范数、判定、原因
    for v in verdicts:
        row = table[ops.get(v.episode_id) or "（未填）"]
        row[0] += 1
        row[1][v.label] += 1
        row[2].update(_cause_set(v))
    rows = "".join(
        f"<tr><td>{escape(op)}</td><td>{r[0]}</td><td>{r[1][Label.DOUBT]}</td><td>{r[1][Label.REJECT]}</td>"
        f"<td>{r[2].most_common(1)[0][0].value if r[2] else '—'}</td></tr>"
        for op, r in sorted(table.items()))
    return ('<section><h2>按采集员</h2><table class="ops"><thead><tr><th>采集员</th><th>示范</th><th>存疑</th>'
            f'<th>作废</th><th>最常见的问题</th></tr></thead><tbody>{rows}</tbody></table></section>')


def _failures(failures: list[tuple[str, str]]) -> str:
    if not failures:
        return ""
    items = "".join(f"<li><code>{escape(name)}</code>：{escape(why)}</li>" for name, why in failures)
    return f'<section><h2>读取失败 {len(failures)} 个文件</h2><ul class="failures">{items}</ul></section>'


def _spark(v: Verdict, duration: float) -> str:
    c = v.progress
    if c is None or not c.times or duration <= 0:
        return ""
    w, h = 320, 56

    def x(t: float) -> float:
        return min(max(t / duration, 0.0), 1.0) * w

    def y(p: float) -> float:
        return (1.0 - min(max(p, 0.0), 1.0)) * (h - 4) + 2

    marks = "".join(
        f'<rect class="mark {CSS_CLASS[v.label]}" x="{x(f.t0):.1f}" y="0" width="{max(x(f.t1) - x(f.t0), 2.0):.1f}" height="{h}"/>'
        for f in v.causes)
    pts = " ".join(f"{x(t):.1f},{y(p):.1f}" for t, p in zip(c.times, c.progress))
    name = f"{source_name(c.model)} 的进度曲线"
    return (f'<figure class="spark"><svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" '
            f'aria-label="{escape(name)}">{marks}<polyline points="{pts}"/></svg>'
            f'<figcaption>{escape(name)}，横轴 0 到 {duration:.1f} 秒，色块是报出问题的时间段</figcaption></figure>')


def _finding(f: Finding) -> str:
    return (f'<li><span class="cause">{f.cause.value}</span><span class="when">第 {f.t0:.1f} 到 {f.t1:.1f} 秒</span>'
            f'<span class="src">{escape(source_name(f.source))}</span>{escape(f.evidence)}</li>')


def _card(v: Verdict, ep: Episode) -> str:
    cls = CSS_CLASS[v.label]
    parts = [f'<article class="card {cls}" data-label="{v.label.value}">',
             f'<div class="head"><span class="badge {cls}">{v.label.value}</span><code>{escape(v.episode_id)}</code>'
             f'<span class="task">{escape(ep.task)}</span><span class="dur">{ep.duration:.1f} 秒</span></div>']
    if v.reasons:
        parts.append('<ul class="reasons">' + "".join(f"<li>{escape(r)}</li>" for r in v.reasons) + "</ul>")
    parts.append(_spark(v, ep.duration))
    if v.causes:
        parts.append('<ul class="findings">' + "".join(_finding(f) for f in v.causes) + "</ul>")
    if v.not_checked:
        items = "".join(f"<li>{escape(k)}：{escape(why)}</li>" for k, why in sorted(v.not_checked.items()))
        parts.append(f"<details><summary>没查 {len(v.not_checked)} 项</summary><ul>{items}</ul></details>")
    parts.append("</article>")
    return "".join(parts)


def _cards(verdicts: list[Verdict], episodes: list[Episode]) -> str:
    by_id = {ep.id: ep for ep in episodes}
    c = Counter(v.label for v in verdicts)
    buttons = [f'<button type="button" data-f="all" class="on">全部 {len(verdicts)}</button>'] + [
        f'<button type="button" data-f="{lb.value}">{lb.value} {c[lb]}</button>' for lb in ORDER]
    ordered = sorted(verdicts, key=lambda v: ORDER.index(v.label))  # 稳定排序：同一类里保持原来的顺序
    cards = "".join(_card(v, by_id[v.episode_id]) for v in ordered)
    return (f'<section><h2>每条示范</h2><nav class="filters">{"".join(buttons)}</nav>'
            f'<div class="cards">{cards}</div></section>')


def render(verdicts: list[Verdict], episodes: list[Episode], *, title: str, judges=(), failures=(),
           generated: str = "") -> str:
    body = "".join([
        _header(title, len(verdicts), list(judges), generated),
        _funnel(verdicts),
        _causes(verdicts),
        _operators(verdicts, episodes),
        _failures(list(failures)),
        _cards(verdicts, episodes),
        FOOTER,
    ])
    return ('<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>拣金质检报告 · {escape(title)}</title><style>{CSS}</style></head>'
            f'<body>{body}<script>{JS}</script></body></html>\n')
```

- [ ] **Step 4: 跑测试**

Run: `uv run pytest -q tests/test_report.py`
Expected: 全部通过（快照第一次跑时生成 `tests/snapshots/report_small.html`）

- [ ] **Step 5: 看一眼快照**

用浏览器打开 `tests/snapshots/report_small.html`：三张卡片、漏斗、原因表、采集员说明都在；第一张卡的曲线上有红色色块。

- [ ] **Step 6: 跑全部测试，提交**

Run: `uv run pytest -q`
Expected: 全部通过

```bash
git add src/jianjin/report.py tests/test_report.py tests/snapshots/report_small.html
git commit -m "feat: 离线单文件网页报告（判定漏斗、问题分布、按采集员、每条体检单）"
```

---

### Task 6：`jianjin check` 出判定和报告；`load` 认出 RoboVAD

**Files:**
- Modify: `src/jianjin/ingest/__init__.py`
- Modify: `src/jianjin/cli.py`（`_cmd_check`、`check` 的参数）
- Test: `tests/test_cli.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_cli.py` 末尾：

```python
def moving6(n, seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.uniform(0.5, 1.5, n))
    return np.tile(x[:, None], (1, 6)) * np.arange(1, 7)


def test_check_writes_report(tmp_path, capsys):
    ds = make_lerobot_v3(tmp_path / "d", [("pick", paused6())])
    out = tmp_path / "r.html"
    assert main(["check", str(ds), "--out", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    assert "拣金质检报告" in html and "没跑视频评判" in html
    assert "存疑 1" in capsys.readouterr().out


def test_check_with_cached_video(tmp_path, capsys):
    from helpers import write_cached_curve

    ds = make_lerobot_v3(tmp_path / "d", [("pick", moving6(300, i)) for i in range(6)])
    cache = tmp_path / "cache"
    cam = "observation.images.front"
    for i, ep in enumerate(load(ds).episodes):
        write_cached_curve(cache, ep, cam, [0.1, 0.5, 0.3 if i == 0 else 0.9])
    js = tmp_path / "v.json"
    assert main(["check", str(ds), "--judges", "robometer", "--cache", str(cache), "--camera", cam,
                 "--json", str(js)]) == 0
    rows = json.loads(js.read_text(encoding="utf-8"))
    assert rows[0]["label"] == "存疑" and rows[0]["findings"][0]["source"] == "robometer"
    assert "Robometer：6/6 条有缓存" in capsys.readouterr().out


def test_load_detects_robovad(tmp_path):
    from helpers import make_robovad

    root = make_robovad(tmp_path / "rv", {"h_stackcubes_ep1": (np.zeros((60, 6)), np.zeros((60, 6)))})
    assert [e.id for e in load(root).episodes] == ["h_stackcubes_ep1"]
```

已有的 `test_check_command_writes_json` 末尾加一行：

```python
    assert rows[0]["label"] == "存疑"
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `uv run pytest -q tests/test_cli.py`
Expected: FAIL（`unrecognized arguments: --out`、`KeyError: 'label'`、`认不出`）

- [ ] **Step 3: `load` 认出 RoboVAD**

`src/jianjin/ingest/__init__.py`：import 区加 `from jianjin.ingest.robovad import read_robovad`；`load` 里 LeRobot 判断之后加：

```python
    if (path / "annotations.json").exists() and (path / "proprioception").is_dir():
        return read_robovad(path)
```

报错信息改成「既不是 LeRobot v3（没有 meta/info.json）、RoboVAD（没有 annotations.json 和 proprioception/），也没有 HDF5 文件」。

- [ ] **Step 4: 改写 `_cmd_check`**

```python
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
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细写到 {args.json}")
    if args.out:
        html = render(verdicts, res.episodes, title=Path(args.path).name, judges=judges,
                      failures=res.failures, generated=datetime.now().strftime("%Y-%m-%d %H:%M"))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(html, encoding="utf-8")
        print(f"报告写到 {args.out}")
    return 0
```

文件顶部的 `from jianjin.checks import run_checks` 不再用到，删掉。`check` 子命令加参数：

```python
    c.add_argument("--out", help="把报告写成一个 HTML 文件")
    c.add_argument("--judges", default="", help="读哪些视频评判的缓存，逗号分隔，如 robometer,topreward")
    c.add_argument("--cache", default="cache")
    c.add_argument("--camera", default=None, help="视频评判用的那一路相机；不写就用第一路")
```

`--json` 的 help 改成「把每条示范的判定和发现写成 JSON」。

- [ ] **Step 5: 跑全部测试**

Run: `uv run pytest -q`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
git add src/jianjin/ingest/__init__.py src/jianjin/cli.py tests/test_cli.py
git commit -m "feat: jianjin check 出判定和网页报告，可读视频评判缓存；load 认出 RoboVAD"
```

---

### Task 7：真数据上出报告，写回设计文档

**Files:**
- Modify: `.gitignore`（加 `reports/`）
- Modify: `docs/superpowers/specs/2026-10-02-jianjin-design.md`（4.3、4.5、第 6 节）
- Modify: `.hackertrip/STATUS.md`

- [ ] **Step 1: 五个考卷 A 数据集各出一份报告**

`.gitignore` 加一行 `reports/`。然后：

```bash
uv run jianjin check data/lerobot__svla_so100_pickplace --judges robometer --camera observation.images.top --out reports/svla_so100_pickplace.html --json reports/svla_so100_pickplace.json
uv run jianjin check data/lerobot__svla_so100_stacking --judges robometer --camera observation.images.top --out reports/svla_so100_stacking.html --json reports/svla_so100_stacking.json
uv run jianjin check data/lerobot__svla_so100_sorting --judges robometer --camera observation.images.top --out reports/svla_so100_sorting.html --json reports/svla_so100_sorting.json
uv run jianjin check data/lerobot__svla_so101_pickplace --judges robometer --camera observation.images.up --out reports/svla_so101_pickplace.html --json reports/svla_so101_pickplace.json
uv run jianjin check data/CoRL2026-CSI__SO101-DualArm-teleop_fold_towel_100epi --judges robometer --camera observation.images.left_top --out reports/fold_towel.html --json reports/fold_towel.json
```

Expected: 每个都打印合格、存疑、作废的条数；`svla_so100_stacking` 里 ep 20、37、38 判作废（示范过短）。记下每个数据集存疑、作废的条数和原因分布：这些是公开的成功示范，存疑率就是「好数据被标存疑」的上限估计（其中也可能有真问题）。

- [ ] **Step 2: RoboVAD 出一份报告**

```bash
uv run jianjin check data/robovad --judges robometer --camera left --out reports/robovad.html --json reports/robovad.json
```

Expected: 1,078 条；记下合格、存疑、作废，以及有异常标注和没有的示范各自的判定分布。

- [ ] **Step 3: 在浏览器里看报告**

用内置浏览器打开 `reports/svla_so100_stacking.html`，看：漏斗数字和命令行打印的一致；作废卡片排在最前、依据写清楚了；进度曲线和色块画对了；窄屏（375 宽）下不横向滚动；深色模式下颜色看得清。看到的问题回到 Task 5 的代码改，改完重新生成快照（`JIANJIN_UPDATE_SNAPSHOTS=1 uv run pytest -q tests/test_report.py`）并重跑 Step 1。

- [ ] **Step 4: 写回设计文档**

- 4.3：「同一处」的新定义和理由（整条示范的发现和什么都重叠）；现在只有 Robometer 一个视频来源，关节规则和 Robometer 报的原因没有交集，作废实际上只来自硬规则和「两段都局部、时间重叠」，等 TOPReward 的阈值定了才有「两个视频模型都报指令不符」这条路。
- 4.5：命令行按实际写：`jianjin check <路径> [--out 报告.html] [--json 明细.json] [--judges robometer,topreward] [--cache cache] [--camera 相机]`。
- 第 6 节考卷 B：加「第四批结果：按示范计（进度回落）」，train 定阈值的数字、test 跑一次的数字，带分母和区间；能说的、不能说的。
- 第 6 节或第 4 节：五个公开数据集和 RoboVAD 上的判定分布（Step 1、2 记下的数），说明这是在成功示范上的存疑率。

- [ ] **Step 5: 更新 STATUS，提交**

`.hackertrip/STATUS.md` 头部 `next` 改成下一步（TOPReward 阈值、演示视频素材），追加一行日志。

```bash
git add .gitignore docs/superpowers/specs/2026-10-02-jianjin-design.md
git commit -m "docs: 判定和报告在真数据上的结果；4.3 同一处问题的定义"
```

---

## 计划之后

- TOPReward 的缓存回来后：`jianjin eval a-swap --judge topreward --split dev` 选分数、定阈值，填进 `WRONG_TASK`；`jianjin eval b-episode --judge topreward --split train` 定 `PROGRESS_DROP`；各自 test 跑一次。之后「两个视频模型都报」这条作废的路才通。
- 报告里加关键帧缩略图（要本机有视频），给路演演示用。
