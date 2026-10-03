# 第四周 b：「空录」规则 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 抓出手臂整条几乎没动的示范（空录，或者录的是人在复位），并让它和视频的「指令不符」互相印证。

**Architecture:** 关节批次规则 `empty_recording`：算各关节离开头那一帧最远走了几个活动幅度（活动幅度按同任务那批算），和同任务中位数比，不到 0.2 就报。原因新开一类「空录」，覆盖整条示范。判定里，两个来源都报整条的「空录」和「指令不符」，算指向同一处问题。

**Tech Stack:** Python 3.12、numpy、pytest。

---

## 背景

核视频（2026-10-03，第四周 a）看到叠方块有两条真问题，关节规则一条也没抓到：

- ep39：14 秒里手臂没动，最后一帧伸进来一只人手，是空录。
- ep55：开头方块已经叠好（上一条留下的），一只人手进来把它们分开，录的是复位。

「停顿过长」抓不到它们，原因有两个：速度按本条示范自己的活动幅度换算，整条不动时幅度只剩噪声；开头、结尾的静止本来就不算。

开发集（考卷 A dev）上「离开头姿态最远走了几个活动幅度，除以同任务中位数」：

| 数据集 | 最小的几条 |
|---|---|
| SO-100 放方块（50 条） | 0.865、0.866、0.892……全都在 0.86 以上 |
| SO-100 叠方块（56 条） | ep20、55、37、38 是 0.000，ep39 是 0.001；接下来是 ep47 的 0.756，其余在 0.86 以上 |

ep20、37、38 是已经被「示范过短」判作废的三条，新规则不重复报。空录和正常示范之间空着一大段，阈值取 0.2，离两边都远。

原因另开一类「空录」，不并进「没做完」：一是数采场关心空录、录成复位这类问题，值得单独数；二是考卷 A 按「没做完」计数，混进来会改动它的成绩。

## 文件

- 改 `src/jianjin/types.py`：`Cause` 加 `EMPTY = "空录"`，排在 `INCOMPLETE` 后面。
- 改 `src/jianjin/checks/config.py`：加 `empty_max`。
- 改 `src/jianjin/checks/signal.py`：加 `excursion`。
- 改 `src/jianjin/checks/batch.py`：加 `empty_recording`。
- 改 `src/jianjin/checks/runner.py`：登记。
- 改 `src/jianjin/diagnose.py`：整条的「空录」和「指令不符」算同一处问题。
- 测试：`tests/test_signal.py`、`tests/test_batch.py`、`tests/test_diagnose.py`。
- 文档：设计文档 4.3、第 5 节、「整条判定」一节；`.hackertrip/STATUS.md`。

---

### Task 1: 「空录」规则

**Files:**
- Modify: `src/jianjin/types.py`、`src/jianjin/checks/config.py`、`src/jianjin/checks/signal.py`、`src/jianjin/checks/batch.py`、`src/jianjin/checks/runner.py`
- Test: `tests/test_signal.py`、`tests/test_batch.py`

- [ ] **Step 1: 写失败的测试**

加到 `tests/test_signal.py` 末尾：

```python
from jianjin.checks.signal import excursion  # noqa: E402


def test_excursion_is_farthest_any_joint_got_from_start():
    s = np.stack([np.array([0.0, 0.5, 0.2]), np.array([1.0, 1.0, 3.0])], axis=1)
    scale = np.array([1.0, 2.0])
    assert excursion(s, scale) == pytest.approx(1.0)  # 第二个关节走了 2，按幅度 2 算是 1
    assert excursion(s[:1], scale) == 0.0
```

（`tests/test_signal.py` 顶部没有 `import pytest` 的话一起加上。）

加到 `tests/test_batch.py` 末尾：

```python
from jianjin.checks.batch import empty_recording  # noqa: E402


def test_arm_that_never_moved_is_empty_recording():
    eps = [make_episode(there(), ep_id=f"ok{i}") for i in range(5)]
    still = np.full(60, 0.3) + np.random.default_rng(0).normal(0, 1e-4, 60)  # 只有读数噪声
    eps.append(make_episode(still, ep_id="still"))
    found, skipped = empty_recording(eps, CFG)
    assert skipped == {} and set(found) == {"still"}
    (f,) = found["still"]
    assert f.cause is Cause.EMPTY and not f.hard and (f.t0, f.t1) == (0.0, eps[-1].duration)
    assert "手臂几乎没动" in f.evidence and f.score < 0.01


def test_partial_motion_is_not_empty():
    eps = [make_episode(there(), ep_id=f"ok{i}") for i in range(5)]
    eps.append(make_episode(there()[:30], ep_id="half"))  # 走了一半，归「没做完」管
    assert empty_recording(eps, CFG)[0] == {}


def test_too_short_left_to_hard_rule():
    eps = [make_episode(there(), ep_id=f"ok{i}") for i in range(5)]
    eps.append(make_episode(np.zeros(10), ep_id="tiny"))  # 0.33 秒，「示范过短」已经判了
    assert empty_recording(eps, CFG)[0] == {}
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `uv run pytest tests/test_signal.py tests/test_batch.py -q`
Expected: FAIL，`ImportError: cannot import name 'excursion'`

- [ ] **Step 3: 实现**

`src/jianjin/types.py` 的 `Cause`，在 `INCOMPLETE` 后面加：

```python
    EMPTY = "空录"
```

`src/jianjin/checks/config.py` 的 `CheckConfig`，在 `end_pose_max` 后面加：

```python
    empty_max: float = 0.2  # 离开头姿态最远不到同任务中位数的这么多，算空录
```

`src/jianjin/checks/signal.py`，在 `path_length` 后面加：

```python
def excursion(state: np.ndarray, scale: np.ndarray) -> float:
    """各关节离开头那一帧最远走了几个活动幅度，取走得最远的那个关节。"""
    if len(state) < 2:
        return 0.0
    d = np.abs(state - state[0]) / scale
    return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
```

`src/jianjin/checks/batch.py`：导入改成 `from jianjin.checks.signal import excursion, joint_scale, path_length`，文件末尾加：

```python
def empty_recording(
    episodes: list[Episode], cfg: CheckConfig, reference: list[Episode] | None = None
) -> tuple[Found, Skipped]:
    """手臂整条几乎没动：离开头姿态最远只到同任务中位数的一小部分，录下的不是示范（空录，或者录的是人在复位）。

    「停顿过长」抓不到这种：它按本条示范自己的活动幅度换算速度，而且不算开头、结尾的静止。
    2026-10-03 开发集上，空录和复位是 0.001 以下，正常示范在 0.75 以上，阈值取 0.2。
    """

    def summarize(group):
        scale = joint_scale(np.concatenate([e.state for e in group]))
        return scale, float(np.median([excursion(e.state, scale) for e in group]))

    def judge(ep, stat, n):
        scale, med = stat
        if med <= 0 or ep.duration < cfg.min_duration_s:  # 过短的已经由「示范过短」判了
            return None
        ratio = excursion(ep.state, scale) / med
        if ratio >= cfg.empty_max:
            return None
        return Finding(
            check="empty_recording",
            cause=Cause.EMPTY,
            t0=0.0,
            t1=ep.duration,
            score=float(ratio),
            evidence=f"手臂几乎没动：离开头姿态最远只到同任务 {n} 条示范中位数的 {ratio:.0%}",
        )

    return _compare(episodes, cfg, reference, summarize, judge)
```

`src/jianjin/checks/runner.py`：导入改成 `from jianjin.checks.batch import empty_recording, end_pose, extra_motion`，`BATCH_CHECKS` 末尾加 `"empty_recording": empty_recording,`。

- [ ] **Step 4: 跑测试**

Run: `uv run pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/jianjin/types.py src/jianjin/checks tests/test_signal.py tests/test_batch.py
git commit -m "feat: 「空录」规则：手臂整条几乎没动（和同任务比离开头姿态多远）"
```

---

### Task 2: 整条的「空录」和「指令不符」互相印证

**Files:**
- Modify: `src/jianjin/diagnose.py`
- Test: `tests/test_diagnose.py`

- [ ] **Step 1: 写失败的测试**

加到 `tests/test_diagnose.py` 末尾：

```python
def test_empty_recording_and_wrong_task_agree():
    empty = f(Cause.EMPTY, 0, DUR)
    wrong = f(Cause.WRONG_TASK, 0, DUR, source="robometer")
    label, reasons = label_of([empty, wrong], DUR)
    assert label == Label.REJECT
    assert reasons == ["关节报「空录」、Robometer 报「指令不符」，说的都是整条没做指令里的任务"]


def test_empty_recording_needs_the_right_partner():
    empty = f(Cause.EMPTY, 0, DUR)
    assert label_of([empty], DUR)[0] == Label.DOUBT
    assert label_of([empty, f(Cause.DATA_ERROR, 0, DUR, source="meta")], DUR)[0] == Label.DOUBT
    part = f(Cause.EMPTY, 0, 5)  # 不是整条的不算
    assert label_of([part, f(Cause.WRONG_TASK, 0, DUR, source="robometer")], DUR)[0] == Label.DOUBT
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `uv run pytest tests/test_diagnose.py -q`
Expected: FAIL，第一个新测试判成「存疑」

- [ ] **Step 3: 实现**

`src/jianjin/diagnose.py`：导入加 `Cause`（`from jianjin.types import Cause, Episode, Finding, source_name, spaced`）；模块 docstring 第一条「作废」后面加一句「整条的「空录」和整条的「指令不符」算同一处问题（2026-10-03 加）。」；`_agree` 前面加：

```python
# 原因不同、但说的是同一件事的：两个来源各报一条，而且都覆盖整条示范，就算指向同一处问题。
# 关节的「空录」（手臂几乎没动）和视频的「指令不符」（没看到在做这个任务），都是说这条没做指令里的任务。
SAME_PROBLEM = {frozenset({Cause.EMPTY, Cause.WRONG_TASK}): "整条没做指令里的任务"}
```

`_agree` 换成：

```python
def _agree(a: Finding, b: Finding, duration: float) -> bool:
    if a.source == b.source or not overlaps((a.t0, a.t1), (b.t0, b.t1)):
        return False
    whole_a, whole_b = _whole(a, duration), _whole(b, duration)
    if whole_a and whole_b:
        return a.cause == b.cause or frozenset({a.cause, b.cause}) in SAME_PROBLEM
    return a.cause == b.cause or not (whole_a or whole_b)
```

`label_of` 里拼理由的地方，在 `if a.cause == b.cause:` 和 `else:` 之间加一支：

```python
        elif frozenset({a.cause, b.cause}) in SAME_PROBLEM:
            agree.append(f"{source_name(a.source)}报「{a.cause.value}」、{source_name(b.source)}报「{b.cause.value}」，"
                         f"说的都是{SAME_PROBLEM[frozenset({a.cause, b.cause})]}")
```

- [ ] **Step 4: 跑测试**

Run: `uv run pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/jianjin/diagnose.py tests/test_diagnose.py
git commit -m "feat: 整条的「空录」和「指令不符」两个来源都报，判作废"
```

---

### Task 3: 在全部数据上跑，写进文档

**Files:**
- Modify: `docs/superpowers/specs/2026-10-02-jianjin-design.md`、`.hackertrip/STATUS.md`

- [ ] **Step 1: 考卷 A 开发集的成绩不该变**

Run: `uv run jianjin eval a --split dev --out <scratchpad>/exam_a_dev_check.json`
Expected：和 `results/exam_a_dev.json` 的检出、原样报出一样（新原因不在考卷 A 的计数里）。不覆盖 `results/` 里的文件。

- [ ] **Step 2: 重跑六份报告**（命令同第四周 a 计划 Task 5 Step 1）

Expected：叠方块 ep39、ep55 判作废（关节「空录」加 Robometer「指令不符」）；放方块 0 条空录。分拣、乐高块、叠毛巾、RoboVAD 是这条规则第一次在上面跑：报出几条就记几条。叠毛巾报出的，用本机视频看是不是真没动。

- [ ] **Step 3: 写进设计文档**

- 4.3：「实现时的出入」加一条：整条的「空录」和「指令不符」算同一处问题。
- 第 5 节：表格加一行「空录 | 手臂整条几乎没动（和同任务比离开头姿态多远），录下的不是示范」，并注明 2026-10-03 因叠方块 ep39、55 加。
- 「整条判定」一节：表格换成重跑后的数；「核视频」里 ep39、55 那两行改成判作废；去掉「关节规则没抓到」那条，改写成补了「空录」规则。

- [ ] **Step 4: STATUS 和 Commit**

STATUS 头部 `next` 去掉「补空录规则」那一项；日志加一行。

```bash
git add docs/superpowers/specs/2026-10-02-jianjin-design.md
git commit -m "docs: 「空录」规则和在全部数据上的结果"
```
