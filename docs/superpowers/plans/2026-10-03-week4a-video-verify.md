# 第四周 a：核视频以后的修正 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 2026-10-03 看视频核出来的问题修掉：新增「视频和关节时长对不上」检查；「进度回落」不算手臂回原位那段，阈值重定；报告里给存疑、作废的示范配关键帧。

**Architecture:** 时长检查只读元数据，放进 `checks/integrity.py`，来源记 `meta`。视频规则先把曲线截到关节数据有的时间里，「进度回落」再截掉回原位那段（按关节离开头姿态的距离定），考卷 B 按示范计用同一个函数。关键帧用 PyAV 按时刻取帧、编成 JPEG，以 data URI 嵌进报告，报告仍是离线单文件。

**Tech Stack:** Python 3.12、numpy、PyAV（读帧、编 JPEG）、pytest。

---

## 背景：核视频看到了什么（2026-10-03）

本地下了 `lerobot/svla_so100_stacking` 的 top 相机（404 MB）和叠毛巾的 left_top 相机第一个文件（208 MB），抽帧看：

- 叠方块报的 12 条「物体掉落」全是误报，方块都叠好了。11 条是手臂收回以后进度往下掉：从顶上看，红块叠得越正，蓝块被挡得越严，模型认不出叠好了。ep28 是手臂在方块上方调了 9 秒，挡住了方块。
- 叠毛巾 ep10（唯一一条作废）也是误报：毛巾叠好了，Robometer 在两只手臂收回时进度往下掉；关节的「没做完」是左手腕转角和别的示范不同。抽看 ep14、ep62 同样是叠好以后收手时掉的。
- 新发现的真问题：叠毛巾 9 条示范的视频比关节数据长（3 路相机一样长），ep83 多 8.1 秒，18.3 秒以后画面里毛巾又是平铺的，像是录进了复位。另外四个数据集视频和关节一样长（差 0）。

改法在选择用的数据上试过（RoboVAD train、考卷 A 开发集），没碰测试集：

| 回原位从哪算起（离开头姿态的距离，最后一次不小于全程最远的几倍） | train 上被去掉那段里开始的标注异常 | train 检出（正常报出 ≤5%） | train 阈值 | 开发集叠方块原样报出 | 开发集放方块原样报出 |
|---|---|---|---|---|---|
| 不去掉（现状） | — | 17/60 | 0.345 | 12/53 | 0/50 |
| 0.6 | 0/129 | 23/60 | 0.239 | 4/53 | 0/50 |
| 0.7 | 0/129 | 23/60 | 0.239 | 3/53 | 0/50 |
| 0.8 | 2/129 | 23/60 | 0.239 | 2/53 | 0/50 |
| 0.9 | 8/129 | 22/60 | 0.239 | — | — |

选法：被去掉的那段里不能有标注异常的开头；满足的里头取开发集原样报出最少的，即 0.7。

不在这份计划里：

- 「没做完」在叠毛巾上被手腕转角误报（11/100）。设计文档早写了要另找双臂开发集才能改，这次不动。
- ep28 那种手臂长时间挡住物体造成的中途回落，这次的改法管不到。
- 考卷 A 换描述的评测（`eval/task_swap.py`）不改；`wrong_task` 截曲线对开发集没有影响（视频和关节一样长）。

## 文件

- 改 `src/jianjin/types.py`：来源名加 `meta`（元数据）。
- 改 `src/jianjin/checks/config.py`：加 `video_slack_frames`。
- 改 `src/jianjin/checks/integrity.py`：加 `video_length`。
- 改 `src/jianjin/checks/runner.py`：登记 `video_length`。
- 改 `src/jianjin/checks/video.py`：`RETURN_FROM`、`return_start`、`clip`、`rule_drawdown`；`wrong_task`、`progress_drop` 用上；`PROGRESS_DROP` 换新阈值。
- 改 `src/jianjin/eval/exam_b_episode.py`：用 `rule_drawdown`。
- 改 `src/jianjin/video.py`：加 `frames_at`。
- 改 `src/jianjin/judges/cache.py`：`_pick_camera` 改名 `pick_camera`。
- 新建 `src/jianjin/keyframes.py`：`keyframe_times`、`jpeg_data_uri`、`shots_for`。
- 改 `src/jianjin/report.py`：页脚、进度曲线只画关节数据那段、关键帧。
- 改 `src/jianjin/cli.py`：`check` 默认配关键帧，`--no-frames` 关掉。
- 测试：`tests/test_integrity.py`、`tests/test_video_checks.py`、`tests/test_exam_b_episode.py`、`tests/test_video.py`、`tests/test_keyframes.py`（新）、`tests/test_report.py`、`tests/test_cli.py`、`tests/snapshots/report_small.html`。
- 文档：设计文档、`.hackertrip/STATUS.md`。

---

### Task 1: 视频和关节时长对不上

**Files:**
- Modify: `src/jianjin/types.py`、`src/jianjin/checks/config.py`、`src/jianjin/checks/integrity.py`、`src/jianjin/checks/runner.py`、`src/jianjin/report.py`
- Test: `tests/test_integrity.py`、`tests/snapshots/report_small.html`

- [ ] **Step 1: 写失败的测试**

加到 `tests/test_integrity.py` 末尾：

```python
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from jianjin.checks.config import NotApplicable  # noqa: E402
from jianjin.checks.integrity import video_length  # noqa: E402
from jianjin.types import VideoRef  # noqa: E402


def with_videos(ep, **lengths):
    """lengths：相机名 -> 元数据里的视频时长（秒），None 是没写。"""
    ep.cameras = {cam: VideoRef(Path(f"{cam}.mp4"), 10.0, None if n is None else 10.0 + n)
                  for cam, n in lengths.items()}
    return ep


def test_video_as_long_as_joints_is_fine():
    ep = with_videos(make_episode(ramp()), top=3.0, wrist=3.0 + 1.0 / 30)  # 差一帧以内不算
    assert video_length(ep, CFG) == []


def test_video_longer_than_joints_needs_a_look():
    ep = with_videos(make_episode(ramp()), top=11.1, wrist=11.1)
    (f,) = video_length(ep, CFG)
    assert f.cause is Cause.DATA_ERROR and not f.hard and f.source == "meta"
    assert (f.t0, f.t1) == (0.0, ep.duration)
    assert "长 8.1 秒" in f.evidence and "2 路相机都这样" in f.evidence


def test_video_shorter_than_joints_is_hard():
    ep = with_videos(make_episode(ramp()), top=2.0, wrist=3.0)
    (f,) = video_length(ep, CFG)
    assert f.hard and "短 1.0 秒" in f.evidence and "top" in f.evidence
    assert f.t0 == pytest.approx(2.0) and f.t1 == pytest.approx(3.0)


def test_unknown_video_length_is_not_checked():
    with pytest.raises(NotApplicable):
        video_length(with_videos(make_episode(ramp()), left=None), CFG)


def test_no_video_nothing_to_compare():
    assert video_length(make_episode(ramp()), CFG) == []


def test_runner_runs_video_length():
    from jianjin.checks import run_checks

    (r,) = run_checks([with_videos(make_episode(ramp()), top=11.1)])
    assert [f.check for f in r.findings] == ["video_length"]
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `uv run pytest tests/test_integrity.py -q`
Expected: FAIL，`ImportError: cannot import name 'video_length'`

- [ ] **Step 3: 实现**

`src/jianjin/types.py`：

```python
SOURCE_NAMES = {"joints": "关节", "meta": "元数据", "robometer": "Robometer", "topreward": "TOPReward"}
```

`src/jianjin/checks/config.py` 的 `CheckConfig` 末尾加：

```python
    video_slack_frames: float = 1.5  # 视频时长和关节数据差超过这么多帧，算两路对不上
```

`src/jianjin/checks/integrity.py`：导入改成 `from jianjin.checks.config import CheckConfig, NotApplicable`，末尾加：

```python
def _cams(bad: dict[str, float], known: dict[str, float]) -> str:
    if len(bad) == len(known) and len(known) > 1:
        return f"{len(known)} 路相机都这样"
    return "、".join(c.split(".")[-1] for c in sorted(bad))


def video_length(ep: Episode, cfg: CheckConfig) -> list[Finding]:
    """元数据里每路视频的时长要和关节数据一样长。只读元数据，不解码视频。

    视频短：最后一段关节数据没有画面，训练时取不到帧，判作废。
    视频长：多出来的画面对不上关节数据，要人看（叠毛巾 ep83 多出的 8 秒是复位画面）。
    """
    if not ep.cameras:
        return []
    known = {cam: ref.to_s - ref.from_s for cam, ref in ep.cameras.items() if ref.to_s is not None}
    if not known:
        raise NotApplicable("元数据里没写视频时长，没法和关节数据对")
    slack = cfg.video_slack_frames / ep.fps
    short = {c: ep.duration - n for c, n in known.items() if n < ep.duration - slack}
    long = {c: n - ep.duration for c, n in known.items() if n > ep.duration + slack}
    out = []
    if short:
        gap = max(short.values())
        out.append(Finding(
            check="video_length", cause=Cause.DATA_ERROR, t0=max(0.0, ep.duration - gap), t1=ep.duration,
            score=gap, evidence=f"视频比关节数据短 {gap:.1f} 秒（{_cams(short, known)}），最后这段关节数据没有画面",
            hard=True, source="meta",
        ))
    if long:
        extra = max(long.values())
        out.append(Finding(
            check="video_length", cause=Cause.DATA_ERROR, t0=0.0, t1=ep.duration, score=extra,
            evidence=f"视频比关节数据长 {extra:.1f} 秒（{_cams(long, known)}），两路时长对不上，要看多出来的画面是什么",
            source="meta",
        ))
    return out
```

`src/jianjin/checks/runner.py`：导入加 `video_length`，`EPISODE_CHECKS` 在 `timestamp_gaps` 后面加一项 `"video_length": video_length,`。

`src/jianjin/report.py` 的 `FOOTER`：「客观硬规则触发（读数缺失、丢帧、示范过短）」改成「客观硬规则触发（读数缺失、丢帧、示范过短、视频比关节数据短）」。

- [ ] **Step 4: 跑测试**

Run: `uv run pytest tests/test_integrity.py -q`
Expected: PASS

Run: `uv run pytest -q`
Expected: 只有 `test_report.py::test_snapshot` 失败（页脚改了）。确认 diff 只有页脚一句后：
`JIANJIN_UPDATE_SNAPSHOTS=1 uv run pytest tests/test_report.py -q`，再跑 `uv run pytest -q` 全过。

- [ ] **Step 5: Commit**

```bash
git add src/jianjin/types.py src/jianjin/checks tests/test_integrity.py src/jianjin/report.py tests/snapshots/report_small.html
git commit -m "feat: 视频和关节数据时长对不上的检查（只读元数据）"
```

---

### Task 2: 视频规则只看关节数据那段，「进度回落」去掉回原位那段

**Files:**
- Modify: `src/jianjin/checks/video.py`、`src/jianjin/eval/exam_b_episode.py`、`src/jianjin/report.py`
- Test: `tests/test_video_checks.py`、`tests/test_exam_b_episode.py`、`tests/test_report.py`

- [ ] **Step 1: 写失败的测试**

加到 `tests/test_video_checks.py` 末尾：

```python
from jianjin.checks.video import clip, return_start  # noqa: E402


def out_and_back():
    """0 到 1 秒走到远处，1 到 2 秒停住，2 到 3 秒走回开头的姿态；夹爪不动。"""
    x = np.concatenate([np.linspace(0, 1, 30), np.ones(30), np.linspace(1, 0, 30)])
    return make_episode(np.stack([x, x, np.zeros(90)], axis=1), names=["j0", "j1", "gripper"])


def test_return_start_is_last_time_far_from_start_pose():
    # 离开头姿态的距离正比于 x；最后一次不小于最远的 0.7 倍是第 68 帧（x = 1 - 8/29）
    assert return_start(out_and_back()) == pytest.approx(68 / 30)


def test_return_start_without_coming_back_is_episode_end():
    ep = make_episode(np.tile(np.linspace(0, 1, 90)[:, None], (1, 2)))  # 一路走远，不回来
    assert return_start(ep) == ep.duration


def test_return_start_ignores_gripper():
    ep = make_episode(np.stack([np.zeros(90), np.linspace(0, 1, 90)], axis=1), names=["j0", "gripper"])
    assert return_start(ep) == ep.duration


def test_clip_keeps_points_up_to_end():
    c = Curve("robometer", "v", [0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3])
    k = clip(c, 2.0)
    assert (k.times, k.progress, k.success) == ([0, 1, 2], [0.1, 0.2, 0.3], [0.0, 0.1, 0.2])
    assert clip(c, -1.0).times == [0]


def test_progress_drop_ignores_fall_while_returning_home():
    ep = out_and_back()
    home = Curve("robometer", "v", [0, 1, 2, 2.5, 3], [0.1, 0.5, 0.9, 0.4, 0.2])  # 2.27 秒开始回原位以后才掉
    early = Curve("robometer", "v", [0, 1, 1.5, 2, 3], [0.1, 0.9, 0.3, 0.5, 0.2])  # 还在远处就掉了
    found, _ = progress_drop([ep], {ep.id: home}, "robometer", CheckConfig(), threshold=0.3)
    assert not found
    found, _ = progress_drop([ep], {ep.id: early}, "robometer", CheckConfig(), threshold=0.3)
    assert (found[ep.id][0].t0, found[ep.id][0].t1) == (1.0, 1.5)


def test_video_rules_ignore_curve_past_joint_data():
    eps = batch()  # 每条 10 秒，关节不动
    curves = {ep.id: curve(0.9) for ep in eps}
    curves["e0"] = Curve("robometer", "v", [0.0, 5.0, 10.0, 14.0], [0.1, 0.2, 0.3, 0.95])  # 高点在关节数据以外
    found, _ = wrong_task(eps, curves, "robometer", CheckConfig())
    assert list(found) == ["e0"]
    late_drop = {"e1": Curve("robometer", "v", [0.0, 5.0, 10.0, 12.0], [0.1, 0.5, 0.9, 0.1])}
    found, _ = progress_drop([eps[1]], late_drop, "robometer", CheckConfig(), threshold=0.3)
    assert not found
```

加到 `tests/test_exam_b_episode.py` 末尾：

```python
def test_fall_while_returning_home_does_not_count(tmp_path):
    x = np.concatenate([np.linspace(0, 1, 50), np.ones(50), np.linspace(1, 0, 50)])  # 5 秒：出去、停住、回原位
    arm = np.stack([x] * 5 + [np.zeros(150)], axis=1)
    names = [f"h_stackcubes_ep{i}" for i in range(5)]
    drop = [{"start_frame": 30, "end_frame": 60, "type": "object_drop", "categories": {}}]  # 第 1 到 2 秒
    root = make_robovad(tmp_path / "rv", {n: (arm, arm) for n in names}, {"h_stackcubes_ep4": drop},
                        validation=names)
    cache = tmp_path / "cache"
    for ep in read_robovad(root).episodes:
        if ep.id == "h_stackcubes_ep4":  # 还在远处就掉了
            write_cached_curve(cache, ep, "left", [0.1, 0.9, 0.3, 0.5, 0.2], times=[0, 1, 1.5, 2, 5])
        else:  # 正常示范：约 3.8 秒开始回原位，之后进度才掉
            write_cached_curve(cache, ep, "left", [0.1, 0.5, 0.9, 0.9, 0.2], times=[0, 1, 2, 3, 5])
    out = run_episode_b(root, "validation", cache)
    assert out["threshold"] == pytest.approx(0.0)
    assert out["detected"] == 1 and out["normal_flagged"] == 0
```

加到 `tests/test_report.py` 末尾：

```python
def test_curve_drawn_only_within_joint_data():
    vs, eps = small()
    vs[2].progress = Curve("robometer", "v", [0.0, 1.5, 3.0, 5.0], [0.1, 0.5, 0.9, 0.2])  # 第 5 秒在示范以外
    html = render([vs[2]], [eps[2]], title="ds")
    assert html.count('<polyline points="') == 1
    pts = html.split('<polyline points="')[1].split('"')[0].split()
    assert len(pts) == 3
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `uv run pytest tests/test_video_checks.py tests/test_exam_b_episode.py tests/test_report.py -q`
Expected: FAIL，`ImportError: cannot import name 'clip'`；导入修好前其余新测试也跑不到

- [ ] **Step 3: 实现**

`src/jianjin/checks/video.py`，在 `drawdown` 后面加：

```python
# 回原位那段：关节（不含夹爪）离开头姿态的距离，最后一次不小于全程最远的 RETURN_FROM 倍之后。
# 手臂收回时进度模型常认不出已经做完（2026-10-03 核视频：叠方块报的 12 条、叠毛巾抽看的 3 条都是），
# 「进度回落」只看这之前。0.7 在 RoboVAD train 和考卷 A 开发集上选：去掉的那段里不能有标注异常的开头，
# 满足的取值里开发集原样报出最少（设计文档「核视频」一节）。没回到开头附近的示范，这个时刻落在结尾。
RETURN_FROM = 0.7


def return_start(ep: Episode, frac: float = RETURN_FROM) -> float:
    """手臂开始回原位的时刻（秒）；没回原位、没动过的，返回整条的时长。"""
    idx = [i for i, n in enumerate(ep.state_names) if i != ep.gripper_index and "grip" not in n.lower()]
    if not idx or len(ep.state) == 0:
        return ep.duration
    q = ep.state[:, idx]
    dist = np.nan_to_num(np.linalg.norm(q - q[0], axis=1))
    far = float(dist.max())
    if far <= 0:
        return ep.duration
    i = int(np.flatnonzero(dist >= frac * far)[-1])
    return ep.duration if i == len(dist) - 1 else float(ep.timestamps[i])


def clip(c: Curve, t_end: float) -> Curve:
    """只留 t_end 以前（含）的点；一个都不剩时留第一个。"""
    keep = [i for i, t in enumerate(c.times) if t <= t_end + 1e-6] or [0]

    def pick(xs):
        return None if xs is None else [xs[i] for i in keep]

    return Curve(c.model, c.version, pick(c.times), pick(c.progress), pick(c.success))


def rule_drawdown(ep: Episode, c: Curve) -> tuple[float, float, float, float, float]:
    """「进度回落」看的那段：关节数据有的时间里，手臂开始回原位以前。"""
    return drawdown(clip(c, min(ep.duration, return_start(ep))))
```

`wrong_task` 里两处 `fn(curves[ep.id])`、`fn(c)` 改成截到关节数据那段：

```python
    for ep in reference if reference is not None else episodes:
        if ep.id in curves:
            groups[ep.task].append(fn(clip(curves[ep.id], ep.duration)))
```

```python
        z = float(relative([fn(clip(c, ep.duration))], ref)[0])
        if z <= thr:
            continue
        value, med = 1.0 - fn(clip(c, ep.duration)), 1.0 - float(np.median(ref))
```

`progress_drop`：docstring 改成「进度突然往回掉：东西掉了，或者退回去重做。只看关节数据有的时间里、手臂开始回原位以前（见 RETURN_FROM）。不和同任务比，cfg、reference 只为和别的规则同一个签名。」；`d, t0, t1, hi, lo = drawdown(c)` 改成 `d, t0, t1, hi, lo = rule_drawdown(ep, c)`。

`src/jianjin/eval/exam_b_episode.py`：导入改成 `from jianjin.checks.video import rule_drawdown`，`d, t0, t1, _, _ = drawdown(c)` 改成 `d, t0, t1, _, _ = rule_drawdown(ep, c)`；模块 docstring 第一段末尾加「规则和 `jianjin check` 里的「进度回落」是同一个函数（去掉回原位那段）。」

`src/jianjin/report.py` 的 `_spark`：`pts` 只画示范时长以内的点：

```python
    pts = " ".join(f"{x(t):.1f},{y(p):.1f}" for t, p in zip(c.times, c.progress) if t <= duration + 1e-6)
```

- [ ] **Step 4: 跑测试**

Run: `uv run pytest -q`
Expected: 全部 PASS（快照不变：小报告里的曲线都在示范时长以内）

- [ ] **Step 5: Commit**

```bash
git add src/jianjin/checks/video.py src/jianjin/eval/exam_b_episode.py src/jianjin/report.py tests
git commit -m "feat: 视频规则只看关节数据那段；进度回落不算手臂回原位那段"
```

---

### Task 3: 「进度回落」重定阈值，test 看第二次

**Files:**
- Modify: `src/jianjin/checks/video.py`（`PROGRESS_DROP`）
- Results: `results/exam_b_episode_train.json`、`results/exam_b_episode_test.json`

- [ ] **Step 1: train 上定阈值**

Run: `uv run jianjin eval b-episode --split train`
Expected：367 条有缓存（异常 60、正常 307），阈值约 0.239，检出 23/60，正常报出 15/307，AUC 约 0.808（和计划开头的表一致）。结果写到 `results/exam_b_episode_train.json`。

- [ ] **Step 2: 阈值写进规则**

`src/jianjin/checks/video.py` 的 `PROGRESS_DROP` 换成 train 结果里的 `threshold` 原值，注释写出处：

```python
PROGRESS_DROP: dict[str, float] = {
    # 2026-10-03 改：去掉回原位那段以后，RoboVAD train 正常示范（307 条）进度回落的 95 分位
    # （results/exam_b_episode_train.json）。原来不去掉时是 0.345。
    "robometer": <train 结果里的 threshold>,
}
```

Run: `uv run pytest -q`
Expected: 全部 PASS

- [ ] **Step 3: test 跑一次（第二次看，要申明）**

Run: `uv run jianjin eval b-episode --split test --dev-result results/exam_b_episode_train.json`
Expected：629 条附近，结果写到 `results/exam_b_episode_test.json`。原来（不去掉回原位那段）是检出 73/387、正常报出 18/242、AUC 0.664。不管涨跌都照实记。

- [ ] **Step 4: Commit**

```bash
git add src/jianjin/checks/video.py results/exam_b_episode_train.json results/exam_b_episode_test.json
git commit -m "feat: 进度回落去掉回原位那段后在 RoboVAD train 上重定阈值；test 第二次看"
```

---

### Task 4: 报告里给存疑、作废的示范配关键帧

**Files:**
- Modify: `src/jianjin/video.py`、`src/jianjin/judges/cache.py`、`src/jianjin/report.py`、`src/jianjin/cli.py`
- Create: `src/jianjin/keyframes.py`
- Test: `tests/helpers.py`、`tests/test_video.py`、`tests/test_keyframes.py`、`tests/test_report.py`、`tests/test_cli.py`、`tests/snapshots/report_small.html`

- [ ] **Step 1: 写失败的测试**

`tests/helpers.py` 的 `make_mp4` 开头加一行，让它能写进 `videos/<相机>/chunk-000/` 这种还不存在的目录：

```python
    path.parent.mkdir(parents=True, exist_ok=True)
```

加到 `tests/test_video.py` 末尾：

```python
from jianjin.video import frames_at  # noqa: E402


def test_frames_at_takes_one_frame_per_time(mp4):
    imgs = frames_at(VideoRef(mp4, 4.0, 6.0), [0.0, 0.5, 1.0], max_side=32)
    assert imgs.shape == (3, 24, 32, 3)
    assert [round(float(i.mean())) for i in imgs] == pytest.approx([120, 135, 150], abs=12)  # 第 120、135、150 帧
```

新建 `tests/test_keyframes.py`：

```python
import numpy as np

from helpers import make_episode, make_lerobot_v3, make_mp4
from jianjin.diagnose import Label, Verdict
from jianjin.ingest import load
from jianjin.keyframes import jpeg_data_uri, keyframe_times, shots_for
from jianjin.types import Cause, Finding


def v_with(*spans, label=Label.DOUBT):
    fs = [Finding("x", Cause.DROP, a, b, 1.0, "e") for a, b in spans]
    return Verdict("e", label, ["r"], fs)


def test_times_are_finding_edges():
    assert keyframe_times(v_with((1.0, 2.0)), 10.0) == [1.0, 2.0]


def test_whole_episode_finding_takes_start_middle_end():
    assert keyframe_times(v_with((0.0, 10.0)), 10.0) == [0.0, 5.0, 9.95]


def test_times_clamped_merged_and_capped():
    assert keyframe_times(v_with((9.0, 12.0)), 10.0) == [9.0, 9.95]
    assert keyframe_times(v_with((1.0, 1.2)), 10.0) == [1.0]
    assert keyframe_times(v_with((0.0, 1.0), (2.0, 3.0), (4.0, 5.0)), 10.0) == [0.0, 2.0, 3.0, 5.0]


def test_jpeg_data_uri():
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    assert jpeg_data_uri(img).startswith("data:image/jpeg;base64,/9j/")


def test_shots_only_for_doubt_and_reject_with_local_video(tmp_path):
    x = np.tile(np.linspace(0, 1, 90)[:, None], (1, 6))
    ds = make_lerobot_v3(tmp_path / "d", [("pick", x), ("pick", x)])
    eps = load(ds).episodes
    make_mp4(ds / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4", n=180)
    vs = [v_with((1.0, 2.0)), v_with(label=Label.PASS)]
    vs[0].episode_id, vs[1].episode_id = eps[0].id, eps[1].id
    shots, missing = shots_for(vs, eps)
    assert list(shots) == [eps[0].id] and missing == 0
    assert [t for t, _ in shots[eps[0].id]] == [1.0, 2.0]
    shots, missing = shots_for(vs, [make_episode(x, ep_id=eps[0].id), eps[1]])  # 没有视频
    assert shots == {} and missing == 1
```

加到 `tests/test_report.py` 末尾：

```python
def test_cards_show_keyframes_when_given():
    vs, eps = small()
    shots = {"ds/ep0": [(1.0, "data:image/jpeg;base64,AAAA"), (2.5, "data:image/jpeg;base64,BBBB")]}
    html = render(vs, eps, title="ds", shots=shots)
    assert html.count('<div class="shots">') == 1
    assert '<img src="data:image/jpeg;base64,BBBB" alt="第 2.5 秒的画面"' in html
```

加到 `tests/test_cli.py` 末尾：

```python
def test_check_report_has_keyframes_when_video_is_local(tmp_path, capsys):
    from helpers import make_mp4

    ds = make_lerobot_v3(tmp_path / "d", [("pick", paused6())])
    make_mp4(ds / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4", n=300)
    out = tmp_path / "r.html"
    assert main(["check", str(ds), "--out", str(out)]) == 0
    assert '<div class="shots">' in out.read_text(encoding="utf-8")
    assert "1 条存疑或作废的示范配了关键帧" in capsys.readouterr().out
    assert main(["check", str(ds), "--out", str(out), "--no-frames"]) == 0
    assert '<div class="shots">' not in out.read_text(encoding="utf-8")
```

- [ ] **Step 2: 跑测试，确认失败**

Run: `uv run pytest tests/test_video.py tests/test_keyframes.py tests/test_report.py tests/test_cli.py -q`
Expected: FAIL，`ImportError: cannot import name 'frames_at'`、`No module named 'jianjin.keyframes'` 等

- [ ] **Step 3: 实现**

`src/jianjin/video.py` 末尾加：

```python
def frames_at(ref: VideoRef, times: list[float], max_side: int = 240) -> np.ndarray:
    """示范里这几个时刻（相对示范开头的秒数）各取一帧：不早于该时刻的第一帧；超出视频就取最后一帧。"""
    out = []
    with av.open(str(ref.path)) as c:
        s = c.streams.video[0]
        w, h = s.codec_context.width, s.codec_context.height
        scale = min(1.0, max_side / max(w, h))
        tw, th = max(2, round(w * scale / 2) * 2), max(2, round(h * scale / 2) * 2)
        for t in times:
            target = ref.from_s + t
            c.seek(int(max(target - 1.0, 0.0) / s.time_base), stream=s, backward=True)
            got = None
            for fr in c.decode(s):
                if fr.time is None:
                    continue
                got = fr
                if fr.time >= target - 1e-3:
                    break
            if got is None:
                raise ValueError(f"{ref.path} 在第 {target:.2f} 秒附近读不到帧")
            out.append(got.reformat(width=tw, height=th).to_ndarray(format="rgb24"))
    return np.stack(out)
```

`src/jianjin/judges/cache.py`：`_pick_camera` 改名 `pick_camera`（定义和文件里两处调用一起改）。

新建 `src/jianjin/keyframes.py`：

```python
"""给存疑、作废的体检单配几张画面，人看一眼就知道报得对不对。

画面在本机视频里按时刻取，编成 JPEG 嵌进报告（data URI），报告还是一个文件、断网能开。
合格的不配，报告不至于太大。
"""

from __future__ import annotations

import base64
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from jianjin.checks.config import NotApplicable
from jianjin.diagnose import Label, Verdict
from jianjin.judges.cache import pick_camera
from jianjin.types import Episode
from jianjin.video import frames_at

Shots = dict[str, list[tuple[float, str]]]  # 示范 id -> [(秒, data URI)]


def keyframe_times(v: Verdict, duration: float, k: int = 4, gap: float = 0.5) -> list[float]:
    """每条发现取起止两个时刻；整条的发现取开头、中间、结尾。相隔不到 gap 秒的并成一个，最多 k 个。"""
    end = max(duration - 0.05, 0.0)  # 最后一帧在示范结束前一点
    ts = []
    for f in v.causes:
        if f.t0 <= 0.0 and f.t1 >= duration - 1e-9:
            ts += [0.0, duration / 2, end]
        else:
            ts += [f.t0, f.t1]
    out: list[float] = []
    for t in sorted(min(max(t, 0.0), end) for t in ts):
        if not out or t - out[-1] >= gap:
            out.append(t)
    if len(out) > k:
        out = [out[round(i * (len(out) - 1) / (k - 1))] for i in range(k)]
    return out


def jpeg_data_uri(img: np.ndarray) -> str:
    cc = av.CodecContext.create("mjpeg", "w")
    cc.width, cc.height, cc.pix_fmt = img.shape[1], img.shape[0], "yuvj420p"
    cc.time_base = Fraction(1, 25)
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24").reformat(format="yuvj420p")
    data = b"".join(bytes(p) for p in [*cc.encode(frame), *cc.encode(None)])
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def shots_for(verdicts: list[Verdict], episodes: list[Episode], camera: str | None = None,
              max_side: int = 240) -> tuple[Shots, int]:
    """存疑、作废的示范各配几张关键帧。返回（关键帧，没有本地视频或读不出画面的条数）。"""
    by_id = {ep.id: ep for ep in episodes}
    shots: Shots = {}
    missing = 0
    for v in verdicts:
        if v.label == Label.PASS:
            continue
        ep = by_id[v.episode_id]
        try:
            ref = ep.cameras[pick_camera(ep, camera)]
        except NotApplicable:
            missing += 1
            continue
        times = keyframe_times(v, ep.duration)
        if not times or not Path(ref.path).exists():
            missing += 1
            continue
        try:
            imgs = frames_at(ref, times, max_side=max_side)
        except (av.error.FFmpegError, ValueError):
            missing += 1
            continue
        shots[v.episode_id] = [(t, jpeg_data_uri(img)) for t, img in zip(times, imgs)]
    return shots, missing
```

`src/jianjin/report.py`：

1. `CSS` 在 `.spark` 那几行后面加一行：

```css
.shots{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:6px;margin:10px 0 0;max-width:760px}
.shots figure{margin:0}.shots img{width:100%;height:auto;display:block;border-radius:6px;background:var(--line)}
```

2. 加 `_shots`，`_card` 多一个参数，`_cards`、`render` 把 `shots` 传下去：

```python
def _shots(shots: list[tuple[float, str]]) -> str:
    if not shots:
        return ""
    figs = "".join(f'<figure><img src="{uri}" alt="第 {t:.1f} 秒的画面" loading="lazy">'
                   f'<figcaption>第 {t:.1f} 秒</figcaption></figure>' for t, uri in shots)
    return f'<div class="shots">{figs}</div>'
```

`_card(v, ep, shots=())`：在 `parts.append(_spark(v, ep.duration))` 后面加 `parts.append(_shots(list(shots)))`。
`_cards(verdicts, episodes, shots=None)`：`_card(v, by_id[v.episode_id], (shots or {}).get(v.episode_id, ()))`。
`render(..., generated: str = "", shots=None)`：`_cards(verdicts, episodes, shots)`。

`src/jianjin/cli.py` 的 `_cmd_check`，`if args.out:` 里在 `render` 前面：

```python
        shots = None
        if not args.no_frames:
            from jianjin.keyframes import shots_for

            shots, missing = shots_for(verdicts, res.episodes, args.camera)
            print(f"  画面：{len(shots)} 条存疑或作废的示范配了关键帧"
                  + (f"，{missing} 条没有本地视频或读不出画面" if missing else ""))
```

`render(...)` 调用加 `shots=shots`。`check` 的参数加：

```python
    c.add_argument("--no-frames", action="store_true", help="报告里不配关键帧（默认有本地视频就配）")
```

- [ ] **Step 4: 跑测试**

Run: `uv run pytest -q`
Expected: 只有快照失败（CSS 多了两行）。确认 diff 只有这两行后：
`JIANJIN_UPDATE_SNAPSHOTS=1 uv run pytest tests/test_report.py -q`，再跑 `uv run pytest -q` 全过。

- [ ] **Step 5: Commit**

```bash
git add src/jianjin tests
git commit -m "feat: 报告给存疑、作废的示范配关键帧（本机有视频时）"
```

---

### Task 5: 重跑五个数据集和 RoboVAD，写进文档

**Files:**
- Modify: `docs/superpowers/specs/2026-10-02-jianjin-design.md`、`.hackertrip/STATUS.md`

- [ ] **Step 1: 重跑报告**

```bash
uv run jianjin check data/lerobot__svla_so100_pickplace --judges robometer --camera observation.images.top --out reports/svla_so100_pickplace.html --json reports/svla_so100_pickplace.json
uv run jianjin check data/lerobot__svla_so100_stacking --judges robometer --camera observation.images.top --out reports/svla_so100_stacking.html --json reports/svla_so100_stacking.json
uv run jianjin check data/lerobot__svla_so100_sorting --judges robometer --camera observation.images.top --out reports/svla_so100_sorting.html --json reports/svla_so100_sorting.json
uv run jianjin check data/lerobot__svla_so101_pickplace --judges robometer --camera observation.images.up --out reports/svla_so101_pickplace.html --json reports/svla_so101_pickplace.json
uv run jianjin check data/CoRL2026-CSI__SO101-DualArm-teleop_fold_towel_100epi --judges robometer --camera observation.images.left_top --out reports/fold_towel.html --json reports/fold_towel.json
uv run jianjin check data/robovad --judges robometer --camera left --out reports/robovad.html --json reports/robovad.json
```

Expected：叠方块「物体掉落」12 → 3 附近；叠毛巾多出「数据异常」9 条（视频比关节长），「物体掉落」大幅减少，ep10 不再作废。叠方块和叠毛巾的报告里，存疑、作废的卡片带关键帧（本机有这两路视频）；其余数据集没有本地视频，CLI 会打印没画面的条数。

- [ ] **Step 2: 抽看新报告**

在浏览器里打开 `reports/fold_towel.html` 和 `reports/svla_so100_stacking.html`，看：关键帧显示正常、手机宽度不横向滚动、深色模式可读；ep83 的卡片写「视频比关节数据长 8.1 秒」。

- [ ] **Step 3: 写进设计文档**

- 第 6 节考卷 B「第四批结果：按示范计」后面加「第五批：去掉回原位那段（2026-10-03）」：train 选取的表（本计划开头那张）、选法、新阈值、test 第二次看的结果（申明是第二次看）。
- 「整条判定在真数据上的样子」那节：加「核视频（2026-10-03）」小节，写本计划开头「背景」里的发现；表格换成重跑后的数，旧数留一行对比；已知问题保留「没做完」在叠毛巾上的手腕误报、ep28 那种中途遮挡。
- 4.3：硬规则加「视频比关节数据短」；「视频比关节数据长」记存疑，来源是元数据。

- [ ] **Step 4: STATUS 和 Commit**

`.hackertrip/STATUS.md` 头部 `next` 改成：「TOPReward 笔记本等用户跑，回来在 dev/train 上定阈值；「没做完」的手腕误报要找双臂开发集；考卷 C 等用户同意 RoboMIND 条款」；日志加一行本周 a 的结果。

```bash
git add docs/superpowers/specs/2026-10-02-jianjin-design.md
git commit -m "docs: 核视频的结果；视频时长检查；进度回落去掉回原位那段后的成绩"
```

（`.hackertrip/` 在 `.gitignore` 里，不提交。）
