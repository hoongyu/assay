# 「动作不流畅」「速度过快」实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 两条同任务比较的关节规则「动作不流畅」「速度过快」，在 RoboMIND 失败示范的真标签上选和量（考卷 C），顺带量二次抓取。

**Architecture:** 读取器补 RoboMIND 失败示范的三种关节布局；`jianjin fetch robomind` 按字节范围只读关节；`checks/signal.py` 加停顿、动作段、jerk 的基本运算；`checks/batch.py` 的 `motion_scores` 一次算出全部候选，规则 `jerky`、`too_fast` 只用选中的那个；`eval/exam_c.py` 在成功示范开发集上定阈值、在 RoboMIND 开发集上选、测试集跑一次。

**Tech Stack:** Python 3.12、numpy、h5py、huggingface_hub（可选依赖 hub）、pytest。

设计见 `docs/superpowers/specs/2026-10-05-motion-quality-design.md`。分支 `motion-quality-rules`。

---

### Task 1：读取器认 RoboMIND 失败示范的三种布局

**Files:** Modify `src/jianjin/ingest/robomind.py`；Test `tests/test_ingest_robomind.py`；`tests/helpers.py` 加造这三种文件的函数。

- [ ] 测试先写：
  - 真机单臂 `puppet/arm_joint_position`（T×6）加 `puppet/hand_joint_position`（T×1）：state 7 列，最后一列叫 gripper，gripper_index 6；master 同构当 action。
  - 双臂 16 列、手 2 列且手就是第 7、15 列：不重复拼，state 16 列，第 7、15 列叫 left_gripper、right_gripper。
  - 天工仿真 `tiangong/...`：左右臂 7、7，手 12、12，帧数差一帧截到最短；名字 left_j*、right_j*、left_hand_j*、right_hand_j*；gripper_index 为 None；action 为 None；fps 按 `sim_time_seq` 的间隔中位估。
  - 缺键的记读取失败。
- [ ] 跑测试看它们失败；实现；全绿；提交。

### Task 2：`jianjin fetch robomind`

**Files:** Create `src/jianjin/eval/robomind_fetch.py`；Modify `src/jianjin/cli.py`；Test `tests/test_robomind_fetch.py`。

- [ ] 测试先写：`copy_joints(读得到的文件对象, 目标路径, 来源, 来源字节数, 上限)` 只拷非图像的数据集和属性，图像留同名空数据集并记原形状；读过上限抛 `TooBig`；`CappedReader` 数的是实际读走的字节。
- [ ] 实现（列文件和多进程下载不进单元测试）；CLI 加 `fetch robomind`；提交。

### Task 3：信号运算

**Files:** Modify `src/jianjin/checks/signal.py`；Test `tests/test_signal.py`。

- [ ] 测试先写：`arm_channels(names)` 去掉 gripper、hand；`stops(v, still, min_frames)` 只数中途的；`bursts(v, lo, hi, min_frames)` 数升到高档的次数、一闪不算；`jerk(state, fps, scale)` 对匀加速轨迹近似 0、对抖动的大；`level_changes` 原有测试保持全绿（和 bursts 共用一个换档函数）。
- [ ] 实现；提交。

### Task 4：候选分数 `motion_scores`

**Files:** Modify `src/jianjin/checks/batch.py`、`src/jianjin/checks/config.py`；Test `tests/test_batch.py`。

- [ ] 测试先写：同任务五条正常加一条中途停停走走的，「中途停顿」分数最高；加一条整体快一倍的，「最快速度」「平均速度」「时长（快）」都约为 2；参照组不够 5 条的记没法比；没有能动的手臂关节的记查不了。
- [ ] 实现；提交。

### Task 5：考卷 C

**Files:** Create `src/jianjin/eval/exam_c.py`；Modify `src/jianjin/cli.py`；Test `tests/test_exam_c.py`。

- [ ] 测试先写（合成数据）：阈值取「每个成功数据集报出不超过 5%」的最小格点；选法按「检出减其他原因报出」、打平看 AUC、再打平取时长；test 不带 `--dev-result` 报错；二次抓取那一项按「同任务非二次抓取」当参照。
- [ ] 实现；CLI 加 `eval c`、`eval c-regrasp`；提交。

### Task 6：开发集上选，定规则

- [ ] 跑 `jianjin eval c --split dev`，结果写 `results/exam_c_dev.json`。
- [ ] 按选中的写 `CheckConfig` 默认值（注明出处）；`types.Cause` 加两类；`jerky`、`too_fast` 进 `BATCH_CHECKS`；测试先写；提交。

### Task 7：测试集各跑一次

- [ ] `jianjin eval c --split test --dev-result results/exam_c_dev.json`；`jianjin eval c-regrasp`。照实记。

### Task 8：报告、文档、收尾

- [ ] 重出五个公开数据集的报告，和加之前比，新报出的逐条记。
- [ ] 设计文档（第 4.3 节规则、第 5 节对照表、考卷 C）、README、STATUS、重打 wheel。
- [ ] 全部测试；合并进 main、推送。

## 执行记录

（做的时候补）
