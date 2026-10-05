# 演示视频脚本（2026-10-05）

给 Nebius 提交用的 YouTube 视频，英文旁白。海创π要是也要演示视频，用同一段录屏换中文旁白，讲稿在路演 PPT 每页的备注里。

## 要求

- 短于 3 分钟：评委不必看 3 分钟以后的部分（比赛规则原文，2026-10-05 核过）。
- 没有硬件的项目，至少要有 1 分钟展示关键模块在运行（调研时从 Physical AI 赛道页面读到的，提交前再核一次）。
- 按 Devpost 的要求，YouTube 上设成公开可见。

## 准备

- 录屏 1920×1080；终端字号调大；关掉系统通知。Windows 自带的 Xbox Game Bar（Win+G）或 OBS 都能录。
- 用到的文件都在本机：
  - `reports/svla_so100_stacking.html`：叠方块 56 条，报告顶部有 Nemotron 写的本批小结。
  - `reports/fold_towel.html`：叠毛巾 100 条，ep2 是二次抓取。
  - `reports/clips/fold_towel_ep2_regrasp_zoom2x_halfspeed.mp4`：ep2 重抓那几秒，右上角放大两倍、半速，9 秒。原速原画面的是同目录下 `fold_towel_ep2_regrasp_11s-15.5s.mp4`。
  - 路演 PPT 的封面、方案、成绩几页。
- 叠方块那批的小结已经缓存，录屏时不用设密钥，终端会直接显示「模型写的」。屏幕上任何时候都不要出现密钥。

## 分镜（约 2 分 45 秒）

| 时间 | 画面 | 旁白 |
|---|---|---|
| 0:00 到 0:12 | 放大慢放的重抓片段，先不加任何说明 | This episode was published as a successful demonstration. Watch the right gripper. |
| 0:12 到 0:28 | 片段停在 13.6 秒，切到叠毛巾报告里 ep2 的体检单 | It loses the towel corner at 12.9 seconds and grabs it again at 13.6. jianjin caught it from the joint data alone: the gripper opened and closed six times, where this task usually needs four. Beijing Humanoid's QA team calls this a regrasp. |
| 0:28 到 0:45 | 路演 PPT 缘起页，接行业页 | Robot foundation models learn from teleoperation demonstrations. In China, dedicated sites sell an hour of them for 500 to 1,000 yuan, and quality control is still people watching episodes one by one. |
| 0:45 到 1:05 | 终端里跑下面那条命令，等待的部分快进 | jianjin runs on the site's own machine, one command per batch. Here are 56 cube-stacking episodes from another public dataset, all published as successful. |
| 1:05 到 1:35 | 浏览器打开叠方块的报告，停在顶部的本批小结和漏斗 | The report opens offline. At the top, a summary written by NVIDIA Nemotron on Nebius Token Factory, in Chinese and English. The model never sees the data, only our statistics, and it writes placeholders: every number you read here was filled in by jianjin. Below, the funnel: 46 pass, 5 doubt, 5 reject. |
| 1:35 到 2:00 | 往下滚到作废的 ep39 体检单，停在关键帧 | Episode 39 is rejected because two independent sources agree: the joints say the arm never moved, and the video model says the task never happened. On video: fourteen seconds of a still arm, then a human hand. |
| 2:00 到 2:30 | 路演 PPT 的「成绩一」「成绩三」两页 | Across five public "successful" datasets, jianjin flagged 39 of 308 episodes, and 18 are real problems. On 1,678 episodes rejected by Beijing Humanoid's QA, the regrasp rule, tuned on another dataset and left untouched, finds 61 percent of regrasps. We also report what fails: common smoothness metrics point the wrong way on these real labels. |
| 2:30 到 2:45 | 路演 PPT 封底，加仓库链接 | jianjin: screen first, explain the second it went wrong, and feed it back to the operators. It is open source on GitHub. |

旁白约 300 词，按每分钟 150 词念大约 2 分钟，其余时间留给画面。开头 28 秒和 0:45 到 2:00 都是数据和软件的画面，合起来超过 1 分钟。

开头先放画面、再说是什么：评委前 10 秒决定要不要认真看，一段「看得出哪里错了吗」比一句介绍抓人。

0:45 那一段的命令：

```bash
uv run jianjin check data/lerobot__svla_so100_stacking --judges robometer,topreward --camera observation.images.top --out reports/svla_so100_stacking.html --summary
```

## 录之前核一遍

- 报告里叠方块 ep39 仍是作废（空录加指令不符），叠毛巾 ep2 仍是存疑（二次抓取）。
- 视频里出现的数字和 README、路演 PPT 一致：308 条标出 39 条、18 条确有问题；RoboMIND 二次抓取 61%。
- 仓库已经设成公开，封底的链接打得开。
