# 演示视频脚本（2026-10-05，第二遍检查后重排）

给 Nebius 提交用的 YouTube 视频，英文旁白。海创π要是也要演示视频，用同一段录屏换中文旁白，讲稿在路演 PPT 每页的备注里。

## 要求

- 短于 3 分钟：评委不必看 3 分钟以后的部分（比赛规则原文，2026-10-05 核过）。
- 没有硬件的项目，至少要有 1 分钟展示关键模块在运行（调研时从 Physical AI 赛道页面读到的，提交前再核一次）。这一版里软件在跑的画面从 0:25 到 2:05，共 1 分 40 秒，终端那段不快进。
- 按 Devpost 的要求，YouTube 上设成公开可见。

## 准备

- 录屏 1920×1080；终端字号调大；关掉系统通知。Windows 自带的 Xbox Game Bar（Win+G）或 OBS 都能录。
- 英文画面卡片：浏览器打开 `docs/submission/video-cards.html`，按 F11 全屏，左右方向键换页。共 5 张：标题、为什么、成绩、Nemotron 怎么用、结尾。视频里不再放中文 PPT，评委看得懂。
- 用到的文件都在本机：
  - `reports/clips/fold_towel_ep2_regrasp_zoom2x_halfspeed.mp4`：ep2 重抓那几秒，右上角放大两倍、半速，9 秒。原速原画面的是同目录下 `fold_towel_ep2_regrasp_11s-15.5s.mp4`。
  - `reports/svla_so100_stacking_en.html`：叠方块 56 条的英文报告，作废的 ep39 在里面（`scripts/run_summaries.py` 生成）。
  - `reports/fold_towel_en.html`：录屏时现场生成。
- 密钥：开始录之前，在要录的那个终端里按之前的办法设好 `NEBIUS_API_KEY`（隐藏输入），然后清屏。屏幕上任何时候都不要出现密钥。
- 叠毛巾这批还没调过 Nemotron（第 4 版提示），第一次跑就是现场调用。要是终端显示 template，或者要重录，先把这批的两份缓存挪走再跑（缓存会把退回模板的结果也存下来）：

```powershell
New-Item -ItemType Directory -Force cache\summary_old | Out-Null; Get-ChildItem cache\summary\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 2 | Move-Item -Destination cache\summary_old
```

## 分镜（约 2 分 50 秒）

| 时间 | 画面 | 旁白 |
|---|---|---|
| 0:00 到 0:10 | 放大慢放的重抓片段，先不加任何说明 | This episode comes from a public robot dataset, published for training. Watch the right gripper. |
| 0:10 到 0:25 | 卡片「Why」 | Robot foundation models learn from teleoperation demonstrations like this one. In China, collection sites with a hundred or more robots now produce them, and one of them sells an hour of data for 500 to 1,000 yuan. Quality control still relies largely on people. |
| 0:25 到 0:55 | 终端里现场跑下面那条命令，不快进，等最后两行出来 | jianjin runs on the collection site's own machine, one command per batch. These are the hundred towel-folding episodes from that dataset. The joint rules run on the CPU, the two video models' results come from a cached GPU run, and the batch summary is a live call to NVIDIA Nemotron on Nebius Token Factory. |
| 0:55 到 1:20 | 浏览器打开 `reports/fold_towel_en.html`，停在顶部的本批小结（英文在前、中文在后）和漏斗 | The report is one HTML file that opens offline. At the top, the summary Nemotron just wrote, in English and Chinese. The model never sees the data, only our statistics. It writes placeholders: every number was filled in by jianjin, and the tip for the operators comes from our own list. Eighty-seven episodes pass, thirteen go to a person, none is rejected. |
| 1:20 到 1:45 | 往下滚到 ep2 的体检单，停在关键帧和证据那一行 | Episode 2 goes to a person. Its right gripper switched six times, where this task usually switches at most four. The keyframes are taken at the switches, and two of them, 12.8 and 13.5 seconds, are the moment you just watched: the gripper loses the towel corner and grabs it again. Beijing Humanoid's QA labels this a regrasp. |
| 1:45 到 2:05 | 切到 `reports/svla_so100_stacking_en.html`，滚到作废的 ep39，停在关键帧 | A rejection needs two independent sources. Episode 39 of a cube-stacking dataset: the joints say the arm never moved, and the video model says the task never happened. On video: fourteen seconds of a still arm, then a human hand. |
| 2:05 到 2:35 | 卡片「Results」，念到哪格指哪格 | In five public datasets, jianjin rejected six of 308 episodes, and all six were unusable. On 1,678 episodes that Beijing Humanoid's QA rejected, the regrasp rule, chosen on another benchmark and never retuned, flags 99 of the 162 regrasps it can compare. We also report what failed: common smoothness metrics did not separate the episodes the reviewers called unsmooth. |
| 2:35 到 2:50 | 卡片「Closing」 | jianjin: screen every episode, explain what went wrong, and feed it back to the operators. It is open source on GitHub. |

旁白约 330 词，按每分钟 150 词念约 2 分 15 秒，其余时间留给画面。时间不够时先删 0:10 到 0:25 那张卡片；「Nemotron」那张卡片是备用的，评委要是更在意 NVIDIA 模型怎么用，可以插在 1:20 前面，同时把 2:05 那段缩短。

开头先放画面、再说是什么：评委前 10 秒决定要不要认真看，一段「看得出哪里错了吗」比一句介绍抓人。

0:25 那一段的命令：

```bash
uv run jianjin check data/CoRL2026-CSI__SO101-DualArm-teleop_fold_towel_100epi --judges robometer,topreward --camera observation.images.left_top --out reports/fold_towel_en.html --summary --lang en
```

2026-10-05 不带 `--summary` 实测，这条命令 13 秒跑完；加上 Nemotron 两次调用，预计半分钟以内。

## 录之前核一遍

- 叠毛巾的报告里 ep2 仍是存疑（二次抓取，换档在 2.9、11.1、12.8、13.5、20.9、22.0 秒，关键帧取在 2.9、12.8、13.5、22.0 秒），整批合格 87、存疑 13、作废 0；叠方块 ep39 仍是作废（空录加指令不符）。
- 终端打印是英文；最后两行应是「Batch summary (zh): written by the model」「Batch summary (en): written by the model」。
- 视频里出现的数字和 README、Devpost 一致：308 条里作废 6 条、6 条全是真问题；RoboMIND 二次抓取能比的 162 条里检出 99 条。
- 仓库已经设成公开，结尾卡片上的链接打得开。
