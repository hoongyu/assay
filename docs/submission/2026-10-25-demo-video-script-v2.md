# 演示视频脚本第二版和录制清单（Nebius × NVIDIA，2026-10-25）

给 Nebius 提交的 YouTube 视频：英文旁白，英文字幕烧进画面，2 分 50 秒以内。第一版（`2026-10-05-demo-video-script.md`）的骨架保留：先放画面、终端现场跑、体检单、成绩卡。这一版按总管 10-19 的方案改了 5 处：

- 前 25 秒用声音讲清 Nebius Token Factory 和两个 Nemotron。
- 终端里现场调 Nemotron 3.5 Super VL 复核，屏幕上能看到模型 ID。
- 体检单上显示复核结论。
- 加一段在开发时没用过的公开数据集上现场跑。
- 全片烧录英文字幕。

一张图、一个数、一句话，片头、中段、片尾各出现一次：
- 图：ep2 在换档时刻 12.8 秒、13.5 秒的两张关键帧。
- 数：308 条里作废 6 条，6 条全是真问题。
- 话：The joints say when. Nemotron says what.

Super VL 的成绩已按 `results/super_vl_review.json`（10-25 用户跑的评测）填好：模型在 ep2 上答对（yes），段 D 照原计划用 ep2。

## 一、录之前准备一次（约 30 分钟）

### 1. 软件

- OBS Studio（免费，obsproject.com 下载安装）：录屏。
- Clipchamp（Windows 11 自带，开始菜单搜 Clipchamp）：剪辑、录旁白、自动生成英文字幕并烧进画面。
- 都不想装的话，Windows 自带的 Xbox Game Bar（Win+G）也能录屏，但只能录单个窗口，切浏览器和终端要分开录。

### 2. OBS 设置（只设一次）

1. 设置 → 视频：基础分辨率和输出分辨率都选 1920×1080，帧率 30。
2. 设置 → 输出：录像格式选 mp4（或 mkv，录完用“文件 → 转封装”转成 mp4），编码器选“硬件（NVENC）”，没有就选 x264；录像质量选“高质量，中等文件大小”。
3. 设置 → 音频：录屏时不录声音，把“麦克风/辅助音频”设为“已禁用”。旁白之后在 Clipchamp 里单独录，念错了好重来。
4. 来源：点“+”加一个“显示器采集”，选主屏幕。
5. 设置 → 热键：开始录制 Ctrl+F9，停止录制 Ctrl+F10。

### 3. 屏幕和终端

- 打开 Windows 的“勿扰”，关掉其他程序的通知。
- 屏幕分辨率 1920×1080，缩放 100%。笔记本是 2K、3K 屏的，录之前先把显示分辨率临时调成 1920×1080。
- 终端用 Windows Terminal：字号 20，配色选深色，窗口最大化。
- 浏览器缩放 125%，按 F11 全屏，书签栏隐藏。

### 4. 密钥（屏幕上任何时候都不能出现）

在要录的那个终端里先跑下面两行，第一行会提示输入密钥，输入时不显示；然后按 Ctrl+L 清屏，再开始录：

```powershell
$s = Read-Host -AsSecureString "NEBIUS_API_KEY"; $env:NEBIUS_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($s))
cd C:\Claude_code\jianjin
```

### 5. 要用到的文件（都在本机）

| 用在哪段 | 文件 |
|---|---|
| A | `reports/clips/fold_towel_ep2_regrasp_zoom2x_halfspeed.mp4`：ep2 重抓那几秒，右上角放大两倍、半速，9 秒 |
| B、G、H | `docs/submission/video-cards.html`：英文卡片，浏览器打开后按 F11 全屏，左右方向键翻页 |
| C、D、E | 段 C 现场生成的 `reports/fold_towel_en.html` |
| F | `data/census/RohanVasa__Switch_On_20260725_160650`：开关操作 50 条，Apache-2.0，开发和调参时没用过 |

## 二、分镜（共约 2 分 45 秒）

每段单独录成一个文件，文件名用段号（A.mp4、B.mp4……），方便剪。旁白按每分钟 140 词念，约 2 分 30 秒，剩下的时间留给画面。

| 段 | 时间 | 时长 | 画面和操作 | 旁白（英文原文，照念） |
|---|---|---|---|---|
| A | 0:00 至 0:09 | 9 秒 | 直接放 A 的片段文件，不加说明 | This episode is from a public robot dataset, published for training. Watch the right gripper. |
| B | 0:09 至 0:25 | 16 秒 | 卡片第 1 页（Assay 是什么），不动 | Assay checks robot teleoperation demonstrations before they are used to train robots. It runs on Nebius Token Factory with two NVIDIA open models: Nemotron 3.5 Super VL watches the robot camera, and Nemotron 3.5 Lightning writes the batch summary. The joints say when. Nemotron says what. |
| C | 0:25 至 1:05 | 40 秒 | 终端：粘贴下面“段 C 的命令”回车，等到最后一行 Report written 出来。不要中途停，终端里能看到模型 ID 和每条复核的回答 | One command checks a batch. These are the hundred towel-folding episodes from that dataset. The joint rules run on the CPU in seconds. For every flagged episode, the rules pick the moment and write one question. Nemotron 3.5 Super VL, on Nebius Token Factory, watches that clip and answers from a fixed list of choices. Here comes each answer, live. |
| D | 1:05 至 1:35 | 30 秒 | 浏览器打开 `reports/fold_towel_en.html`，往下滚到 ep000002 的体检单，停在关键帧和下面的 Video review 那一块 | Episode two. The joints say when: the right gripper switched six times, where this task usually switches four, and the keyframes sit at 12.8 and 13.5 seconds. Nemotron says what: asked whether the gripper loses the towel and grasps it again, it answers yes. Beijing Humanoid's quality team calls this a regrasp. |
| E | 1:35 至 1:50 | 15 秒 | 同一页滚回顶部，停在 Batch summary 和漏斗 | At the top, Nemotron 3.5 Lightning writes the batch summary from our statistics. Every number in it is filled in by Assay, never by the model. |
| F | 1:50 至 2:05 | 15 秒 | 终端：粘贴下面“段 F 的命令”回车，2 秒左右出结果 | Assay also runs on data it has never seen. This switch-flipping dataset was never used to tune it. Fifty episodes, checked in two seconds. |
| G | 2:05 至 2:35 | 30 秒 | 卡片第 2 页（成绩），念到哪格用鼠标指哪格 | In five public datasets, Assay rejected six of 308 episodes, and all six were unusable. Watching the clip, Nemotron cleared eleven of twelve false alarms and confirmed both real problems it could judge; on very short clips, keyframes did better. On robomimic, removing the episodes Assay flags beats removing the same number at random, 36 against 21 percent success, but does not beat keeping them all. We report both. |
| H | 2:35 至 2:48 | 13 秒 | 卡片第 3 页（结尾和数据出处），停 3 秒；再切到浏览器打开 hoongyu.github.io/assay-demo 停 3 秒 | Assay. The joints say when, Nemotron says what. It is open source, and you can try it in your browser. |

段 C 的命令（一行，整行复制）：

```powershell
uv run assay check data/CoRL2026-CSI__SO101-DualArm-teleop_fold_towel_100epi --judges robometer,topreward --camera observation.images.left_top --review --review-no-cache --summary --out reports/fold_towel_en.html --lang en
```

段 F 的命令：

```powershell
uv run assay check data/census/RohanVasa__Switch_On_20260725_160650 --lang en --no-frames
```

- 段 C 要是超过 40 秒（12 条复核现场调用，加上切片段），剪辑时把中间等待的部分加速到 2 倍，并在画面角上加一行字“2× speed”，不能偷偷快进。
- 段 C 第一次录失败要重录时，命令照旧带 `--review-no-cache`，每次都是现场调用。每次 12 条，按探针的用量不到 1 美分。

## 三、旁白和字幕（Clipchamp）

1. 新建项目，比例选 16:9，把 A 到 H 的录屏按顺序拖进时间线。
2. 录旁白：左侧“录制与创建”→“音频”，戴耳机对着麦克风，一段一段照上表念，念错了删掉这段重录。
3. 每段画面的长度按旁白剪齐：画面比旁白长就剪掉尾巴，比旁白短就让最后一帧停留。
4. 字幕：选中旁白音轨，右侧“字幕”→“打开自动字幕”，语言选英语（English），等它转写完。
5. 对着上表的旁白原文逐句改字幕里的错字，重点核对 Nemotron、Nebius、Token Factory、Assay、robomimic 这几个词，以及所有数字。
6. 字幕样式：白字黑底，字号中等，位置靠下居中。
7. 导出：右上角“导出”→ 1080p。Clipchamp 导出时字幕会烧进画面。
8. 导出后从头看一遍：全长不超过 2 分 50 秒，字幕和声音对得上，画面上没有密钥。

不用 Clipchamp 的话：项目窗口可以按你剪好的时间线出一份 SRT 字幕文件，再用下面这行命令把字幕烧进视频（ffmpeg 本机已装）：

```powershell
ffmpeg -i assay_demo.mp4 -vf "subtitles=assay_demo.srt:force_style='FontSize=22,BorderStyle=3'" -c:a copy assay_demo_subs.mp4
```

## 四、上传 YouTube

- 标题：Assay: automatic QA for robot demonstrations (Nebius x NVIDIA, Physical AI)
- 公开范围选“公开”（Public），Devpost 要求公开可见。
- 说明栏第一行写仓库地址 github.com/hoongyu/assay，第二行写在线演示 hoongyu.github.io/assay-demo，第三行写数据出处（和卡片第 3 页一致）。
- 上传完把链接发给项目窗口，填进 Devpost 草稿。

## 五、录之前核一遍（项目窗口在成绩出来后先核，你录之前再看一眼）

- 段 C 跑完，终端最后几行有“Video review: nvidia/Nemotron-3.5-Super-VL on Nebius Token Factory”和每条复核的回答；整批合格 87、存疑 13、作废 0。
- ep000002 的体检单里，Video review 那一块的回答和段 D 旁白里说的一致。要是模型在 ep2 上答错，段 D 换成一条答对的示范，旁白跟着改，成绩卡照实报总体成绩。
- 视频里的数字和 README、Devpost 说明一致：308 条里作废 6 条；robomimic 下游 36% 对 21%；Super VL 复核：12 条误报里视频推翻 11 条（关键帧 9 条），能判断的 2 条真问题都确认，5 条作废里视频确认 3 条、关键帧确认 5 条。
- 画面里不放 NVIDIA、Nebius 的标志图，只说名字。
