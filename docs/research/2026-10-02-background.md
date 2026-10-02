# 选题过程和资料（2026-10-02）

总管会话构思拣金时攒下的材料，给项目窗口参考。设计以 `docs/superpowers/specs/2026-10-02-jianjin-design.md` 为准，这里只放来龙去脉和出处。

## 怎么定下来的

- 起因：守窗（协作机械臂旁工人的疲劳和意图识别）被判定不成熟、吸引力有限，指标全靠仿真，意图识别是规则打分。用户要以它为灵感，彻底换一个巧、不花太多钱的具身智能角度，投海创π 具身智能专项赛。
- 用户定的约束：不用机械臂；演示里不要实体机器人；只用手机、摄像头、笔记本这类手边的东西。
- 三个候选：
  1. 数采场的自动质检员：选中。
  2. 手机就是数采场（人戴着手机采示范数据）：没选。「这些数据真能训出机器人」没有机器人证明不了，又会落回「全靠仿真」；大公司也在走这条路。
  3. 听懂「把那个放这儿」（看手指方向和视线，加上语音解析指令）：没选。没有机器人，演示只能停在「理解对没对」；而且和守窗的意图识别是一类东西。
- 选第一个的理由：不用机器人，也能在真机采的真数据上验证；痛点和钱都有出处；对口承办方的「数据与算力」中心；同一个项目还能投 Nebius × NVIDIA 的 Physical AI 赛道。

## 国内数采场

调研 agent 从报道里读到，总管没有逐条复核，写进路演材料前要回原文核。

- 全国至少 15 座大型数采场，单场 3,000 到 5,000 平米；大型数采场投入超过一亿元，基本没有盈利。来源：甲子光年，2026-06-26。
- 北京人形机器人创新中心的基地有 120 多台机器人，年产能最高十几万小时，数据卖 500 到 1,000 元一小时。来源同上。
- 同一个动作要重复 700 多遍，要求用正常速度一半的慢动作；过快或多余的动作会让数据作废；手出汗浸湿手套会影响灵敏度。来源同上。
- 一班约 8 小时，实采 6 小时，过三道质检后剩 3 到 4 小时有效。来源：北京市政府网 2026-09-04 或腾讯 2026-09-03，出自哪篇待核。
- 「采了 100 个小时，漏斗筛选下来只有 50 个小时有效」，质检和标注主要靠人工。来源：甲子光年；单个受访者的说法，不是行业统计。
- 没查到把采集员疲劳、示范质量、策略成功率三者连起来的研究。

## 已有的同类

差异要在路演里讲清：拣金的重点是失败原因诊断、国内数采场三道质检的流程、数据不出场。

| 名字 | 是什么 |
|---|---|
| Loop-Engineering-for-VLA（GitHub 约 740 星） | 开源的 LeRobot 数据集工具包：采集、合并、审计、改进。判定也偏保守：只有结构、数值、解码错误判 drop，语义和阈值类问题判 review，缺前提的检查记 not_checked |
| hflow（约 280 星） | 开源 SDK，帮机器人团队核验训练数据的质量 |
| kalanos | 开源，训练前给机器人数据打分，支持 LeRobot、HDF5、MCAP、ROS 2 |
| Claru | 数据厂商，自称每段采集限 45 分钟，并逐条自动打分 |
| DQAF（arXiv 2605.26349，2026-05） | 按任务进度、平滑度、停顿、关节限位给示范打分，并反馈给采集员；只测了 3 名新手，没测疲劳，代码没查到 |
| DemInf（RSS 2025）、CUPID（CoRL 2025）、Demo-SCORE（2025） | 示范数据筛选的研究；CUPID 在仿真里只用不到 33% 的数据就达到最好水平，Demo-SCORE 报告成功率绝对值提升 15 到 35 个百分点 |

国内的现成质检产品：没查到。

## 能用的零件

- Robometer-4B：基于 Qwen3-VL-4B，在一百多万条机器人轨迹上训练，从视频加任务描述预测每帧的进度和成功概率，不用针对任务训练。Apache-2.0。社区有 4 位量化版，在 8 GB 显存的卡上约占 3.3 GB，那个版本的授权和精度损失待核。论文 arXiv 2603.02115。
- TOPReward：零样本，包一个现成的 Qwen3-VL，读它回答「True」的概率来判成败，没有自己的权重。
- 以上两个都在 LeRobot v0.6.0（2026-07-07）里，带标注脚本，能把每帧进度曲线写回数据集。
- Cosmos 3（NVIDIA，2026-05-31 开源）：能看视频、按时间点定位事件、做物理常识判断；另有 4B 的 Cosmos 3 Edge（2026-07-21）能在端侧跑。Nebius 赛道要求至少用一个 NVIDIA 开源模型，就用它。

## 考卷用的数据

- RoboMIND（北京人形机器人创新中心等，RSS 2025）：107k 条示范，479 个任务，96 类物体；论文说另有 5,000 条真实失败示范，每条标了原因。搜索摘要里说最常见的失败是「定位不准」，在人形机器人的失败里占 48%，这个数待核。
  - Hugging Face 版：Apache-2.0；要同意共享联系方式才能下载；HDF5 格式，总共 12.3 TB；目录按机型分（h5_franka_1rgb、h5_franka_3rgb、h5_ur_1rgb、h5_tienkung_gello_1rgb、h5_agilex_3rgb、h5_simulation），再按任务分，下面是 success_episodes 的 train 和 val。
  - 失败示范有没有公开、在哪：待核。数据卡里没写采集员编号。
  - 另有 RoboMIND 2.0（arXiv 2512.24653），是双臂移动操作的数据集。
- RoboVAD（arXiv 2609.17843，2026-09-15）：机械臂操作视频的帧级异常检测基准，跨域设置（测试时的动作和异常类型训练时没见过）。最难的设置下，所有方法的帧级 micro AUC 都不到 70%。数据和代码在 Zenodo（records/22754659），CC BY-SA 4.0。

## 承办方

- 清华大学具身智能与机器人研究院：2025-11-30 成立，院长是自动化系主任张涛，由自动化、机械、电子、计算机四个系共建，主攻「强健本体加智慧大脑」，目标是打通研发、中试和场景应用。
- 2026-04-30，研究院和 EID 具身智能创新中心一起入驻海淀聚智园，揭牌五个中心：模型与交互、感知与控制、软硬件与本体、数据与算力、战略与标准。
- EID 具体做什么、海创π 的官方评分标准：都没查到。

## 调研成本

调研 agent 1 个，30 次工具调用，约 14.2 万 token；之后总管自己又核了 RoboMIND、RoboVAD、LeRobot v0.6.0、Robometer 和 Cosmos 3。

## 出处

- 甲子光年，2026-06-26：https://news.pedaily.cn/202606/565591.shtml
- 北京市政府网，2026-09-04：https://www.beijing.gov.cn/fuwu/lqfw/gggs/202609/t20260904_4849956.html
- 腾讯新闻，2026-09-03：https://news.qq.com/rain/a/20260903A04W8K00
- 清华新闻，2025-12：https://www.tsinghua.edu.cn/info/1182/123012.htm
- 中关村示范区，2026-04-30：https://www.ncsti.gov.cn/kjdt/scyq/zgckxc/zgcdt/202604/t20260430_245473.html
- RoboMIND 项目页：https://x-humanoid-robomind.github.io/
- RoboMIND 数据卡：https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND
- RoboVAD：https://arxiv.org/abs/2609.17843
- LeRobot v0.6.0：https://huggingface.co/blog/lerobot-release-v060
- Robometer-4B：https://huggingface.co/robometer/Robometer-4B
- Robometer 的 4 位量化版：https://huggingface.co/OpenRAL/rskill-robometer_4b-any-general-nf4
- Cosmos 3：https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/
- Cosmos 3 Edge：https://www.marktechpost.com/2026/07/21/nvidia-releases-cosmos-3-edge-a-4b-parameter-open-world-model-that-reasons-and-generates-robot-actions-on-device/
- DQAF：https://arxiv.org/abs/2605.26349
- CUPID：https://arxiv.org/abs/2506.19121
- DemInf：https://arxiv.org/html/2502.08623
- Demo-SCORE：https://awesomepapers.io/robotics/papers/2503.03707
- Claru：https://claru.ai/training-data/teleoperation
- Loop-Engineering-for-VLA：https://github.com/dsta022/Loop-Engineering-for-VLA
- hflow：https://github.com/Hebbian-Robotics/hflow
- kalanos：https://github.com/KalanosAI/kalanos
