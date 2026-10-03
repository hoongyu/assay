"""关节体检的全部阈值。默认值在考卷 A 的开发集上定，测试集只用来报成绩。"""

from __future__ import annotations

from dataclasses import dataclass


class NotApplicable(Exception):
    """缺了某种数据，这条规则查不了；记进 not_checked，不当成通过。"""


@dataclass(frozen=True)
class CheckConfig:
    # 速度单位是「每秒走过本条示范里该关节活动幅度的几分之几」
    still_rate: float = 0.02  # 低于它算没动
    pause_s: float = 3.0  # 中途连续不动超过这么久，算停顿过长
    min_duration_s: float = 1.0  # 短于它，硬规则判过短
    max_gap_factor: float = 1.5  # 相邻时间戳间隔超过 factor/fps，算丢帧
    extra_factor: float = 2.0  # 路程超过同任务中位数这么多倍，算多余动作
    end_pose_max: float = 3.0  # 收尾姿态离同任务中位收尾姿态，超过收尾常见出入（四分位距）的这么多倍，算没做完
    end_pose_floor: float = 0.1  # 收尾常见出入至少按关节活动幅度的这么多算，免得一批太齐时一点小差别被放大
    end_spread_max: float = 0.15  # 同任务收尾位置普遍散开超过活动幅度的这么多，收尾本来就不固定，不查
    empty_max: float = 0.2  # 离开头姿态最远不到同任务中位数的这么多，算空录
    min_task_episodes: int = 5  # 同任务少于这么多条，不做同任务比较
    video_slack_frames: float = 1.5  # 视频时长和关节数据差超过这么多帧，算两路对不上
