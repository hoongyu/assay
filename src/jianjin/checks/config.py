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
    end_window_s: float = 0.5  # 看结尾多长一段
    end_moving_rate: float = 0.1  # 结尾平均速度高于它，算没做完就停了
    min_duration_s: float = 1.0  # 短于它，硬规则判过短
    max_gap_factor: float = 1.5  # 相邻时间戳间隔超过 factor/fps，算丢帧
    extra_factor: float = 2.0  # 路程超过同任务中位数这么多倍，算多余动作
    min_task_episodes: int = 5  # 同任务少于这么多条，不做同任务比较
