"""评判注册表。真模型的适配器只在装了 judges 依赖组的机器上能实例化，所以注册的是构造函数。"""

from collections.abc import Callable

from jianjin.judges.base import Curve, FakeJudge, Judge

JUDGES: dict[str, Callable[[], Judge]] = {
    "fake": FakeJudge,
}

__all__ = ["JUDGES", "Curve", "FakeJudge", "Judge"]
