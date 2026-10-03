import numpy as np
import pytest

from helpers import make_episode
from jianjin.checks import CheckConfig
from jianjin.checks.video import WRONG_TASK, relative, wrong_task
from jianjin.judges.base import Curve
from jianjin.types import Cause


def batch(n=6, task="pick"):
    return [make_episode(np.zeros(300), ep_id=f"e{i}", task=task) for i in range(n)]


def curve(peak):
    return Curve("robometer", "v", [0.0, 5.0, 10.0], [0.1, peak / 2, peak])


def test_relative_uses_median_and_iqr_with_floor():
    assert relative([0.5], [0.1, 0.2, 0.3, 0.4, 0.5]).tolist() == pytest.approx([1.0])
    assert relative([0.2], [0.1, 0.1, 0.1]).tolist() == pytest.approx([10.0])  # 四分位距 0，按 0.01 算


def test_wrong_task_compares_with_same_task_batch():
    eps = batch()
    curves = {ep.id: curve(0.9) for ep in eps}
    curves["e0"] = curve(0.3)
    found, skipped = wrong_task(eps, curves, "robometer", CheckConfig())
    assert list(found) == ["e0"] and not skipped
    (f,) = found["e0"]
    assert f.cause == Cause.WRONG_TASK and f.source == "robometer"
    assert (f.t0, f.t1) == (0.0, eps[0].duration) and f.score > WRONG_TASK["robometer"][1]
    assert "30%" in f.evidence and "90%" in f.evidence and "pick" in f.evidence


def test_wrong_task_skips_small_batches_and_missing_curves():
    eps = batch(4) + [make_episode(np.zeros(300), ep_id="p0", task="place")]
    curves = {ep.id: curve(0.9) for ep in eps[:3]}
    found, skipped = wrong_task(eps, curves, "robometer", CheckConfig())
    assert not found
    assert "只有 3 条" in skipped["e0"] and "缓存里没有" in skipped["e3"]


def test_wrong_task_uses_reference_when_given():
    clean = batch()
    odd = make_episode(np.zeros(300), ep_id="odd", task="pick")
    curves = {ep.id: curve(0.9) for ep in clean} | {"odd": curve(0.3)}
    found, _ = wrong_task([odd], curves, "robometer", CheckConfig(), reference=clean)
    assert list(found) == ["odd"]


def test_wrong_task_without_threshold_is_not_checked():
    eps = batch()
    found, skipped = wrong_task(eps, {ep.id: curve(0.9) for ep in eps}, "topreward", CheckConfig())
    assert not found and "还没在开发数据上给 TOPReward 定" in skipped["e0"]


from jianjin.checks.video import drawdown, progress_drop  # noqa: E402


def test_drawdown_finds_biggest_fall_from_running_peak():
    c = Curve("robometer", "v", [0, 1, 2, 3, 4], [0.1, 0.6, 0.3, 0.7, 0.65])
    assert drawdown(c) == pytest.approx((0.3, 1.0, 2.0, 0.6, 0.3))


def test_drawdown_of_rising_curve_is_zero():
    assert drawdown(Curve("robometer", "v", [0, 1, 2], [0.1, 0.2, 0.3]))[0] == 0.0


def test_progress_drop_reports_where_progress_fell():
    eps = [make_episode(np.zeros(90), ep_id=f"e{i}") for i in range(2)]
    curves = {"e0": Curve("robometer", "v", [0, 1, 2], [0.2, 0.8, 0.3]),
              "e1": Curve("robometer", "v", [0, 1, 2], [0.2, 0.5, 0.9])}
    found, skipped = progress_drop(eps, curves, "robometer", CheckConfig(), threshold=0.3)
    (f,) = found["e0"]
    assert f.cause == Cause.DROP and f.source == "robometer" and (f.t0, f.t1) == (1.0, 2.0)
    assert "80%" in f.evidence and "30%" in f.evidence
    assert "e1" not in found and not skipped


def test_progress_drop_without_threshold_or_curve():
    eps = [make_episode(np.zeros(90), ep_id="e0")]
    _, skipped = progress_drop(eps, {}, "topreward", CheckConfig())
    assert "还没在开发数据上给 TOPReward 定" in skipped["e0"]
    _, skipped = progress_drop(eps, {}, "robometer", CheckConfig(), threshold=0.3)
    assert "缓存里没有" in skipped["e0"]
