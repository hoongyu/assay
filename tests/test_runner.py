import numpy as np

from helpers import make_episode
from jianjin.checks import CheckConfig, run_checks
from jianjin.types import Cause


def test_runner_collects_findings_and_not_checked():
    x = np.concatenate([np.linspace(0, 1, 60), np.ones(150), np.linspace(1, 2, 60), np.full(30, 2.0)])
    paused = make_episode(x, ep_id="paused")
    tiny = make_episode(np.linspace(0, 1, 10), ep_id="tiny")
    results = {r.episode_id: r for r in run_checks([paused, tiny], CheckConfig())}

    assert [f.cause for f in results["paused"].findings] == [Cause.LONG_PAUSE]
    assert "extra_motion" in results["paused"].not_checked

    tiny_causes = {f.check for f in results["tiny"].findings}
    assert "too_short" in tiny_causes
    assert "end_pose" in results["tiny"].not_checked
    assert "regrasp" in results["paused"].not_checked


def test_runner_keeps_input_order():
    eps = [make_episode(np.linspace(0, 1, 60), ep_id=f"e{i}") for i in range(3)]
    assert [r.episode_id for r in run_checks(eps)] == ["e0", "e1", "e2"]
