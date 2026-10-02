import numpy as np

from helpers import make_robovad
from jianjin.ingest.robovad import read_robovad, task_key


def arm(n=90):
    s = np.tile(np.linspace(0, 1, n)[:, None], (1, 6))
    return s, s + 0.01


def test_task_key_prefers_longest_prefix():
    assert task_key("h_pickandplace_multi_ep3") == "h_pickandplace_multi"
    assert task_key("h_pickandplace_ep3") == "h_pickandplace"


def test_reads_episodes_and_annotations(tmp_path):
    ann = {"h_stackcubes_ep2": [{"start_frame": 10, "end_frame": 20, "type": "collision", "categories": {}}]}
    root = make_robovad(tmp_path / "rv", {"h_stackcubes_ep2": arm(), "h_pour_rice_ep5": arm(60)}, annotations=ann)
    res = read_robovad(root)
    assert res.failures == []
    eps = {e.id: e for e in res.episodes}
    e = eps["h_stackcubes_ep2"]
    assert e.source == "robovad" and e.fps == 30.0
    assert e.state.shape == (90, 6) and e.action.shape == (90, 6)
    assert e.gripper_index == 5
    assert np.isclose(e.timestamps[0], 0.0)
    assert e.meta["task_key"] == "h_stackcubes"
    assert e.meta["anomalies"][0]["type"] == "collision"
    np.testing.assert_array_equal(e.meta["frame_index"], np.arange(90))
    assert eps["h_pour_rice_ep5"].meta["anomalies"] == []
    assert "rice" in eps["h_pour_rice_ep5"].task


def test_only_requested_episodes(tmp_path):
    root = make_robovad(tmp_path / "rv", {"h_stackcubes_ep1": arm(), "h_stackcubes_ep2": arm()})
    assert [e.id for e in read_robovad(root, names=["h_stackcubes_ep2"]).episodes] == ["h_stackcubes_ep2"]
