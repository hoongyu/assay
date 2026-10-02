import numpy as np

from helpers import make_robomind_h5
from jianjin.ingest.robomind import read_robomind


def ramp(n, d):
    return np.tile(np.linspace(0, 1, n)[:, None], (1, d))


def test_success_episode(tmp_path):
    root = tmp_path / "RoboMIND"
    make_robomind_h5(
        root / "h5_franka_1rgb/apple_storage/success_episodes/train/ep_0001/data/trajectory.hdf5",
        state=ramp(90, 8),
    )
    res = read_robomind(root)
    assert res.failures == []
    (ep,) = res.episodes
    assert ep.id == "h5_franka_1rgb/apple_storage/success_episodes/train/ep_0001"
    assert ep.task == "apple storage"
    assert ep.source == "robomind"
    assert ep.state.shape == (90, 8) and ep.action.shape == (90, 8)
    assert ep.gripper_index == 7
    assert ep.meta["robot"] == "h5_franka_1rgb"
    assert ep.meta["fps_assumed"] is True
    assert ep.meta["image_keys"] == ["camera_top"]
    assert "robomind_failure_cause" not in ep.meta


def test_failure_episode_keeps_cause(tmp_path):
    root = tmp_path / "RoboMIND"
    make_robomind_h5(root / "failure_data/apple_storage/二次抓取/ep_7/data/trajectory.hdf5", state=ramp(60, 8))
    (ep,) = read_robomind(root).episodes
    assert ep.task == "apple storage"
    assert ep.meta["robomind_failure_cause"] == "二次抓取"


def test_dual_arm_concatenated(tmp_path):
    root = tmp_path / "r"
    lr = np.stack([ramp(60, 7), ramp(60, 7) * 2])
    make_robomind_h5(root / "h5_agilex_3rgb/fold/success_episodes/val/e1/data/trajectory.hdf5", left_right=lr)
    (ep,) = read_robomind(root).episodes
    assert ep.state.shape == (60, 14)
    assert ep.action is None
    assert ep.state_names[6] == "left_gripper" and ep.state_names[13] == "right_gripper"
    assert ep.gripper_index == 6


def test_missing_joints_is_a_failure(tmp_path):
    root = tmp_path / "r"
    make_robomind_h5(root / "h5_ur_1rgb/t/success_episodes/train/e/data/trajectory.hdf5")
    res = read_robomind(root)
    assert res.episodes == []
    assert len(res.failures) == 1 and "joint_position" in res.failures[0][1]


def test_single_file(tmp_path):
    f = make_robomind_h5(tmp_path / "h5_ur_1rgb/t/success_episodes/train/e/data/trajectory.hdf5", state=ramp(30, 8))
    assert len(read_robomind(f).episodes) == 1
