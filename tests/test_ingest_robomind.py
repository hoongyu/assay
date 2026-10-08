import numpy as np

from helpers import make_robomind_h5
from assay.ingest.robomind import read_robomind


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


def arm_cols(n, d):
    """各列互不相同的关节读数。"""
    return np.linspace(0, 1, n)[:, None] * np.arange(1, d + 1)


def square(n, period=10):
    return (np.arange(n) // period % 2).astype(float)


def test_failure_layout_single_arm_appends_gripper(tmp_path):
    from helpers import make_robomind_failure_h5

    root = tmp_path / "failure_data"
    make_robomind_failure_h5(root / "ur_open/速度过快/e1/data/trajectory.hdf5", arm=arm_cols(40, 6), hand=square(40)[:, None])
    (ep,) = read_robomind(root).episodes
    assert ep.state.shape == (40, 7) and ep.action.shape == (40, 7)
    assert ep.state_names == ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"]
    assert ep.gripper_index == 6
    assert ep.meta["robomind_failure_cause"] == "速度过快"
    np.testing.assert_allclose(ep.state[:, 6], square(40))


def test_failure_layout_dual_arm_hand_already_inside(tmp_path):
    """双臂 16 列里第 7、15 列就是两只夹爪，和 hand 那两列一样：不重复拼。"""
    from helpers import make_robomind_failure_h5

    arm = arm_cols(50, 16)
    arm[:, 7], arm[:, 15] = square(50), square(50, 7) * 0.9
    make_robomind_failure_h5(tmp_path / "dual_bus/动作不流畅/e/data/trajectory.hdf5", arm=arm, hand=arm[:, [7, 15]].copy())
    (ep,) = read_robomind(tmp_path).episodes
    assert ep.state.shape == (50, 16)
    assert ep.state_names[0] == "left_j0" and ep.state_names[8] == "right_j0"
    assert ep.state_names[7] == "left_gripper" and ep.state_names[15] == "right_gripper"
    assert ep.gripper_index == 7


def test_failure_layout_hand_inside_despite_small_lag(tmp_path):
    """从臂那边手的列和臂里那一列不是逐点相等（实测 95 分位差在幅度的几个百分点），也算已经在臂里；主臂、从臂都是 8 列。"""
    from helpers import make_robomind_failure_h5

    arm = arm_cols(80, 8)
    arm[:, 7] = square(80) * 0.7
    hand = np.roll(arm[:, 7], 1)[:, None]  # 慢一帧：换档那几帧差满幅度，其余一样
    make_robomind_failure_h5(tmp_path / "franka_x/二次抓取/e/data/trajectory.hdf5", arm=arm, hand=hand)
    (ep,) = read_robomind(tmp_path).episodes
    assert ep.state.shape == (80, 8) and ep.action.shape == (80, 8)
    assert ep.state_names[-1] == "gripper" and ep.gripper_index == 7


def test_failure_layout_picks_closest_column(tmp_path):
    """前面有一列碰巧也挨得近（实测 dual_04 有一条第 6 列差的中位数在幅度的 2% 以内），要挑差最小的那一列。"""
    from helpers import make_robomind_failure_h5

    arm = arm_cols(60, 8)
    arm[:, 7] = square(60) * 0.9
    arm[:, 6] = arm[:, 7] + 0.01
    make_robomind_failure_h5(tmp_path / "franka_y/二次抓取/e/data/trajectory.hdf5", arm=arm, hand=arm[:, [7]].copy())
    (ep,) = read_robomind(tmp_path).episodes
    assert ep.gripper_index == 7 and ep.state_names[6] == "j6"


def test_franka_sim_layout(tmp_path):
    """仿真 Franka：franka/joint_position（7 个关节加夹爪），帧率按 sim_time_seq 估。"""
    import h5py

    f = tmp_path / "470-put_apple/动作不流畅/e/data/trajectory.hdf5"
    f.parent.mkdir(parents=True)
    with h5py.File(f, "w") as h:
        h.attrs["sim"] = True
        h["franka/joint_position"] = arm_cols(50, 8)
        h["sim_time_seq"] = np.arange(45) / 120.0
    (ep,) = read_robomind(tmp_path).episodes
    assert ep.state.shape == (50, 8) and ep.gripper_index == 7 and ep.state_names[-1] == "gripper"
    assert abs(ep.fps - 120.0) < 1e-6 and ep.meta["layout"] == "franka_sim" and ep.meta["sim"] is True


def test_tiangong_sim_layout(tmp_path):
    """仿真天工：左右臂各 7、灵巧手各 12，帧数差一帧截到最短；没有主臂；帧率按 sim_time_seq 的间隔中位估。"""
    from helpers import make_tiangong_h5

    make_tiangong_h5(tmp_path / "470-put_apple/动作不流畅/e/data/trajectory.hdf5",
                     arms=(arm_cols(61, 7), arm_cols(60, 7)), hands=(arm_cols(61, 12), arm_cols(60, 12)),
                     sim_time=np.arange(55) / 120.0)
    (ep,) = read_robomind(tmp_path).episodes
    assert ep.state.shape == (60, 38)
    assert ep.state_names[:2] == ["left_j0", "left_j1"] and ep.state_names[7] == "right_j0"
    assert ep.state_names[14] == "left_hand_j0" and ep.state_names[-1] == "right_hand_j11"
    assert ep.gripper_index is None and ep.action is None
    assert abs(ep.fps - 120.0) < 1e-6 and ep.meta["fps_assumed"] is False
    assert ep.meta["robomind_failure_cause"] == "动作不流畅"


def test_tiangong_missing_arm_is_a_failure(tmp_path):
    from helpers import make_tiangong_h5

    f = make_tiangong_h5(tmp_path / "470-x/动作不流畅/e/data/trajectory.hdf5", arms=(arm_cols(30, 7),), hands=None)
    res = read_robomind(f)
    assert res.episodes == [] and len(res.failures) == 1
