import numpy as np

from assay.checks.signal import joint_scale, path_length, runs, speed


def test_runs():
    m = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
    assert runs(m) == [(1, 3), (4, 5), (7, 8)]
    assert runs(np.zeros(0, dtype=bool)) == []


def test_joint_scale_ignores_still_joint():
    s = np.stack([np.linspace(0, 10, 101), np.full(101, 3.0)], axis=1)
    sc = joint_scale(s)
    assert np.isclose(sc[0], 9.0)  # 95 分位减 5 分位
    assert np.isinf(sc[1])


def test_speed_is_unit_free():
    deg = np.linspace(0, 90, 31)[:, None]
    rad = np.deg2rad(deg)
    np.testing.assert_allclose(speed(deg, 30), speed(rad, 30))
    assert speed(deg, 30).shape == (30,)


def test_path_length_doubles_for_back_and_forth():
    there = np.linspace(0, 1, 50)[:, None]
    back = np.concatenate([there, there[::-1], there])
    scale = np.array([1.0])
    assert np.isclose(path_length(back, scale), 3 * path_length(there, scale))


import pytest  # noqa: E402

from assay.checks.signal import excursion  # noqa: E402


def test_excursion_is_farthest_any_joint_got_from_start():
    s = np.stack([np.array([0.0, 0.5, 0.2]), np.array([1.0, 1.0, 3.0])], axis=1)
    scale = np.array([1.0, 2.0])
    assert excursion(s, scale) == pytest.approx(1.0)  # 第二个关节走了 2，按幅度 2 算是 1
    assert excursion(s[:1], scale) == 0.0


def test_level_changes_counts_with_hysteresis():
    from assay.checks.signal import level_changes

    x = np.array([0, 0, 10, 10, 0, 0, 10, 10, 0], dtype=float)  # 开合两回
    assert level_changes(x, 0, 10, 0.15) == [2, 4, 6, 8]
    jitter = np.array([0, 4, 6, 4, 6, 0], dtype=float)  # 在中间那段抖，不算换档
    assert level_changes(jitter, 0, 10, 0.15) == []
    assert level_changes(np.array([0, 10, np.nan, 0]), 0, 10, 0.15) == [1, 3]
    assert level_changes(np.ones(5), 1, 1, 0.15) == []  # 没有幅度


def test_gripper_channels_by_name_then_index():
    from assay.checks.signal import gripper_channels

    assert gripper_channels(["left_j0", "left_gripper.pos", "right_j0", "right_gripper.pos"], 1) == [1, 3]
    assert gripper_channels(["j0", "j1"], 1) == [1]
    assert gripper_channels(["j0"], None) == []


def test_level_changes_ignores_brief_visits():
    """在一档里待不够 min_frames 的那一趟不算，前后同一档的接起来。"""
    from assay.checks.signal import level_changes

    x = np.array([0, 0, 0, 10, 10, 10, 10, 0, 0, 0, 10, 0, 0, 0], dtype=float)  # 第 10 帧一闪到开
    assert level_changes(x, 0, 10, 0.15) == [3, 7, 10, 11]
    assert level_changes(x, 0, 10, 0.15, min_frames=2) == [3, 7]
    edges = np.array([0, 10, 10, 10, 10, 0], dtype=float)  # 开头、结尾那一趟再短也算：不是去了又回来
    assert level_changes(edges, 0, 10, 0.15, min_frames=2) == [1, 5]


def test_arm_channels_skip_gripper_and_hand():
    from assay.checks.signal import arm_channels

    assert arm_channels(["j0", "j1", "gripper"]) == [0, 1]
    assert arm_channels(["left_j0", "left_hand_j0", "right_gripper", "shoulder_pan.pos"]) == [0, 3]


def test_stops_counts_only_mid_episode_runs():
    from assay.checks.signal import stops

    v = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0], dtype=float)
    assert stops(v, 0.5, 1) == [(4, 7), (9, 10)]  # 开头、结尾那两段不算
    assert stops(v, 0.5, 2) == [(4, 7)]
    assert stops(np.zeros(5), 0.5, 1) == []


def test_bursts_counts_rises_and_ignores_blips():
    from assay.checks.signal import bursts

    v = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0], dtype=float)
    assert bursts(v, 0.2, 0.5) == [2, 7, 10]
    assert bursts(v, 0.2, 0.5, min_frames=2) == [2, 10]  # 第 7 帧一闪到高档不算
    dip = np.array([0, 1, 1, 0, 1, 1, 0], dtype=float)
    assert bursts(dip, 0.2, 0.5) == [1, 4]
    assert bursts(dip, 0.2, 0.5, min_frames=2) == [1]  # 中途一闪慢下来又快起来，算一段
    assert bursts(np.array([1, 1, 0, 0, 1, 1.0]), 0.2, 0.5) == [0, 4]  # 一开头就在动，算一段
    assert bursts(np.array([0, 0.3, 0.4, 0.3, 0.0]), 0.2, 0.5) == []  # 没到高档


def test_jerk_is_zero_for_constant_acceleration_and_unit_free():
    from assay.checks.signal import jerk

    t = np.linspace(0, 1, 31)
    quad = (t**2)[:, None]
    assert jerk(quad, 30, np.array([1.0])).shape == (28,)
    np.testing.assert_allclose(jerk(quad, 30, np.array([1.0])), 0, atol=1e-6)
    shaky = quad + 0.01 * (np.arange(31) % 2)[:, None]
    assert jerk(shaky, 30, np.array([1.0])).mean() > 100
    deg = np.rad2deg(shaky)
    np.testing.assert_allclose(jerk(deg, 30, np.array([np.rad2deg(1.0)])), jerk(shaky, 30, np.array([1.0])))
