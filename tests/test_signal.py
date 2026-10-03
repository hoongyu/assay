import numpy as np

from jianjin.checks.signal import joint_scale, path_length, runs, speed


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

from jianjin.checks.signal import excursion  # noqa: E402


def test_excursion_is_farthest_any_joint_got_from_start():
    s = np.stack([np.array([0.0, 0.5, 0.2]), np.array([1.0, 1.0, 3.0])], axis=1)
    scale = np.array([1.0, 2.0])
    assert excursion(s, scale) == pytest.approx(1.0)  # 第二个关节走了 2，按幅度 2 算是 1
    assert excursion(s[:1], scale) == 0.0
