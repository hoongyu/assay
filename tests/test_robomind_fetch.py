import h5py
import numpy as np
import pytest

from assay.eval.robomind_fetch import CappedReader, TooBig, copy_joints
from assay.ingest.robomind import read_robomind


def make_source(path):
    """像 RoboMIND 失败示范：关节很小，图像很大。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.attrs["sim"] = False
        h["puppet/arm_joint_position"] = np.linspace(0, 1, 40)[:, None] * np.arange(1, 7)
        h["puppet/hand_joint_position"] = (np.arange(40) // 10 % 2)[:, None].astype(float)
        h["language_instruction"] = b"Open drawer."
        h["observations/rgb_images/camera_top"] = np.random.default_rng(0).integers(0, 255, (40, 50_000), dtype=np.uint8)
    return path


def test_copy_joints_keeps_joints_and_drops_images(tmp_path):
    src = make_source(tmp_path / "src.hdf5")
    dst = tmp_path / "out/ur_open/速度过快/e1/data/trajectory.hdf5"
    with open(src, "rb") as raw:
        f = CappedReader(raw, cap=500_000)
        info = copy_joints(f, dst, source="hf://datasets/x/failure_data/e1", source_bytes=src.stat().st_size)
    assert f.n < 500_000 < src.stat().st_size  # 图像那 2 MB 没读
    assert info["frames"] == 40 and info["keys"]["puppet/arm_joint_position"] == [40, 6]
    with h5py.File(dst, "r") as h:
        np.testing.assert_allclose(h["puppet/hand_joint_position"][()][:, 0], (np.arange(40) // 10 % 2))
        assert h["observations/rgb_images/camera_top"].shape == (0,)
        assert tuple(h["observations/rgb_images/camera_top"].attrs["original_shape"]) == (40, 50_000)
        assert h.attrs["assay_joints_only"] and h.attrs["assay_source"] == "hf://datasets/x/failure_data/e1"
        assert h.attrs["sim"] == False  # noqa: E712  原文件的属性照抄
    (ep,) = read_robomind(tmp_path / "out").episodes
    assert ep.state.shape == (40, 7) and ep.meta["image_keys"] == ["camera_top"]


def test_copy_joints_stops_past_cap(tmp_path):
    src = make_source(tmp_path / "src.hdf5")
    dst = tmp_path / "out/e/data/trajectory.hdf5"
    with open(src, "rb") as raw, pytest.raises(TooBig):
        copy_joints(CappedReader(raw, cap=1_000), dst, source="x", source_bytes=1)
    assert not dst.exists() and not dst.with_suffix(".part").exists()
