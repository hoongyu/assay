import json

import h5py
import numpy as np

from assay.ingest import load
from assay.ingest.robomimic import read_robomimic


def make_robomimic_h5(path, lengths, masks):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "NutAssemblySquare", "env_kwargs": {"control_freq": 20}})
        for i, n in enumerate(lengths):
            g = data.create_group(f"demo_{i}")
            g.create_dataset("obs/robot0_joint_pos", data=np.tile(np.linspace(0, 1, n)[:, None], (1, 7)))
            fingers = np.linspace(0.04, 0.0, n)
            g.create_dataset("obs/robot0_gripper_qpos", data=np.stack([fingers, -fingers], axis=1))
            g.create_dataset("actions", data=np.zeros((n, 7)))
            g.attrs["num_samples"] = n
        for name, demos in masks.items():
            f.create_dataset(f"mask/{name}", data=np.array(demos, dtype="S"))


def test_reads_joints_gripper_and_operator(tmp_path):
    path = tmp_path / "square" / "mh" / "low_dim_v15.hdf5"
    make_robomimic_h5(path, [50, 60, 70], {
        "better": ["demo_0"], "better_operator_1": ["demo_0"],
        "worse": ["demo_1", "demo_2"], "worse_operator_2": ["demo_1", "demo_2"], "train": ["demo_0", "demo_1"],
    })
    res = read_robomimic(path)
    assert res.failures == []
    assert [ep.id for ep in res.episodes] == ["square/mh/demo_0", "square/mh/demo_1", "square/mh/demo_2"]
    ep = res.episodes[1]
    assert ep.source == "robomimic"
    assert ep.task == "NutAssemblySquare"
    assert ep.fps == 20
    assert ep.state.shape == (60, 8)
    assert ep.state_names[-1] == "gripper" and ep.gripper_index == 7
    np.testing.assert_allclose(ep.state[0, 7], 0.08)  # 两指开口相加
    assert ep.action.shape == (60, 7)
    np.testing.assert_allclose(ep.timestamps[:2], [0.0, 0.05])
    assert ep.operator_id == "worse_operator_2"
    assert ep.meta["robomimic_quality"] == "worse"
    assert res.episodes[0].meta["robomimic_quality"] == "better"


def test_demo_order_is_numeric(tmp_path):
    path = tmp_path / "can" / "mh" / "low_dim_v15.hdf5"
    make_robomimic_h5(path, [30] * 11, {})
    ids = [ep.id.rsplit("/", 1)[1] for ep in read_robomimic(path).episodes]
    assert ids[:3] == ["demo_0", "demo_1", "demo_2"] and ids[-1] == "demo_10"
    assert read_robomimic(path).episodes[0].operator_id is None


def test_load_dispatches_robomimic_file(tmp_path):
    path = tmp_path / "lift" / "ph" / "low_dim_v15.hdf5"
    make_robomimic_h5(path, [40], {})
    (ep,) = load(path).episodes
    assert ep.source == "robomimic"
