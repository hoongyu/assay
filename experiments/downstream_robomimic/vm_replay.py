"""Sanity check: replay recorded demo actions open-loop in the simulator (as robomimic's playback_dataset.py does).
If the recorded actions of human demos do not finish the task, the env setup does not match the data and no
policy can succeed. Writes /content/logs/replay.out."""

import time

import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

WORK = "/content/data/square_mh_assay.hdf5"
ObsUtils.initialize_obs_modality_mapping_from_dict(
    {"low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]})
env_meta = FileUtils.get_env_metadata_from_dataset(WORK)
print("env_meta:", {k: env_meta[k] for k in ("env_name", "type")}, "controller:",
      str(env_meta["env_kwargs"].get("controller_configs"))[:300], flush=True)
env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)
f = h5py.File(WORK, "r")
better = [d.decode() for d in f["mask/better"][()]][:10]
worse = [d.decode() for d in f["mask/worse"][()]][:5]
ok = 0
for d in better + worse:
    t0 = time.time()
    g = f[f"data/{d}"]
    states, actions = g["states"][()], g["actions"][()]
    env.reset()
    env.reset_to({"states": states[0], "model": g.attrs["model_file"]})
    success = False
    for a in actions:
        env.step(a)
        success = success or env.is_success()["task"]
    # also compare the final simulator state with the recorded one
    drift = float(np.abs(env.get_state()["states"] - states[-1]).max())
    ok += success
    print(d, "steps", len(actions), "success", success, "max state drift", round(drift, 4), f"{time.time() - t0:.1f}s", flush=True)
print(f"replayed {len(better) + len(worse)} demos, {ok} succeeded")
