"""Train BC-RNN on subsets of robomimic Square MH and record rollout success, as in
notebooks/colab_downstream_robomimic.ipynb (cells 6 to 8), without Google Drive.

    python3 vm_runner.py debug          2 epochs, 2 short rollouts: the whole path works
    python3 vm_runner.py probe          10 epochs, 5 rollouts: time per epoch and per rollout
    python3 vm_runner.py main WORKER    claim runs one at a time until none is left

Results go to /content/results/<run>.json. Workers claim a run by creating
/content/results/<run>.claim exclusively, so several workers can share one VM.
"""

import glob
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import traceback

import h5py
import robomimic.utils.torch_utils as TorchUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train

TASK = "square"
WORK = "/content/data/square_mh_assay.hdf5"
RESULTS = "/content/results"
EPOCHS, RATE, N = 600, 100, 50
SEEDS = [1, 2, 3]
CONDITIONS = ["exp_all", "exp_assay", "exp_random", "exp_oracle"]
os.makedirs(RESULTS, exist_ok=True)

spec = importlib.util.spec_from_file_location("rm_example", "/content/robomimic/examples/train_bc_rnn.py")
rm_example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm_example)


def make_config(name, filter_key, seed, epochs, rate, n, horizon=400, steps=100):
    config = config_factory(algo_name="bc")
    config.experiment.name = name
    config.train.data = WORK
    config.train.output_dir = "/content/runs"
    config = rm_example.robosuite_hyperparameters(config)   # robomimic's own BC-RNN low-dim settings
    config.experiment.validate = False
    config.experiment.logging.terminal_output_to_txt = True
    config.experiment.render_video = False
    config.experiment.save.enabled = False
    config.experiment.epoch_every_n_steps = steps
    config.experiment.rollout.rate = rate
    config.experiment.rollout.n = n
    config.experiment.rollout.horizon = horizon
    config.train.hdf5_filter_key = filter_key
    config.train.hdf5_validation_filter_key = None
    config.train.num_epochs = epochs
    config.train.seed = seed
    return config


def run(name, filter_key, seed, epochs, rate, n, horizon=400, steps=100):
    shutil.rmtree(f"/content/runs/{name}", ignore_errors=True)   # robomimic stops to ask when the folder exists
    config = make_config(name, filter_key, seed, epochs, rate, n, horizon, steps)
    t0 = time.time()
    train(config, device=TorchUtils.get_torch_device(try_to_use_cuda=True))
    wall = time.time() - t0
    log = open(glob.glob(f"/content/runs/{name}/*/logs/log.txt")[0], encoding="utf-8").read()
    curve = {str(int(e)): float(s) for e, s in re.findall(r'Epoch (\d+) Rollouts took.*?"Success_Rate": ([0-9.]+)', log, re.S)}
    took = [float(x) for x in re.findall(r"Rollouts took ([0-9.]+)s", log)]
    return {"name": name, "filter_key": filter_key, "seed": seed, "epochs": epochs, "rollouts_per_eval": n,
            "success_by_epoch": curve, "rollout_avg_seconds": took, "wall_seconds": round(wall, 1)}


def plan():
    # seed-major: if the budget runs out, the runs that finished cover whole seeds
    return [(f"{TASK}_{cond}_s{seed}", f"{cond}_s{seed}" if cond == "exp_random" else cond, seed)
            for seed in SEEDS for cond in CONDITIONS]


def save(res):
    path = f"{RESULTS}/{res['name']}.json"
    json.dump(res, open(path + ".part", "w"), ensure_ascii=False, indent=1)
    os.replace(path + ".part", path)


mode = sys.argv[1]
if mode == "debug":
    save(run("debug", "exp_all", 1, epochs=2, rate=1, n=2, horizon=10, steps=3))
elif mode == "probe":
    tag = sys.argv[2] if len(sys.argv) > 2 else ""   # several tagged probes at once measure parallel throughput
    res = run("probe" + tag, "exp_all", 1, epochs=10, rate=10, n=5)
    per_rollout = res["rollout_avg_seconds"][-1]
    res["per_rollout"] = per_rollout
    res["per_epoch"] = (res["wall_seconds"] - per_rollout * 5) / 10
    res["one_run_seconds"] = res["per_epoch"] * EPOCHS + per_rollout * N * (EPOCHS // RATE)
    res["runs"] = len(plan())
    save(res)
elif mode == "main":
    worker = sys.argv[2]
    for name, key, seed in plan():
        if os.path.exists(f"{RESULTS}/{name}.json"):
            continue
        try:
            os.close(os.open(f"{RESULTS}/{name}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        print(time.strftime("%H:%M:%S"), worker, "starts", name, flush=True)
        try:
            res = run(name, key, seed, EPOCHS, RATE, N)
            res["n_demos"] = len(h5py.File(WORK, "r")[f"mask/{key}"])
            res["worker"] = worker
            save(res)
            print(time.strftime("%H:%M:%S"), worker, "done", name, res["success_by_epoch"], flush=True)
        except Exception:
            open(f"{RESULTS}/{name}.error", "w").write(traceback.format_exc())
            print(time.strftime("%H:%M:%S"), worker, "FAILED", name, flush=True)
    open(f"{RESULTS}/worker_{worker}.done", "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
