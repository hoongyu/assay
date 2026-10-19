"""Set up the Colab VM for the downstream experiment. ``colab exec -s jianjin --timeout 1200 -f vm_setup.py``.

Installs robosuite 1.5.2 and robomimic v0.5.0, downloads Square MH (low-dim, v1.5) from Hugging Face,
checks it is the same file Assay was run on locally, and writes the four training subsets as
``mask/exp_*`` from /content/ds/flagged_square.json (the 45 demos Assay did not pass).
"""

import json
import subprocess
import time


def sh(cmd, tail=1200):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {cmd[:120]}  [{time.time() - t0:.0f}s, exit {r.returncode}]\n{(r.stdout + r.stderr)[-tail:]}", flush=True)
    return r.returncode


open("/content/ds/vm_start", "a").close()
sh("date -u; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | head -2; python3 -V")
# Colab runs Python 3.13 (2026-10-19): robosuite 1.5.2 pulls numpy 1.26.4 (built from source, ~6 min);
# mujoco 3.16 breaks robosuite 1.5.2's joint lookup, 3.3.7 works; robomimic's pinned transformers 4.41.2
# does not build, and BC-RNN never uses transformers or diffusers, so robomimic goes in with --no-deps.
sh('MAKEFLAGS=-j8 pip -q install "robosuite==1.5.2" "mujoco==3.3.7" 2>&1 | grep -v "^\\s*$" | tail -3')
sh("rm -rf /content/robomimic && git clone -q --depth 1 --branch v0.5.0 https://github.com/ARISE-Initiative/robomimic.git /content/robomimic")
sh("pip -q install --no-deps -e /content/robomimic 2>&1 | tail -3; pip -q install tensorboardX imageio-ffmpeg 2>&1 | tail -3")
sh("cd /content/ds && python3 -c 'import robosuite, robomimic, robomimic.scripts.train, torch, mujoco, numpy; "
   "print(robosuite.__version__, robomimic.__version__, torch.__version__, torch.cuda.is_available(), mujoco.__version__, numpy.__version__)' "
   "2>&1 | grep -v -E 'UNEXPECTED|WARNING|^Key|^---|^Notes|can be ignored|LOAD REPORT|Loading weights' | tail -3")

PREP = r'''
import hashlib, json, shutil
import h5py, numpy as np
from huggingface_hub import hf_hub_download
src = hf_hub_download("robomimic/robomimic_datasets", "v1.5/square/mh/low_dim_v15.hdf5", repo_type="dataset", local_dir="/content/data")
want = json.load(open("/content/ds/flagged_square.json"))
md5 = hashlib.md5(open(src, "rb").read()).hexdigest()
print("md5", md5, "same as local" if md5 == want["md5"] else "DIFFERENT FROM LOCAL")
assert md5 == want["md5"]
flagged = set(want["flagged"])
work = "/content/data/square_mh_assay.hdf5"
shutil.copy(src, work)
with h5py.File(work, "a") as f:
    demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    def put(name, keep):
        if f"mask/{name}" in f:
            del f[f"mask/{name}"]
        f.create_dataset(f"mask/{name}", data=np.array(sorted(keep, key=lambda k: int(k.split("_")[1])), dtype="S"))
    put("exp_all", demos)
    put("exp_assay", [d for d in demos if d not in flagged])
    for s in (1, 2, 3):
        rng = np.random.default_rng(s)
        drop = set(rng.choice(demos, size=len(flagged), replace=False))
        put(f"exp_random_s{s}", [d for d in demos if d not in drop])
    put("exp_oracle", [d.decode() for d in f["mask/okay_better"][()]])
    sizes = {k: len(f[f"mask/{k}"]) for k in sorted(f["mask"].keys()) if k.startswith("exp_")}
print(sizes)
json.dump(sizes, open("/content/ds/mask_sizes.json", "w"))
'''
open("/content/ds/prep.py", "w").write(PREP)
sh("python3 /content/ds/prep.py")
