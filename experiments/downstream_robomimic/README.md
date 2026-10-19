# Downstream experiment: does training on Assay-filtered demonstrations give a better policy?

Result: `results/downstream_robomimic.json`. Same design as `notebooks/colab_downstream_robomimic.ipynb`, run on 2026-10-19 with the local `colab` CLI instead of the notebook, four trainings at a time on one high-RAM T4.

## Design

robomimic v1.5 Square MH, low-dim: 300 human teleoperation demonstrations, 6 operators with 50 each, labelled better, okay or worse (100 each). BC-RNN from robomimic v0.5.0 (`robosuite_hyperparameters`), 600 epochs of 100 steps, evaluated every 100 epochs with 50 rollouts of at most 400 steps. Three seeds per condition.

| Condition | Training demonstrations |
|---|---|
| all | all 300 |
| assay | without the 45 that `assay check --no-frames` did not pass (`flagged_square.json`), 255 left |
| random | without 45 random ones, drawn per seed, 255 left |
| oracle | without the 100 labelled worse, 200 left (reference) |

Hypotheses written down before the run: assay is not worse than all; assay is better than random.

## Result at epoch 600 (3 seeds pooled, 150 rollouts per condition, Wilson 95%)

| Condition | Success | Per seed |
|---|---|---|
| all | 59/150 = 39% [32, 47] | 44%, 34%, 40% |
| assay | 54/150 = 36% [29, 44] | 52%, 24%, 32% |
| random | 32/150 = 21% [15, 29] | 48%, 4%, 12% |
| oracle | 51/150 = 34% [27, 42] | 22%, 50%, 30% |

- Removing the 45 episodes Assay flags is better than removing 45 random ones (36% vs 21%; higher in each of the 3 seeds). Counted over 450 rollouts the gap is larger than evaluation noise; counted over 3 seeds it is not significant, because the random seeds vary from 4% to 48%.
- It is not better than keeping all 300 (36% vs 39%, within noise). Removing the 100 demonstrations robomimic labels worse is not better either (34%). At this training length, fewer demonstrations cost about as much as dropping the bad ones gains.
- Success was still rising at epoch 600; robomimic trains 2,000 epochs. The result holds for 600 epochs only.

## Files

| File | Runs on | Does |
|---|---|---|
| `vm_setup.py` | VM, `colab exec -f` | installs robosuite 1.5.2, MuJoCo 3.3.7, robomimic v0.5.0 (`--no-deps`, its pinned transformers does not build on Python 3.13); downloads the data, checks its md5, writes the `mask/exp_*` subsets |
| `vm_runner.py` | VM | `debug`, `probe` (timing) and `main WORKER` (claims runs one at a time) |
| `vm_launch.py`, `vm_guard.py` | VM | starts the guard detached; the guard starts the workers, writes a status line and a results tarball every 5 minutes, enforces a deadline, releases the VM at the end |
| `vm_replay.py` | VM | sanity check: replays recorded actions open-loop (12 of 15 demonstrations succeed) |
| `summarize_ds.py` | anywhere | the table above from the per-run JSON files |

Colab reclaims a VM about 35 minutes after the last client connection to its kernel drops (for example when the laptop that started it goes to standby), even while training runs. Keep the machine awake and send a short `colab exec` every few minutes. Two earlier attempts were lost this way and are not part of the result. Compute: about 6 units for the final run at 1.27 units per hour, about 10 units in total.
