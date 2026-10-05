# jianjin (拣金)

[中文](README.zh-CN.md)

Automatic quality control for robot teleoperation demonstrations. For every recorded episode, jianjin returns PASS, DOUBT or REJECT and names the problem and the second it happens, so QA reviewers only look at the episodes the machine is unsure about, and every rejection comes with evidence. It runs on site; the report is a single HTML file that opens offline, so the data never has to leave the collection site.

- **Public "successful" datasets are not clean.** In 308 episodes from five public LeRobot datasets, all published as successful, jianjin flagged 39; 18 are real problems, from one-second recordings to a gripper that loses the towel corner and grabs it again.
- **It holds up on real QA labels.** On 1,678 episodes rejected by Beijing Humanoid's QA team (RoboMIND), the regrasp rule finds 61% of regrasps, with parameters tuned on another dataset and never touched.
- **NVIDIA Nemotron on Nebius Token Factory** writes the batch summary, and jianjin fills in every number, so the model cannot invent one.

![The top of a jianjin report: batch summary, verdict funnel, problem table](docs/images/report-stacking.png)

*The top of a report for 56 cube-stacking episodes. The batch summary was written by Nemotron; the numbers in it come from jianjin.*

![An episode card: regrasp detected, with progress curve and keyframes](docs/images/card-towel-ep2-regrasp.png)

*One episode card. The right gripper opened and closed 6 times where this task usually needs at most 4. On video, it loses the towel corner at 12.9 s and grabs it again at 13.6 s.*

Entered in the Nebius × NVIDIA Global AI Hackathon (Physical AI track) and the first Haichuang π embodied-AI competition (首届「海创π」具身智能专项赛).

## Try it in two minutes

```bash
git clone https://github.com/hoongyu/jianjin && cd jianjin
uv sync
uv run --extra hub jianjin fetch exam_a   # the public LeRobot datasets of exam A, joint data only (about 73 MB)
uv run jianjin check data/lerobot__svla_so100_stacking --out report.html
```

Open `report.html` in a browser. Without a GPU only the joint checks run, and they already flag 5 of the 56 episodes: three recordings shorter than one second (rejected) and two where the arm never moves (doubt). We checked all five on video; all five are real problems. The interface is in Chinese, because our first users are collection sites in China.

## Why

Embodied-AI training data is mostly collected by teleoperation: an operator drives a leader arm, the robot follows, and video and joint data are recorded. China has built dedicated data-collection sites for this, at least 15 large ones according to a count by China Business Journal. Data sells for 500 to 1,000 CNY per hour. QA and labeling are still largely manual; the head of one large site estimated that of 100 hours collected, maybe 50 survive the QA funnel (his estimate, not an industry statistic). Whether rejection reasons reach the operators is not reported; our guess is that they rarely do, so the same mistakes repeat on the next shift. Sources and how we checked them: [background notes (Chinese)](docs/research/2026-10-02-background.md).

jianjin does two things:

1. The machine screens every episode first; people only review what it is unsure about.
2. It says why: which second, which kind of problem, aggregated per operator for shift feedback.

## How it judges

| Looks at | Checks | Reports | Needs |
|---|---|---|---|
| Metadata and joint readings | Missing readings; dropped or out-of-order timestamps; too short; video shorter than joint data | Data error (objective hard rule: one is enough to reject) | CPU |
| Metadata | Video longer than joint data | Data error (doubt) | CPU |
| Joints | Stands still too long mid-episode | Long pause | CPU |
| Joints, compared with the other episodes of the same task | Path far longer than usual; end pose unlike a finished task; arm barely moves; gripper opens and closes more than usual; takes far longer than usual | Extra motion, incomplete, empty recording, regrasp, unsmooth motion | CPU |
| Video judge: Robometer-4B, zero-shot | Progress over the episode far below the batch; progress suddenly drops | Wrong task, object dropped | GPU, cached |
| Video judge: TOPReward (wraps Qwen3-VL-4B), zero-shot | Asks the model whether the robot completed the task; the probability of "yes" is far below the batch | Wrong task | GPU, cached |
| NVIDIA Nemotron on Nebius Token Factory (optional) | Writes the batch summary for the QA lead and the operators, in Chinese and English | Nothing: it never judges | API call |

Verdicts are conservative, because throwing away a good episode costs more than missing a bad one:

- **REJECT**: an objective hard rule fires, or two independent sources (the joint rules count as one, each video model as one) point at the same problem, for example both video models report the same wrong task.
- **DOUBT**: something was found, but not enough to reject; a person should look.
- **PASS**: nothing was found. Checks that could not run (no video, too few episodes of the task, a batch whose end poses vary anyway) are listed separately and never count as passed.

The report shows the verdict funnel for the batch, the problem distribution, per-operator statistics when the data carries operator ids, and one card per episode with the reasons, the progress curve, the flagged time span and keyframes.

## Results

Thresholds are chosen on development data; test sets are only used to report. Where we looked at a test set more than once, we say so. Brackets are 95% Wilson intervals. Full record: [design doc (Chinese)](docs/superpowers/specs/2026-10-02-jianjin-design.md), section 6.

### Bad episodes hidden in public "successful" datasets

Five public LeRobot datasets (SO-100, SO-101, single-arm and bimanual), 308 episodes, all published as successful demonstrations. jianjin flagged 39: 33 DOUBT (11%) and 6 REJECT.

- **Real problems, 18**: 3 shorter than one second; 3 where the arm barely moves (one checked on video: 14 seconds of a motionless arm, then a human hand in the last frame; two checked on joint data only); 1 recording of a person resetting the cubes; 1 that stops while the arm is still returning; 1 towel fold where the right gripper loses the corner and grasps again (RoboMIND calls this a regrasp); 9 whose video runs longer than the joint data (the two we watched show the scene being reset).
- **False positives, 12, confirmed on video**: cubes stacked so neatly that the lower one is hidden, an arm covering the cubes for a long time, a towel folded fine but given low progress, and five pick-and-place episodes that succeeded but scored slightly below the batch.
- **Not checked yet, 9**: 8 without local video, 1 where the gripper blocks the camera.

### Exam A: defects injected into real successful episodes

These failures are synthetic. The test set uses other tasks and other robots (SO-101, bimanual).

| Defect | Detected on test, at the right time | Clean episodes flagged |
|---|---|---|
| Cut the ending → incomplete | 197/201 = 98% (95% to 99%) | 0/202 |
| Insert frozen frames → long pause | 199/201 = 99% (96% to 100%) | 0/202 |
| Swap the task description → wrong task (Robometer) | 184/201 = 92% (87% to 95%) | 10/201 = 5% (3% to 9%) |
| Swap the task description → wrong task (TOPReward) | 128/201 = 64% (57% to 70%) | 0/201 = 0% (0% to 2%) |
| Swap the task description, both video models agree → reject | 115/201 = 57% (50% to 64%) | 0/201 = 0% (0% to 2%) |

The incomplete check changed its scaling once to handle bimanual data, so that row is our second look at the test set; its thresholds were still chosen on dev. For TOPReward we tried two prompt formats; format, score and threshold were chosen on dev and the test set was looked at once.

### Exam B: real anomalies in RoboVAD

[RoboVAD](https://arxiv.org/abs/2609.17843) is a frame-level anomaly detection benchmark of robot manipulation videos. jianjin is zero-shot: it was never trained on RoboVAD.

| Setting | Frame-level micro/macro AUC | Rank among the paper's 9 methods | FIDeL, the paper's only joint-data method |
|---|---|---|---|
| in_domain | 66.9/65.6 | 4 | 56.3/54.0 |
| cross_anomaly | 68.4/67.0 | 3 | 52.1/53.4 |
| cross_task | 60.2/60.2 | 5 | 58.1/53.7 |
| cross_task+anomaly | 56.4/53.9 | 7 | 63.2/51.2 |

- All methods in the paper were trained on RoboVAD; the best one beats jianjin by 11 to 28 points of micro AUC.
- Per episode (629 test episodes, joint rules plus two video models): 247/387 = 64% (59% to 68%) of anomalous episodes are flagged, 7 of them rejected; 33/242 = 14% (10% to 19%) of normal episodes are flagged, none rejected.
- The regrasp rule alone (joint data only, compared with normal training episodes): 243/325 = 75% (70% to 79%) of failed grasps or drops, 0/242 (0% to 2%) of normal episodes. This is our second look at the test set: after the first, we stopped counting gripper visits shorter than 0.2 s, to fix a false positive on the Lego dataset.

### Exam C: real QA labels from Beijing Humanoid (RoboMIND)

RoboMIND published 1,678 teleoperation episodes that its QA rejected, with the reason in 11 classes, but no successful episodes from the same batches. jianjin reads only the joint data; it compares each episode with episodes of the same task rejected for other reasons, so "other reasons flagged" is not a false-positive rate: those episodes failed too. Tasks were split into dev and test before we looked at any metric.

| Reason | jianjin's rule | Detected | Same-task episodes with other reasons flagged |
|---|---|---|---|
| Regrasp (284) | Gripper opens and closes more than usual; parameters chosen on RoboVAD, not tuned on RoboMIND | 99/162 = 61% (53% to 68%) | 72/1,085 = 7% (5% to 8%) |
| Unsmooth motion (557) | Takes over 1.7 times the task's median duration (test set, run once) | 26/187 = 14% (10% to 20%) | 18/458 = 4% (3% to 6%) |
| Too fast (36) | No rule: peak and mean speed against the batch found none of the 17 on dev | | |

- Smoothness metrics (counting stops, counting movement segments, jerk) all point the wrong way on RoboMIND (dev AUC 0.31 to 0.45): episodes labeled "unsmooth" have fewer stops and segments than episodes rejected for other reasons. Other open-source tools have warned about this (hflow does not use smoothness to reject; an audit of SAL and TED found they mostly measure episode length); we measured it on real QA labels. So the unsmooth-motion rule only looks at duration; it flags none of the 444 successful test episodes.
- Not every episode is comparable: 162 of the 284 regrasp episodes (the Tiangong simulation uses dexterous hands with no gripper, and small tasks lack 5 reference episodes), 187 of the 237 unsmooth test episodes.

### What we can and cannot claim

- We can say: on real data recorded on real robots, jianjin finds bad episodes, says why, and every number has a denominator.
- We cannot say that policies trained on jianjin-filtered data perform better. For that we only cite published work (CUPID, CoRL 2025; DemInf, RSS 2025) until a pilot with a robot or a collection site.
- False positives remain: neatly stacked cubes and long occlusions fool the video models. Of the 18 RoboVAD episodes both video models rejected, 8 have no anomaly label; we have not watched them, so we cannot tell whether the models or the labels are wrong.
- Motion quality is only partly covered. In RoboMIND, the big classes are unsmooth motion, regrasp, unwanted contact and collision before grasp; the task itself was usually completed. jianjin has a rule aimed at 954 of the 1,678 rejections (57%), but measured detection is far apart: regrasp 61%, unsmooth motion 14%. 656 (39%) need vision, mostly contact and collision.

## Nebius and NVIDIA

`jianjin check --summary` adds a batch summary at the top of the report, written by NVIDIA Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`, open weights under OpenMDW-1.1) through the Nebius Token Factory OpenAI-compatible API.

- The model only sees statistics jianjin computed: counts, problem names, episode ids. It never sees joint data or video.
- It writes prose with placeholders such as `{n_doubt}`, and jianjin fills in the numbers. Any digit or number word the model writes on its own is rejected; jianjin retries once and then falls back to a template. The report says which one you are reading.
- Reasoning is turned off with `chat_template_kwargs: {"enable_thinking": false}`. With it on, the reasoning trace landed in the answer text and the answer was cut off at the token limit.
- Answers are cached by their inputs, so re-rendering a report makes no new call.

```bash
export NEBIUS_API_KEY=...   # set it in your own shell; never commit it
uv run jianjin check <dataset dir> --out report.html --summary
```

Without the key, the template is used.

## Quick start

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run jianjin check <dataset dir> --out report.html --json findings.json
```

It reads LeRobot v3, RoboMIND HDF5 and RoboVAD directories. Without `--judges`, only the joint checks run, on CPU.

The video judges need a GPU: run the notebooks in `notebooks/` on Colab, put the cache files they produce in `cache/`, then:

```bash
uv run jianjin check <dataset dir> --judges robometer,topreward --camera observation.images.top --out report.html
```

When the video is on disk, the report adds keyframes to DOUBT and REJECT cards; `--no-frames` turns that off.

Exams and tests:

```bash
uv run --extra hub jianjin fetch exam_a     # public data for exam A; joint data only by default
uv run jianjin eval a --split dev            # exam A on dev; --split test for the test set
uv run jianjin eval b-episode --split train
uv run jianjin eval b-regrasp --split train  # regrasp; on test add --dev-result results/exam_b_regrasp_train.json
uv run --extra hub jianjin fetch robomind   # RoboMIND rejections, joint data only: 1.75 GB read, 253 MB stored; accept the dataset terms on Hugging Face and log in first
uv run jianjin eval c --split dev            # exam C; on test add --split test --dev-result results/exam_c_dev.json
uv run jianjin eval c-regrasp                # the regrasp rule on RoboMIND, run once
uv run pytest
```

## Next

- TOPReward's progress-drop check is unstable on the first frames; re-tune it on train without the opening.
- Problems that need vision (unwanted contact, collision before grasp, object displaced: 620 RoboMIND episodes) need a video model that can answer specific questions.
- Unsmooth motion is only caught 14% of the time and "too fast" shows no signal; we first need to learn what the reviewers look at.

## Layout

```
src/jianjin/
  ingest/       read LeRobot v3, RoboMIND, RoboVAD
  checks/       joint and metadata rules; video.py holds the video rules
  judges/       video judge adapters (Robometer, TOPReward) and their cache
  diagnose.py   merge findings into a verdict
  report.py     offline HTML report; keyframes.py adds keyframes
  summary.py    batch summary written by Nemotron, numbers filled in by jianjin
  eval/         the three exams
notebooks/      Colab notebooks for the video judges
configs/        datasets, splits and licenses used by the exams
results/        exam results (JSON, aggregates only)
docs/           design docs, plans, background notes (Chinese)
```

## Licenses

Code: Apache-2.0, see [LICENSE](LICENSE).

Models:

| Model | Used for | License |
|---|---|---|
| Robometer-4B (`lerobot/Robometer-4B`) | Video progress judge | Apache-2.0 |
| Qwen3-VL-4B-Instruct (used by the TOPReward method) | Second video judge, wrong task only | Apache-2.0 |
| LeRobot 0.6.1 | Loads the two models above; installed on Colab only | Apache-2.0 |
| Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`, via Nebius Token Factory) | Batch summary with `--summary`; optional | OpenMDW-1.1, open weights |

No data ships with the repository: `data/`, `cache/` and `reports/` are not committed.

| Dataset | Used for | License |
|---|---|---|
| `lerobot/svla_so100_pickplace`, `lerobot/svla_so100_stacking` | Exam A dev | Apache-2.0 |
| `jun-seo/bimanual_tilt`, `armnet/armnetbench_v01_lerobot_bimanual_so101` (without its towel-folding task) | Exam A dev, bimanual, joint data only | Apache-2.0 |
| `lerobot/svla_so100_sorting`, `lerobot/svla_so101_pickplace`, `CoRL2026-CSI/SO101-DualArm-teleop_fold_towel_100epi` | Exam A test | Apache-2.0 |
| RoboVAD (Zenodo) | Exam B, evaluation only | Stated three ways (CC BY-NC-ND 4.0, CC BY 4.0, CC BY-SA 4.0); we follow the strictest, CC BY-NC-ND 4.0: no derivatives, no commercial use, no redistribution. Only aggregate results are in this repository |
| RoboMIND (Hugging Face) | Exam C: 1,677 episodes of `failure_data/`, joint data only (253 MB locally) | Apache-2.0; downloading requires agreeing to share contact information |
