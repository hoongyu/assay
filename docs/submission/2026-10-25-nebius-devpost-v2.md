# Devpost submission draft v2: Nebius × NVIDIA Global AI Hackathon, Physical AI track

Draft for the user to paste into Devpost; the user saves the draft and submits. Official deadline: 2026-10-30, 10:00 Pacific (18:00 Berlin); internal target: submit by 10-28. Items marked [TBD] wait for the Super VL evaluation (`results/super_vl_review.json`). Do not select a city prize (we did not attend a city event). Replaces the 2026-10-05 draft.

---

**Project name:** Assay

**Tagline:** Automatic QA for robot teleoperation data. The joints say when; Nemotron says what.

## 30-second judge briefing

| Criterion | Where to look |
|---|---|
| Technological implementation | Nemotron 3.5 Super VL on Nebius Token Factory reviews every flagged episode: the joint rules pick the moment and write one question, the model watches that clip (sent as `video_url`) and must answer from a fixed list. Nemotron 3.5 Lightning writes the batch summary with placeholders; Assay fills in every number. Measured: [TBD Super VL score with denominators]. |
| Design | One command, one offline HTML report in English or Chinese: verdict funnel, per-operator statistics, and one card per flagged episode with the reason, time span, keyframes and the model's review. Browser demo: hoongyu.github.io/assay-demo. |
| Potential impact | Robot data is now produced in collection sites with 100+ robots each; QA is still mostly manual. On five public datasets, 6 of 308 episodes Assay rejected were all unusable. On robomimic, removing the episodes Assay flags beats removing the same number at random (36% vs 21% policy success). |
| Quality of the idea | The vision model never decides alone: rules say when, the model says what, a rejection still needs two independent sources, and every number in the report comes from code. |

## Inspiration

Robot foundation models are trained on teleoperation demonstrations, and in China this data is now produced in dedicated collection sites: at least 15 large ones, typically with a hundred or more robots each. Beijing Humanoid's site sells its data at 500 to 1,000 CNY per hour. QA and labeling still rely largely on people, and the head of one large site estimated that of 100 hours recorded, maybe 50 come out usable. We wanted a QA tool that a site can run on its own machines, because its data is its product, and that tells operators what they did wrong, not just which episodes were thrown away.

The problem is not only Chinese. On five public LeRobot datasets on Hugging Face, published for training with no episode marked as failed, Assay rejected 6 of 308 episodes, and all 6 were unusable: recordings under one second, an arm that never moves, a person resetting the scene. Anyone training on community datasets inherits these episodes.

## What it does

`assay check <dataset>` reads a batch of episodes (LeRobot v2.x and v3, robomimic, RoboMIND, RoboVAD) and writes a single offline HTML report:

- Every episode is judged PASS, DOUBT or REJECT. Rejection is conservative: an objective hard rule (missing readings, dropped frames, too short), or two independent sources pointing at the same problem.
- The joints say when. Joint rules on the CPU find the moment: a gripper that loses its object and grasps again, a recording that stops while the arm still moves, footage that runs longer than the joint data.
- Nemotron says what. With `--review`, each flagged episode gets one question generated from its rule, for example "Around 1.5 s and 2.1 s into this clip, does a robot gripper lose its grip on the object and grasp it again?" Nemotron 3.5 Super VL on Nebius Token Factory watches exactly that clip and answers from a fixed list (yes, no, unclear, or what the extra footage shows). An answer outside the list is asked once more, then recorded as unclear. The answer goes on the episode's card; it never changes the verdict on its own.
- With `--summary`, Nemotron 3.5 Lightning on Token Factory writes a short batch summary for the QA lead and the operators, in English and Chinese. It writes placeholders; Assay fills in every number, and the tips come only from a fixed list.

## How we built it

- Joint rules on CPU: integrity checks, long pauses, empty recordings, unfinished end poses, extra motion, regrasps (gripper open and close cycles compared with the same task's batch) and unusually long episodes.
- Video review (`assay/review.py`, standard library only, so the browser demo can use it too): one request per flagged episode, built from the finding: the two closest gripper switches for a regrasp, the extra footage for a length mismatch, the whole episode for a task the video judges doubt. The clip is cut with PyAV, re-encoded at 10 fps with the long side at 448 px, and sent as a base64 mp4 in a `video_url` part to Token Factory's OpenAI-compatible endpoint. Answers are cached by request content.
- Two zero-shot video judges, Robometer-4B and TOPReward on Qwen3-VL-4B, run on Colab GPUs with cached results.
- Evaluation on real data with thresholds chosen on dev sets, every number with its denominator, second looks at a test set disclosed.

## Results

- Public demonstration datasets (5 LeRobot datasets, 308 episodes): 6 rejected, all 6 real problems. 33 sent to a person: 12 real problems, 12 false positives, 9 not checked.
- Nemotron 3.5 Super VL review of the 39 flagged episodes (29 with local video, human labels in `configs/public_review_labels.json`): [TBD: of the 12 false positives, cleared X with video input and X' with 8 keyframes; of the N real problems it could see, agreed with Y; what it says the extra footage shows in the 9 length mismatches; tokens and cost].
- Downstream training on robomimic Square MH (BC-RNN, 3 seeds, 150 rollouts per condition): removing the 45 episodes Assay flags gives 36% success, removing 45 random ones 21%, keeping all 300 gives 39%. Filtering picks the right episodes but does not beat using everything at 600 epochs; we report both.
- RoboMIND, 1,678 episodes rejected by Beijing Humanoid's QA: the regrasp rule, tuned on another dataset and not retuned, flags 99 of the 162 regrasp rejections it can compare.
- Injected defects on held-out datasets: cut endings 197/201 and frozen frames 199/201, with 0/202 clean episodes flagged (second look at the test set, disclosed).

## Challenges we ran into

- Video or keyframes? Our probe on one clip: 8 keyframes cost 2,450 prompt tokens and missed the regrasp; the 4.5 s clip as video cost 881 and got it right. One example is not a result, so the evaluation asks both ways [TBD: which input won].
- Smoothness metrics did not work on real QA labels (AUC 0.31 to 0.45 on dev, near chance on test). We kept a duration rule, which catches 14%, and say so.
- With reasoning on (the default), Nemotron's reasoning trace came back inside the answer text and ran into the token limit; `enable_thinking: false` fixed it.

## Accomplishments that we're proud of

- A vision model that is used where it adds information: it answers a question the joints cannot, about a moment the joints found, and it cannot write its own numbers into the report.
- A regrasp rule chosen on one benchmark that still finds 99 of 162 comparable regrasps in another lab's QA labels.
- Honest evaluation, including the downstream result that filtering did not beat using all data.

## What we learned

- Send the clip, not a handful of frames, when the question is about motion; it was also cheaper per second of footage on Token Factory.
- On real QA labels, "obvious" motion-quality metrics can score below chance; measure before shipping.

## What's next

- A pilot with one collection site, to measure how much review time Assay saves and whether per-operator feedback lowers the rejection rate.
- More review questions for the problems joints cannot see: unwanted contact, collision before grasp, objects knocked away.

## Built with

Python, NumPy, h5py, PyArrow, PyAV, Nebius Token Factory, NVIDIA Nemotron 3.5 Super VL, NVIDIA Nemotron 3.5 Lightning, Robometer-4B, Qwen3-VL-4B, LeRobot, robomimic, robosuite, Hugging Face Hub, Google Colab, Pyodide.

## Feedback on Nebius and NVIDIA tools

**What we used:** Nebius Token Factory's OpenAI-compatible API with NVIDIA Nemotron 3.5 Super VL (video review of flagged episodes) and Nemotron 3.5 Lightning (batch summary), both at runtime.

**What worked well:**
- `video_url` with a base64 mp4 works on Token Factory for Nemotron 3.5 Super VL, though only `image_url` is documented. Our 4.5 s clip cost 881 prompt tokens, less than 8 keyframes (2,450).
- Latency was about one second per call for short clips, fast enough to review a whole batch while the report is generated.
- The OpenAI-compatible endpoint needs no SDK: the request is a few lines of standard-library Python, and the same request body works from a browser.
- Open weights (OpenMDW-1.1) matter for our users: a collection site can later run the same models on its own machines.

**What could be better:**
- Document `video_url` for the vision models: accepted formats, maximum length, how frames are sampled, and how many tokens a second of video or one image costs. We had to measure it.
- The vision docs still use a model that is no longer in the catalog (Qwen2-VL-72B); an example with Nemotron 3.5 Super VL would save time.
- The `chat_template_kwargs: {"enable_thinking": false}` switch is only on the Hugging Face model card; with reasoning on, the trace comes back inside `content` with no tags.
- No Cosmos model on serverless since August; for a Physical AI track a serverless Cosmos Reason would be welcome next to Super VL.

**Onboarding:** The first call worked as soon as the key was set; the time went into reasoning traces and finding out that video works.

**Would we use it again:** Yes, for both text and video.
